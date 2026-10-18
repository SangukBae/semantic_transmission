"""BSU-Gen: source video -> Qwen -> NTSCC/AWGN -> Open-Sora -> BSU refinement.

No cost router, DCVC or RIFE is dispatched. This registers a new standalone
protocol using the existing generative algorithms, without relabeling old runs.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess

from .artifacts import sha256, write_json
from .webvid5 import fingerprint

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/bsu_gen.json'
MODULE = 'semantic_transmission.bsu_gen'
POLICY = 'bsu_gen_v1'


def read(path):
    return json.loads(Path(path).read_text())


def profile():
    value = read(CONFIG)
    expected = dict(name='BSU-Gen', policy=POLICY, route='generative_only',
        candidate_stride=4, visual_codec='NTSCC', generator='Open-Sora v1.2 / STDiT3-XL/2',
        receiver_mode='combined_context', width=576, height=320, fps=24,
        snr_db=10, channel_seed=42, generation_seed=2025, steps=30,
        extra_rate_ratio=.05, predictive_fallback=False, adaptive_route_selection=False)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('BSU-Gen fixed profile changed; register a new policy')
    return value


def code_identity():
    from . import tail_reference, precision_text_cache, bsu_webvid
    names = ('bsu_gen', 'bsu_receiver', 'receiver_key_context', 'bsu_kinetics_base',
        'qwen35_pairs_video', 'qwen35_pairs_comparison', 'qwen35_semantic_selection',
        'qwen35_feedback', 'qwen35_state_delta', 'qwen35_event_pairs',
        'qwen35_feedback_eval', 'qwen35_backend', 'qwen35_caption_adapter',
        'auto_extraction', 'assisted_captions', 'fc_lgvsc_qwen35', 'hybrid_reconstruction',
        'condition_collision', 'short_schedule', 'decoder_memory', 'generation_noise',
        'trajectory_receiver', 'webvid_ablation', 'artifacts',
        'codec_transport', 'workers', 'etri_60s', 'etri_60s_check', 'research_quality',
        'semantic_clips', 'temporal', 'decoder_runner', 'video_io', 'webvid5')
    files = set(tail_reference.CODE) | set(precision_text_cache.CODE)
    files.update('src/semantic_transmission/' + n + '.py' for n in names)
    files.update('src/semantic_transmission/' + n for n in bsu_webvid.CODE)
    files.update('scripts/' + n for n in ('run_bsu_gen.sh', 'qwen35_caption.py',
        'qwen35_advanced_caption.py', 'etri_schedule_fix_decoder.py',
        'etri_condition_fix_decoder.py', 'etri_precision_decoder.py'))
    files.update(('configs/bsu_gen.json', 'configs/lgvsc_variable.json'))
    for vendor in ('NTSCC_JSAC22',):
        root = REPO / '.local/vendor' / vendor
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.update(str(p.relative_to(REPO)) for p in root.rglob('*.py'))
    return {p: sha256(REPO / p) for p in sorted(files)}


def runtime_template(value):
    """Build from installed assets and public recipes, not a historical experiment."""
    from . import auto_extraction as ae
    from .cli import settings
    from .qwen35_backend import MODEL_ID, REVISION, SNAPSHOT
    from .qwen35_caption_adapter import default_adapter
    local = settings(REPO)
    interpreters = dict(qwen=str(REPO / '.local/qwen35_env/bin/python'),
                        **{k: local[k] for k in ('python', 'channel_python')})
    for path in interpreters.values():
        if not Path(path).is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f'required interpreter: {path}')
    models = read(REPO / '.local/model_paths.json')
    installed = {}
    for name in ('stdit', 'vae', 't5', 'vae2d'):
        directory = Path(models[name])
        files = [p for p in directory.rglob('*') if p.is_file()]
        if not files or not any(p.suffix in ('.bin', '.safetensors', '.pt', '.pth') for p in files):
            raise FileNotFoundError(f'model weights missing: {name}: {directory}')
        # Existing generation workers validate these files against this frozen
        # installation identity. This stat inventory is not a weight-hash audit.
        installed.update({str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                          for p in files})
    manifest_path = REPO / 'outputs/qwen35_setup/model_manifest.json'
    manifest = read(manifest_path)
    if (manifest['model_id'], manifest['revision']) != (MODEL_ID, REVISION):
        raise ValueError('Qwen setup manifest differs from the pinned model')
    base_files = {}
    for item in manifest['files']:
        path = SNAPSHOT / item['file']; stat = path.stat()
        if stat.st_size != item['size']:
            raise ValueError(f'incomplete Qwen file: {path}')
        base_files[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=item['sha256'])
    assets = {}
    for name in ('ntscc_hyperprior_quality_4_psnr.pth', 'unimatch.pth'):
        path = REPO / '.local/checkpoints' / name
        assets[str(path)] = sha256(path)
    if not (REPO / '.local/qwen35_training_probe_deps/peft/__init__.py').is_file():
        raise FileNotFoundError('pinned caption adapter requires local PEFT')
    cfg = read(REPO / 'configs/lgvsc_variable.json')
    cfg.update(profile=POLICY, seed=value['generation_seed'], channel_seed=value['channel_seed'],
        snr_db=value['snr_db'], steps=value['steps'],
        visual_channel='source_frame_seeded_numpy_awgn_v1',
        benchmark_digital_bits_per_symbol=4, concatenation_policy='endpoint_exact',
        cpu_video_storage=True, semantic_clip_policy='frame_exact',
        caption_provider='local_qwen35_caption_repair', models=models)
    return dict(policy=POLICY, receiver=cfg, interpreters=interpreters,
        caption=dict(backend='qwen35', model_id=MODEL_ID, model_revision=REVISION,
            model_path=str(SNAPSHOT), quantization='nf4', max_pixels=262144,
            caption_max_words=79, caption_adapter=default_adapter(),
            prompts=dict(caption=ae.CAPTION_PROMPT, review=ae.REVIEW_PROMPT)),
        base_model_files=base_files, installed_model_files=installed, asset_sha256=assets,
        qwen_manifest_sha256=sha256(manifest_path), code=code_identity(),
        model_weight_hashes_rechecked=False, model_inference_verified=False)


def validate_prepared(path):
    from .qwen35_pairs_video import validate_source_window
    from .auto_extraction import check_frame
    source = read(path); validate_source_window(source)
    if (source['width'], source['height'], source['fps']) != (576, 320, 24):
        raise ValueError('BSU-Gen requires full-duration 576x320, 24 FPS source')
    if source['frames'] < 2 or source['recipe'].get('source_start_seconds') != 0:
        raise ValueError('source must contain at least two frames starting at zero')
    for name in ('source', 'normalized'):
        key = 'source_path' if name == 'source' else 'normalized_video'
        if sha256(source[key]) != source[name + '_sha256']:
            raise ValueError(f'{name} video changed')
    for i in range(source['frames']):
        check_frame(source, i)
    return source


def preflight(args):
    from .qwen35_pairs_comparison import forbid_overlap
    from .video_io import probe
    value = profile(); output = args.output.resolve()
    source_path = (args.video or args.prepared).resolve(strict=True)
    protected = [source_path, REPO / '.local', REPO / 'src', REPO / 'configs',
                 REPO / 'outputs/qwen35_evidence_repair_v3']
    if args.prepared:
        source = validate_prepared(source_path)
        protected.extend((source_path.parent, source['source_path'], source['source_frames'], source['normalized_video']))
        inputs = dict(prepared=str(source_path), sha256=sha256(source_path))
    else:
        info = probe(source_path)
        if not math.isfinite(info['duration']) or info['duration'] < 2/24:
            raise ValueError('video must have a finite duration of at least two output frames')
        inputs = dict(video=str(source_path), sha256=sha256(source_path), probe=info)
    forbid_overlap(output, protected)
    if output.exists() and any(output.iterdir()) and not (output / 'gen_protocol.json').exists():
        raise ValueError('unrecognized nonempty output; use a new --output')
    binaries = {}
    for name in ('ffmpeg', 'ffprobe'):
        path = shutil.which(name)
        if not path:
            raise FileNotFoundError(name)
        binaries[name] = dict(path=path, sha256=sha256(path))
    result = dict(profile=value, output=str(output), inputs=inputs,
        template=runtime_template(value), binaries=binaries)
    result['signature'] = fingerprint(result)
    prior = output / 'gen_protocol.json'
    if prior.exists() and read(prior) != result:
        raise ValueError('frozen source/configuration/code/assets changed; use a new --output')
    return result


def prepare_source(record):
    """Normalize the entire source; never impose the historical 384-frame cap."""
    from .video_io import probe
    root = Path(record['output']); inputs = record['inputs']
    source = Path(inputs['video'])
    if sha256(source) != inputs['sha256']:
        raise ValueError('source video changed')
    dest = root / 'source'; dest.mkdir()
    video = dest / 'normalized.mp4'; frames = dest / 'source_frames'; frames.mkdir()
    ffmpeg = record['binaries']['ffmpeg']['path']
    subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-threads', '2', '-i', str(source),
        '-map', '0:v:0', '-an', '-vf', 'scale=576:320,fps=24', '-c:v', 'libx264',
        '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-threads', '2', str(video)], check=True)
    subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-threads', '2', '-i', str(video),
        '-map', '0:v:0', '-start_number', '0', '-threads', '2', str(frames / '%d.png')], check=True)
    info = probe(video); count = info['frames']
    result = dict(id='video', dataset='user', source_id=source.stem,
        source_path=str(source), source_sha256=inputs['sha256'],
        source_duration_seconds=inputs['probe']['duration'],
        normalized_video=str(video), normalized_sha256=sha256(video),
        source_frames=str(frames), source_frame_hashes={str(i): sha256(frames / f'{i}.png') for i in range(count)},
        frames=count, fps=24, width=576, height=320,
        recipe=dict(source_start_seconds=0, source_end='EOF', duration_limit=None,
            normalization='scale=576:320,fps=24; libx264 crf18 medium yuv420p; no audio'))
    write_json(dest / 'source_prepared.json', result)
    validate_prepared(dest / 'source_prepared.json')


def make_plan(record, prepared):
    from .qwen35_pairs_video import plan as selection_plan
    source = validate_prepared(prepared); root = Path(record['output'])
    template = root / 'runtime_template.json'
    selection_plan(prepared, root / 'benchmark/snr_10/video/base', template_path=template)
    value = record['profile']
    p = dict(policy=POLICY, model='BSU-Gen', route='generative_only', root=str(root),
        profile=value, gen_protocol_sha256=sha256(root / 'gen_protocol.json'),
        template=str(template), template_sha256=sha256(template),
        videos=[dict(id='video', prepared=str(prepared), prepared_sha256=sha256(prepared),
            source_sha256=source['source_sha256'], frames=source['frames'], fps=24, width=576, height=320)],
        snr_db=[10.], mode=value['receiver_mode'], extra_rate_ratio=value['extra_rate_ratio'],
        interpreters=record['template']['interpreters'], generation_seed=value['generation_seed'],
        channel_seed=value['channel_seed'], steps=value['steps'], reuse_base=None,
        code=record['template']['code'], independent_semantic_review='PENDING')
    p['signature'] = fingerprint(p)
    return p


def verify_plan(plan):
    if (plan.get('policy') != POLICY or plan.get('route') != 'generative_only'
            or plan.get('profile') != profile()
            or fingerprint({k:v for k,v in plan.items() if k != 'signature'}) != plan['signature']):
        raise ValueError('BSU-Gen protocol changed or uses a different route')
    root = Path(plan['root'])
    for path, digest in plan['code'].items():
        if sha256(REPO / path) != digest:
            raise ValueError(f'frozen BSU-Gen implementation changed: {path}')
    if (sha256(root / 'gen_protocol.json') != plan['gen_protocol_sha256']
            or sha256(plan['template']) != plan['template_sha256']):
        raise ValueError('BSU-Gen runtime contract changed')
    template = read(plan['template'])
    for path, expected in {**template.get('installed_model_files', {}),
                           **template.get('base_model_files', {})}.items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected['size'], expected['mtime_ns']):
            raise ValueError(f'frozen model installation changed: {path}')
    for path, digest in template.get('asset_sha256', {}).items():
        if sha256(path) != digest:
            raise ValueError(f'frozen codec or motion weights changed: {path}')
    for video in plan['videos']:
        if sha256(video['prepared']) != video['prepared_sha256']:
            raise ValueError('prepared source description changed')


@contextmanager
def generative_pipeline():
    """Reuse the generative batch implementation with explicit new worker identity."""
    from . import bsu_webvid as legacy
    original_verify, original_command = legacy.verify_plan, legacy.command
    def dispatch(cmd, log, env):
        cmd = list(cmd)
        if cmd[1:3] == ['-m', 'semantic_transmission.bsu_webvid_base']:
            cmd[2] = 'semantic_transmission.bsu_kinetics_base'
        elif cmd[1:3] == ['-m', 'semantic_transmission.bsu_sequence_refinement']:
            cmd[2] = MODULE
            cmd.insert(3, '_worker')
        return original_command(cmd, log, env)
    legacy.verify_plan, legacy.command = verify_plan, dispatch
    try:
        yield
    finally:
        legacy.verify_plan, legacy.command = original_verify, original_command


def worker(stage, job_path):
    from . import bsu_receiver as receiver, bsu_sequence_refinement as legacy
    job = read(job_path); root = job_path.parent
    plan = read(root.parent.parent.parent / 'protocol.json')
    verify_plan(plan)
    if job['benchmark_signature'] != plan['signature'] or job['mode'] != plan['mode']:
        raise ValueError('worker job does not belong to this BSU-Gen protocol')
    if stage == 'prepare':
        receiver.prepare(job, root / 'sender')
    elif stage == 'channel':
        legacy.channel(job, root / 'sender', root / 'received')
    elif stage == 'refine':
        # The pixel restoration API receives no original-frame path or quality
        # labels. Source-based quality evaluation is a separate worker below.
        receiver.refine(dict(base=job['base'], mode=job['mode']), root / 'received', root / 'refined')
    elif stage == 'evaluate':
        legacy.evaluate(job, root / 'refined')
    else:
        raise ValueError('unknown BSU-Gen worker stage')


def execute(record):
    from . import bsu_webvid as legacy
    from .qwen35_pairs_video import freeze
    from .etri_60s_check import lock
    from .webvid_ablation import Stages
    root = Path(record['output']); root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'gen_execution.lock'):
        freeze(root / 'gen_protocol.json', record)
        freeze(root / 'runtime_template.json', record['template'])
        try:
            (root / 'RESULT.json').unlink(missing_ok=True)
            write_json(root / 'status.json', dict(status='RUNNING', model='BSU-Gen', stage='source', completed=0, total=3))
            stages = Stages(root, record['signature'])
            if 'video' in record['inputs']:
                stages.step('source', ['source'], ['source'], lambda _: prepare_source(record))
                prepared = root / 'source/source_prepared.json'
            else:
                prepared = Path(record['inputs']['prepared'])
            plan = make_plan(record, prepared); freeze(root / 'protocol.json', plan)
            verify_plan(plan)
            write_json(root / 'status.json', dict(status='RUNNING', model='BSU-Gen', stage='generation_and_refinement', completed=1, total=3))
            with generative_pipeline():
                condition = root / 'benchmark/snr_10/video'
                result = legacy.load_result(condition, plan)
                if result is None:
                    result = legacy.run_one(plan, plan['videos'][0], 10., condition)
            movie = Path(result['video'])
            output = dict(result, status='COMPLETE_BSU_GEN', model='BSU-Gen', route='generative_only',
                predictive_fallback=False, adaptive_route_selection=False,
                generator=record['profile']['generator'], video_sha256=sha256(movie),
                independent_semantic_review='PENDING', training_executed=False,
                gen_protocol_sha256=sha256(root / 'gen_protocol.json'))
            write_json(root / 'RESULT.json', output)
            write_json(root / 'status.json', dict(status=output['status'], model='BSU-Gen', completed=3, total=3,
                                                video=str(movie), result=str(root / 'RESULT.json')))
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
            return output
        except BaseException as error:
            (root / 'RESULT.json').unlink(missing_ok=True)
            write_json(root / 'status.json', dict(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                model='BSU-Gen', error=str(error), resume='Repeat the same command; completed stages are hash-verified.'))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('profile', 'check', 'run', 'status'), nargs='?', default='run')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--video', type=Path, help='entire video; normalized to 576x320 at 24 FPS')
    source.add_argument('--prepared', type=Path, help='existing full-duration source_prepared.json')
    p.add_argument('--output', type=Path)
    return p


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_worker':
        p = argparse.ArgumentParser()
        p.add_argument('stage', choices=('prepare', 'channel', 'refine', 'evaluate'))
        p.add_argument('--job', type=Path, required=True)
        args = p.parse_args(argv[1:]); return worker(args.stage, args.job.resolve())
    p = parser(); args = p.parse_args(argv)
    if args.action == 'profile':
        print(json.dumps(profile(), ensure_ascii=False, indent=2)); return
    if not args.output:
        p.error('--output is required')
    if args.action == 'status':
        path = args.output / 'status.json'
        print(path.read_text() if path.exists() else json.dumps(dict(status='NOT_STARTED', model='BSU-Gen'))); return
    if not (args.video or args.prepared):
        p.error('--video or --prepared is required')
    record = preflight(args)
    if args.action == 'check':
        print(json.dumps(dict(status='READY_FILES_VERIFIED', profile=record['profile'],
            model_inference_started=False, runtime_imports_and_gpu_inference_verified=False,
            model_weight_hashes_rechecked=False), ensure_ascii=False, indent=2)); return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    execute(record)


if __name__ == '__main__':
    main()
