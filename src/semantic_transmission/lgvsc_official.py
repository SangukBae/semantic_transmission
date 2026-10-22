"""Reproducible, isolated author-code baseline with minimum memory adaptations.

Commands: fetch, check, prepare, rate, run, status. Resume is at verified stage
boundaries. Interrupted stages restart with their partial output archived.
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import signal
import time

from .artifacts import sha256, write_json, runtime_state
from .cli import settings
from .official_release import read, source_path, materialize, fetch
from .webvid_ablation import Stages, environment, exclusive_lock
from .kinetics_assets import event
from .official_reporting import EVALUATION_PROFILE, AUTHOR_KEYS, average_videos, rate_breakdown

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.lgvsc_official'
ROOT = REPO/'outputs/lgvsc_official_webvid55_10db_v2'
PROFILE = REPO/'configs/lgvsc_official_baseline.json'
STAGES = ('prepare', 'select', 'caption', 'flow', 'ntscc', 'channel', 'reconstruct', 'evaluate')


def validate_profile(cfg):
    required = dict(profile='lgvsc_official_table_10db_v2',
        width=576, height=320, fps=24, duration_limit_seconds=16,
        candidate_interval_seconds=.2, threshold=.35, method='key_framesinternvl_diff_0.35',
        selector='skem', steps=30, seed=42, channel_seed=1024, snr_db=10, ntscc_quality=4,
        paper_caption=True, paper_flow=True, official_semantic_clips=True, internvl_8bit=False,
        semantic_clip_policy='official_release', decoder_policy='official_release',
        concatenation_policy='official_release', reference_policy='whole_previous_segment_before_vae',
        condition_collision_repair=False, short_schedule_repair=False, short_reference_padding=False,
        append_actual_eof=False, paper_reproduction=False, max_new_tokens=1024,
        caption_max_new_tokens=256, caption_attention='flash_attention_2',
        evaluation_profile=EVALUATION_PROFILE, flash_attn=True, internvl_offload_layers=3,
        metadata_transport='csv_utf8_ldpc_6144_9216_16qam')
    if any(cfg.get(k) != value for k, value in required.items()):
        raise ValueError('Configuration violates the fixed official baseline; use a separate research runner')
    forbidden = {'caption_provider','caption_adapter','selector_checkpoint','caption_checkpoint',
                 'key_budget','max_gap','conditioning_alignment','generator_backend','visual_codec'}
    if forbidden & cfg.keys():
        raise ValueError('Research override is forbidden in the official baseline')


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def protect(root, inputs):
    root = root.resolve()
    protected = [REPO/p for p in ('.git','.local','src','scripts','configs','tests','docs','data','assets')]
    protected += [Path(p).resolve() for p in inputs]
    if root in (REPO, REPO/'outputs') or REPO.is_relative_to(root):
        raise ValueError('Output must be a separate experiment directory')
    for path in protected:
        if root == path or root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError(f'Output overlaps protected input/code: {path}')
    for parent in root.parents:
        if parent == REPO:
            break
        if (parent/'protocol.json').exists() or (parent/'run_config.json').exists():
            raise ValueError('Output cannot be nested inside another experiment')


def model_inventory(models):
    result = {}
    for role in ('internvl', 'pllava', 'stdit', 'vae', 't5', 'vae2d'):
        folder = Path(models[role])
        config = folder/('vae/config.json' if role == 'vae2d' else 'config.json')
        if not config.is_file():
            raise FileNotFoundError(config)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                stat = path.stat()
                result[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def build_plan(root, inputs, cohort):
    from .video_io import probe
    protect(root, inputs)
    if not inputs or len({p.stem for p in inputs}) != len(inputs):
        raise ValueError('Inputs must have unique video stems')
    cfg = read(PROFILE)
    validate_profile(cfg)
    # Recipe is intentionally not a user-overridable merged config. SNR 0-8
    # requires the author's corresponding weights, which are not released.
    models = read(REPO/'.local/model_paths.json')
    local = settings(REPO)
    for key in ('python','channel_python','internvl_python'):
        if not Path(local[key]).is_file():
            raise FileNotFoundError(local[key])
    sources = read(REPO/'configs/lgvsc_official_sources.json')
    for name in sources['files']:
        source_path(name)
    paths = list((REPO/'src/semantic_transmission').glob('*.py'))
    paths += [PROFILE, REPO/'configs/lgvsc_official_sources.json', REPO/'configs/official_opensora.py',
              REPO/'scripts/run_lgvsc_official.sh', REPO/'scripts/lgvsc_official_session.sh']
    for vendor in ('Open-Sora', 'NTSCC_JSAC22', 'InternVL', 'PLLaVA'):
        folder = REPO/'.local/vendor'/vendor
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths += list(folder.rglob('*.py'))
    code = {str(p): sha256(p) for p in sorted(set(paths)) if '__pycache__' not in p.parts}
    checkpoints = {str(REPO/'.local/checkpoints'/p): sha256(REPO/'.local/checkpoints'/p)
                   for p in ('ntscc_hyperprior_quality_4_psnr.pth','unimatch.pth')}
    videos = [dict(id=p.stem, input=str(p.resolve()), sha256=sha256(p), source_info=probe(p)) for p in inputs]
    plan = dict(profile=cfg, models=models, model_inventory=model_inventory(models),
        checkpoints=checkpoints, code=code, interpreters=local, upstream=sources,
        videos=videos, cohort=cohort, output=str(root),
        scope=dict(paper_reproduction=False, upstream_code_baseline=True,
            metadata='CSV UTF-8 serialization adapter, LDPC(6144,9216)/16-QAM/AWGN 10 dB; no CRC or extra headers',
            cbr='author CBR, measured simulation CBR and estimated semantic-total CBR reported separately',
            evaluation='author same-index MP4 comparison, reference first/last-100 windows, equal video means',
            evaluation_profile=EVALUATION_PROFILE,
            downstream_tasks='not evaluated by this video-quality runner',
            resume='hash-verified completed stages; interrupted stage restarts; no per-frame/model-step resume',
            checkpoint_identity='local public checkpoint pinned; identity to author paper training unverified'))
    plan['signature'] = signature(plan)
    return plan


def validate_plan(plan, root):
    validate_profile(plan['profile'])
    value = dict(plan)
    sig = value.pop('signature')
    if signature(value) != sig or Path(plan['output']) != root:
        raise ValueError('Baseline protocol signature/output changed')
    protect(root, [v['input'] for v in plan['videos']])
    for path, digest in {**plan['code'], **plan['checkpoints'],
                         **{v['input']: v['sha256'] for v in plan['videos']}}.items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen baseline input/code changed: {path}; choose a new output')
    if model_inventory(plan['models']) != plan['model_inventory']:
        raise ValueError('Model inventory changed; choose a new output')
    for name in plan['upstream']['files']:
        source_path(name)


def initialize(dest, plan, video):
    run = dest/'run'
    run.mkdir(exist_ok=True)
    cfg = dict(plan['profile'], input=video['input'], models=plan['models'])
    write_json(run/'request.json', cfg)
    materialize(run/'release')


def stage_files(stage):
    method = 'key_framesinternvl_diff_0.35'
    return {
        'prepare': ['run/data', 'run/run_config.json', 'run/prepare.json'],
        'select': [f'run/data/frames/sample/{method}', 'run/keyframes.json'],
        'caption': ['run/data/clips', 'run/captions.json', 'run/caption_sampling.json'],
        'flow': ['run/metadata_tx.json', 'run/flow_sampling.json'],
        'ntscc': [f'run/data/frames/sample/{method}_10', 'run/ntscc.json'],
        'channel': ['run/received', 'run/metadata_channel.json'],
        'reconstruct': ['run/receiver'],
        'evaluate': ['run/quality.json', 'run/quality_delivered_mp4.csv'],
    }[stage]


def concrete_files(dest, paths):
    # Freeze file names now: later selection/clip/transmission stages create
    # neighbouring files inside data, which must not invalidate preparation.
    result = []
    for name in paths:
        path = dest/name
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        if not files:
            raise ValueError(f'Empty stage artifact: {path}')
        result += [str(p.relative_to(dest)) for p in files]
    return result


def run_one(root, plan, video, mode, on_stage=None):
    from .lgvsc_qwen_baseline import launch
    dest = root/'videos'/video['id']
    dest.mkdir(parents=True, exist_ok=True)
    stages = Stages(dest, plan['signature'])
    init = ['run/request.json', 'run/release']
    stages.step('initialize', init, lambda: concrete_files(dest, init), lambda _: initialize(dest, plan, video))
    limit = 'prepare' if mode == 'prepare' else 'channel' if mode == 'rate' else 'evaluate'
    for stage in STAGES:
        if on_stage:
            on_stage(stage)
        event(stage=f'공식 기준선 {stage}', current=video['id'])
        interpreter = plan['interpreters']['channel_python' if stage == 'channel' else 'python']
        command = [interpreter, '-m', MODULE, '--root', str(root), '--worker', stage, '--video-id', video['id']]
        env = dict(environment(42), PYTHONPATH=str(REPO/'src'),
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   PYTORCH_ALLOC_CONF='expandable_segments:True', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if stage in ('prepare','channel'):
            env['CUDA_VISIBLE_DEVICES'] = '-1'
        owned = stage_files(stage) + [f'resources/{stage}.json', f'resources/{stage}.progress.json']
        required = stage_files(stage) + [f'resources/{stage}.json']
        stages.step(stage, owned, lambda names=required: concrete_files(dest, names),
                    lambda log, cmd=command, e=env, s=stage:
                    launch(cmd, log, e, dest/f'resources/{s}.json', label='공식 LGVSC '+s,
                           decoder_run=dest/'run' if s == 'reconstruct' else None, interval=1))
        if stage == limit:
            break
    return dest


def report_rates(root, plan):
    rows = []
    for video in plan['videos']:
        run = root/'videos'/video['id']/'run'
        ntscc, metadata = read(run/'ntscc.json'), read(run/'metadata_channel.json')
        rows.append(dict(id=video['id'], **rate_breakdown(ntscc, metadata)))
    result = dict(status='COMPLETE_REFERENCE_RATE', signature=plan['signature'], videos=rows,
                  cohort=plan['cohort'], snr_db=plan['profile']['snr_db'],
                  mean_official_reference_cbr=sum(r['official_reference_cbr'] for r in rows)/len(rows),
                  pooled_official_reference_cbr=sum(r['official_reference_cbr']*r['official_denominator'] for r in rows)
                      /sum(r['official_denominator'] for r in rows),
                  mean_measured_simulation_cbr=sum(r['measured_simulation_cbr'] for r in rows)/len(rows),
                  mean_estimated_semantic_total_cbr=sum(r['estimated_semantic_total_cbr'] for r in rows)/len(rows),
                  pooled_measured_simulation_cbr=sum(r['measured_simulation_channel_uses'] for r in rows)
                      /sum(r['normalized_source_denominator'] for r in rows),
                  pooled_estimated_semantic_total_cbr=sum(r['estimated_semantic_total_channel_uses'] for r in rows)
                      /sum(r['normalized_source_denominator'] for r in rows),
                  actual_complete_transmission_cbr=None, scope=plan['scope'])
    write_json(root/'RATE_RESULT.json', result)
    table = dict(model='SKEM+DSA', snr_db=result['snr_db'], cohort=plan['cohort']['name'],
                 videos=len(rows), expected_videos=plan['cohort']['expected_videos'],
                 complete_cohort=plan['cohort']['complete'],
                 mean_official_reference_cbr=result['mean_official_reference_cbr'],
                 mean_measured_simulation_cbr=result['mean_measured_simulation_cbr'],
                 mean_estimated_semantic_total_cbr=result['mean_estimated_semantic_total_cbr'],
                 paper_reproduction=False)
    write_csv(root/'CBR_TABLE.csv', [table])
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_quality(root, plan):
    quality, table = [], []
    for video in plan['videos']:
        result = read(root/'videos'/video['id']/'run/quality.json')
        if result['status'] != 'PASSED' or result['evaluation_profile'] != EVALUATION_PROFILE:
            raise ValueError('Incomplete or incompatible quality evidence')
        quality.append(dict(id=video['id'], **result))
        table.append(dict(name=video['id'], **{k: result['official'][k] for k in AUTHOR_KEYS}))
    average = average_videos(table)
    write_csv(root/'comparison_scores.csv', table + [average])
    return dict(status='COMPLETE_VIDEO_QUALITY', evaluation_profile=EVALUATION_PROFILE,
                cohort=plan['cohort'], aggregation='equal weight per video', average=average, videos=quality)


def execute(root, plan, mode):
    from .etri_60s_check import lock
    state = dict(status='RUNNING', mode=mode, completed=0, total=len(plan['videos']),
                 signature=plan['signature'], resume=plan['scope']['resume'])
    started = time.monotonic()
    try:
        gpu_lock = contextlib.nullcontext() if mode == 'prepare' else lock(REPO/'.local/etri_60s_check.lock')
        with gpu_lock:
            for index, video in enumerate(plan['videos']):
                def update(stage):
                    state.update(current=video['id'], stage=stage, elapsed_seconds=time.monotonic()-started)
                    write_json(root/'status.json', state)
                    event(overall=f'영상 {index}/{len(plan["videos"])} ({100*index/len(plan["videos"]):.1f}%)',
                          eta='전체 완료 예상: 미확정')
                run_one(root, plan, video, mode, update)
                state['completed'] = index+1
                write_json(root/'status.json', state)
        if mode != 'prepare':
            report_rates(root, plan)
        state['status'] = {'prepare':'PREPARED_ONLY', 'rate':'COMPLETE_REFERENCE_RATE',
                           'run':'COMPLETE_RECONSTRUCTION_QUALITY'}[mode]
        if mode == 'run':
            write_json(root/'RESULT.json', dict(status=state['status'], signature=plan['signature'],
                scope=plan['scope'], rate=read(root/'RATE_RESULT.json'),
                quality=report_quality(root, plan)))
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        write_json(root/'status.json', state)
    event(state=state['status'], overall=f'영상 {state["completed"]}/{state["total"]} (100%)')


def resolve_inputs(args):
    if args.input:
        if args.cohort or args.allow_incomplete_cohort:
            raise ValueError('--input is a custom cohort; do not combine with --cohort/--allow-incomplete-cohort')
        paths = [p.resolve(strict=True) for p in args.input]
        return paths, dict(name='custom', expected_videos=len(paths), available_videos=len(paths),
                           complete=False, missing_files=[], identity='user inputs; not a complete paper cohort')
    name = args.cohort or 'webvid55'
    manifest_name, dataset, expected = {
        'webvid55': ('docs/eval_manifest_webvid.csv', 'webvid55', 55),
        'kinetics14': ('docs/eval_manifest_kinetics.csv', 'kinetics400_14', 14),
    }[name]
    manifest = source_path(manifest_name)
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected or len({r['filename'] for r in rows}) != expected:
        raise ValueError('Pinned cohort manifest has an unexpected count or duplicate files')
    directory = args.input_dir or Path(read(REPO/'.local/datasets.json')[dataset])/'raw'
    paths, missing = [], []
    for row in rows:
        path = directory/row['filename']
        if not path.is_file():
            missing.append(row['filename'])
            continue
        path = path.resolve(strict=True)
        if sha256(path) != row['sha256']:
            raise ValueError(f'Author {name} source hash mismatch: {path}')
        paths.append(path)
    if not paths or (missing and not args.allow_incomplete_cohort):
        raise ValueError(f'{name}: {len(paths)}/{expected} verified raw videos. Missing: '
            + ', '.join(missing)
            + '. Supply the missing originals, or explicitly use --allow-incomplete-cohort for a labeled subset.')
    return paths, dict(name=name, expected_videos=expected, available_videos=len(paths),
        complete=not missing, missing_files=missing, manifest=str(manifest), manifest_sha256=sha256(manifest),
        identity=('WebVid main quality cohort' if name == 'webvid55' else 'Kinetics action cohort; quality only here'),
        source_hashes_verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['fetch','check','prepare','rate','run','status'])
    parser.add_argument('--root', type=Path, help='separate output directory; default depends on cohort')
    parser.add_argument('--cohort', choices=['webvid55', 'kinetics14'],
                        help='author raw-video manifest; default: webvid55')
    parser.add_argument('--allow-incomplete-cohort', action='store_true',
                        help='explicitly run the hash-verified available subset; never label it a full cohort')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--input', type=Path, action='append', help='custom raw MP4; may be repeated')
    inputs.add_argument('--input-dir', type=Path, help='raw-video directory for the selected author cohort')
    parser.add_argument('--worker', choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument('--video-id', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    default_name = 'custom' if args.input else (args.cohort or 'webvid55')
    root = (args.root or REPO/f'outputs/lgvsc_official_{default_name}_10db_v2').resolve()
    if args.command == 'fetch':
        fetch()
        print('Pinned author sources verified. No model inference started.')
        return
    if args.command == 'status':
        print(json.dumps(read(root/'status.json') if (root/'status.json').exists() else {'status':'NOT_STARTED'},
                         ensure_ascii=False, indent=2))
        return
    if args.worker:
        from .official_baseline_workers import worker
        plan = read(root/'protocol.json')
        validate_plan(plan, root)
        video = next(v for v in plan['videos'] if v['id'] == args.video_id)
        run = root/'videos'/video['id']/'run'
        request = dict(plan['profile'], input=video['input'], models=plan['models'])
        if read(run/'request.json') != request:
            raise ValueError('Worker request differs from frozen protocol')
        if args.worker != 'prepare':
            cfg = read(run/'run_config.json')
            validate_profile(cfg)
            if any(cfg.get(k) != v for k,v in request.items()):
                raise ValueError('Resolved configuration changed a frozen setting')
        worker(args.worker, REPO, run)
        return
    if not args.command:
        parser.error('Choose check, prepare, rate, run, status or fetch')
    if (root/'protocol.json').exists():
        plan = read(root/'protocol.json')
        validate_plan(plan, root)
        if args.input or args.input_dir or args.cohort or args.allow_incomplete_cohort:
            inputs, _ = resolve_inputs(args)
            if [str(p) for p in inputs] != [v['input'] for v in plan['videos']]:
                raise ValueError('Resume inputs differ from frozen protocol')
    else:
        try:
            inputs, cohort = resolve_inputs(args)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        plan = build_plan(root, inputs, cohort)
    if args.command == 'check':
        print(json.dumps(dict(status='PREFLIGHT_ONLY', videos=len(plan['videos']), output=str(root),
                              cohort=plan['cohort'], upstream_commit=plan['upstream']['commit'],
                              scope=plan['scope']), ensure_ascii=False, indent=2))
        return
    protect(root, [v['input'] for v in plan['videos']])
    if root.exists() and not (root/'protocol.json').exists() and any(p.name != 'run.lock' for p in root.iterdir()):
        raise ValueError('Refusing to adopt a nonempty output directory without this protocol')
    root.mkdir(parents=True, exist_ok=True)
    def interrupted(*_):
        raise KeyboardInterrupt('Stopped by signal; completed stages are reusable')
    signal.signal(signal.SIGTERM, interrupted)
    with exclusive_lock(root/'run.lock'):
        if (root/'protocol.json').exists():
            if read(root/'protocol.json') != plan:
                raise ValueError('Protocol changed while waiting for lock')
        else:
            # If environment inspection fails, no adoptable protocol is left.
            env_record = runtime_state(REPO, plan['interpreters'])
            write_json(root/'protocol.json', plan)
            write_json(root/'environment.json', env_record)
        if not (root/'environment.json').is_file():
            raise ValueError('Missing environment provenance; use a new output')
        execute(root, plan, args.command)


if __name__ == '__main__':
    main()
