"""Matched local LGVSC/BSU Kinetics-14 comparison; historical BSU files are read-only.

LGVSC uses InternVL2 SKEM/PSSS .35 and PLLaVA, with common full-source
normalization, transport, and corrected base Open-Sora execution. This is not
the unmodified release: EOF coverage, frame-exact clips and the shared decoder
repairs are explicit. BSU trajectory/cadence/key-context refinement is excluded.
"""
import argparse
import csv
import json
from pathlib import Path
import signal
import time

from . import bsu_kinetics as bsu, bsu_webvid as batch
from . import kinetics_timesformer as action, lgvsc_qwen_baseline as legacy
from .artifacts import sha256, write_json
from .cli import settings
from .etri_60s_check import lock
from .kinetics_assets import event
from .qwen35_pairs_comparison import forbid_overlap
from .qwen35_pairs_video import freeze, validate_source_window
from .transmission_accounting import channel_breakdown
from .webvid_ablation import Stages, environment
from .wire import unpack

REPO = batch.REPO
ROOT = REPO / 'outputs/lgvsc_kinetics_matched'
MODULE = 'semantic_transmission.lgvsc_kinetics'
POLICY = 'lgvsc_kinetics14_matched_bsu_v1'
read = batch.read
COMMON = ('width', 'height', 'fps', 'steps', 'seed', 'channel_seed', 'snr_db',
          'models', 'flash_attn', 'decoder_policy', 'concatenation_policy',
          'semantic_clip_policy', 'visual_channel', 'evaluation_profile')


def candidates(frames, fps=24):
    if frames < 2 or fps != 24:
        raise ValueError('expected a complete 24fps video')
    return sorted(set(range(0, frames, int(.2 * fps))) | {frames - 1})


def config(shared, source, dest):
    cfg = read(REPO / 'configs/etri_official.json')
    cfg.pop('source_video', None)
    cfg.update({k: shared[k] for k in COMMON})
    cfg.update(profile=POLICY, frames=source['frames'], max_frames=source['frames'],
        input=source['normalized_video'], input_sha256=source['normalized_sha256'],
        preserve_input=True, official_preprocessing=False, selection_stride=4,
        official_semantic_clips=True, paper_caption=True, paper_flow=True,
        benchmark_digital_bits_per_symbol=4, cpu_video_storage=True,
        internvl_offload_layers=3, internvl_compact_kv_cache=True,
        selector_checkpoint=str(dest / 'checkpoints/selector.json'),
        caption_checkpoint=str(dest / 'checkpoints/caption.json'))
    if (cfg['snr_db'], cfg['semantic_clip_policy'], cfg['concatenation_policy']) != (10, 'frame_exact', 'endpoint_exact'):
        raise ValueError('unsupported BSU input/transport/timeline policy')
    return cfg


def protect(root, bsu_root):
    forbid_overlap(root, [bsu_root, *[REPO / p for p in
        ('.local', '.git', 'data', 'assets', 'src', 'scripts', 'configs', 'tests', 'docs')]])


def model_inventory(models):
    result = {}
    for folder in models.values():
        folder = Path(folder)
        if not (folder / 'config.json').is_file() and not (folder / 'vae/config.json').is_file():
            raise FileNotFoundError(folder / 'config.json')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                stat = path.stat()
                result[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def verify_bsu(root):
    plan = read(root / 'benchmark/protocol.json'); batch.verify_plan(plan)
    result = read(root / 'RESULT.json')
    if (result['status'] != 'COMPLETE_RECONSTRUCTION_QUALITY_ACTION_EVALUATION'
            or result['signature'] != plan['signature'] or plan['mode'] != 'combined_context'
            or plan['snr_db'] != [10.] or len(plan['videos']) != 14):
        raise ValueError('a complete 14-video, 10dB combined_context BSU run is required')
    sources = bsu.originals(Path(read(plan['inventory'])['videos'][0]['source_path']).parent,
                            Path(plan['manifest']))
    if [(v['id'], v['label']) for v in plan['videos']] != [(v['id'], v['label']) for v in sources]:
        raise ValueError('BSU cohort differs from the exact author manifest')
    items = []
    for n, video in enumerate(plan['videos'], 1):
        event(current=video['id'], stage=f'BSU 입력·완료 산출물 검증 {n-1}/14 ({100*(n-1)/14:.1f}%)')
        if sha256(video['prepared']) != video['prepared_sha256']:
            raise ValueError('prepared input receipt changed')
        source = read(video['prepared']); validate_source_window(source)
        if ((source['frames'], source['fps'], source['width'], source['height']) !=
                (video['frames'], 24, 576, 320) or sha256(source['normalized_video']) != source['normalized_sha256']):
            raise ValueError('BSU source geometry/bytes changed')
        if set(source['source_frame_hashes']) != {str(i) for i in range(source['frames'])}:
            raise ValueError('incomplete BSU source frame inventory')
        for i, digest in source['source_frame_hashes'].items():
            if sha256(Path(source['source_frames']) / f'{i}.png') != digest:
                raise ValueError('BSU source frame changed')
        item = batch.load_result(root / 'benchmark/snr_10' / video['id'], plan)
        if not item or (item['id'], item['frames'], item['snr_db'], item['mode']) != (
                video['id'], video['frames'], 10, 'combined_context'):
            raise ValueError('missing or mismatched BSU comparison input')
        items.append(item)
    if batch.aggregate(items) != result['quality']:
        raise ValueError('BSU aggregate differs from the per-video evidence')
    evaluated = action.validate_evaluation(read(root / 'action_job.json'))
    if evaluated['summary'] != result['action'] or sha256(root / 'action/RESULT.json') != result['action_result_sha256']:
        raise ValueError('BSU action aggregate changed')
    return plan, result, items


def build_plan(root, bsu_root):
    protect(root, bsu_root)
    plan, result, items = verify_bsu(bsu_root)
    shared = read(plan['template'])['receiver']
    local = settings(REPO)
    for name in ('python', 'channel_python', 'internvl_python'):
        if not Path(local[name]).is_file(): raise FileNotFoundError(local[name])
    action.validate_runtime(plan['action_runtime'])
    inputs = {str(bsu_root / p): sha256(bsu_root / p) for p in
              ('RESULT.json', 'benchmark/protocol.json', 'action_job.json', 'action/RESULT.json')}
    checkpoints = {str(REPO / '.local/checkpoints' / n): sha256(REPO / '.local/checkpoints' / n)
                   for n in ('ntscc_hyperprior_quality_4_psnr.pth', 'unimatch.pth')}
    codec_sha = checkpoints[str(REPO / '.local/checkpoints/ntscc_hyperprior_quality_4_psnr.pth')]
    videos = []
    for video, item in zip(plan['videos'], items):
        dest = root / 'videos' / video['id']
        old_run = Path(item['base']) / 'reconstruction/event_pairs/run'
        old_config = read(old_run / 'run_config.json')
        header, _ = unpack((old_run / 'transmitter/metadata.bin').read_bytes())
        if header['checkpoint_sha256'] != codec_sha or any(old_config[k] != shared[k] for k in COMMON):
            raise ValueError('shared NTSCC/receiver configuration differs from the actual BSU run')
        channel = read(old_run / 'channel_accounting.json')
        size = (old_run / 'transmitter/visual.c64').stat().st_size
        if size % 8 or (channel['ldpc_k'], channel['ldpc_n'], channel['bits_per_symbol']) != (6144,9216,4):
            raise ValueError('actual BSU transport does not use the common digital link')
        ledger = channel_breakdown((old_run/'transmitter/metadata.bin').read_bytes(), size//8, header['video'], channel)
        extra = read(bsu_root/'benchmark/snr_10'/video['id']/'received/channel.json')['complex_channel_uses']
        if (ledger != item['base_transmission_ledger'] or ledger['complex_channel_uses']['total'] != item['base_channel_uses']
                or extra != item['trajectory_channel_uses'] or item['total_channel_uses'] != item['base_channel_uses']+extra
                or ledger['cbr']['denominator_source_scalars'] != 3*video['width']*video['height']*video['frames']):
            raise ValueError('actual BSU packets disagree with the saved transmission totals')
        inputs[video['prepared']] = sha256(video['prepared'])
        receipt = bsu_root / 'benchmark/snr_10' / video['id'] / 'RESULT.json'
        inputs[str(receipt)] = sha256(receipt)
        videos.append(dict(video, config=config(shared, read(video['prepared']), dest),
                           bsu_result=str(receipt), bsu_channel_uses=item['total_channel_uses']))
    # Freeze local implementations and vendor code without editing any frozen BSU module.
    paths = list((REPO / 'src/semantic_transmission').glob('*.py'))
    paths += list((REPO / 'scripts').glob('*.py'))
    paths += [REPO / 'scripts/run_lgvsc_kinetics.sh', REPO / 'configs/etri_official.json',
              REPO / 'configs/official_opensora.py', REPO / 'configs/rtx4080_opensora.py', action.CONFIG,
              REPO / '02_semantic_encoder/skem/MLM-keyframe-internvl.py',
              REPO / '04_semantic_decoder/scripts/mydemo_new_align_sh.py']
    for vendor in ('InternVL', 'PLLaVA', 'Open-Sora', 'NTSCC_JSAC22'):
        folder = REPO / '.local/vendor' / vendor
        if not folder.is_dir(): raise FileNotFoundError(folder)
        paths += list(folder.rglob('*.py'))
    value = dict(policy=POLICY, root=str(root), bsu_root=str(bsu_root), inputs=inputs,
        bsu_signature=plan['signature'], videos=videos, snr_db=10,
        interpreters={k:local[k] for k in ('python', 'channel_python', 'internvl_python')},
        action_runtime=plan['action_runtime'], checkpoints=checkpoints,
        model_files=model_inventory(shared['models']),
        code={str(p.relative_to(REPO)):sha256(p) for p in sorted(set(paths))},
        comparison=dict(selector='InternVL2-8B SKEM/PSSS threshold 0.35; BF16',
            caption='PLLaVA-7B, released four-frame caption inference',
            candidates='int(0.2*24)=4 frame grid plus final frame; no key-count budget',
            common='exact BSU normalized bytes and all source frames; NTSCC checkpoint; '
                   'LDPC(6144,9216), 16-QAM; source-frame-seeded AWGN; base Open-Sora weights, seeds and steps',
            common_receiver_repairs='tail17, FP32 T5 cache, endpoint collision/short-schedule repair, '
                                    'memory chunking and endpoint-exact concatenation',
            common_semantic_clips='frame_exact, matching the BSU base runtime',
            excluded_from_lgvsc=['BSU trajectory packets', 'cadence refinement', 'key-context stitching'],
            cost='all visual symbols + serialized metadata/CRC/LDPC padding/parity; BSU trajectory included',
            physical_link_overhead_included=False, exact_paper_reproduction=False))
    value = action.signed(value)
    receipt = root / 'protocol.json'
    if receipt.exists() and read(receipt) != value:
        raise ValueError('frozen comparison inputs/code/models changed; use a new --root')
    if not receipt.exists() and root.exists() and any(p.name != 'execution.lock' for p in root.iterdir()):
        raise ValueError('unrecognized nonempty output; use a new --root')
    return value


def load_plan(root):
    p = read(root / 'protocol.json'); action.verify_signature(p)
    if p['policy'] != POLICY or p['root'] != str(root): raise ValueError('comparison root/policy changed')
    for path, digest in {**p['inputs'], **p['checkpoints']}.items():
        if sha256(path) != digest: raise ValueError('frozen input changed: ' + path)
    for path, digest in p['code'].items():
        if sha256(REPO / path) != digest: raise ValueError('frozen code changed: ' + path)
    for path, entry in p['model_files'].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            raise ValueError('model file changed: ' + path)
    return p


def initialize(dest, video):
    from .hybrid_reconstruction import initialize as init
    from .workers import csv_write
    source = read(video['prepared'])
    init(dest, source, dict(indices=[0, source['frames']-1], selector='NOT_SELECTED'), video['config'])
    run = dest / 'run'; (run / 'keyframes.json').unlink()
    folder = run / 'data/frames/sample'
    csv_write(folder / 'frames.csv', [dict(frame_path=str(folder / f'{i}.png'))
              for i in candidates(source['frames'])], ['frame_path'])
    csv_write(run / 'data/16x24/videos.csv', [dict(path=source['normalized_video'], frame_save_dir=str(folder))],
              ['path', 'frame_save_dir'])


def initialization_files(frames):
    from .hybrid_reconstruction import initialization_products
    return [p for p in initialization_products(frames) if p != 'run/keyframes.json'] + ['run/data/16x24/videos.csv']


def jobs():
    from . import fc_lgvsc_qwen35 as fc
    tail = [(s, products) for s, _, products in fc.job_plan()
            if s not in ('comparison', 'input-audit', 'semantic-clips', 'caption', 'flow', 'send', 'channel')]
    return [('input-audit', ['input_audit.json']),
        ('select', ['keyframes.json', 'selector_runtime.json', 'data/frames/sample/key_framesinternvl_diff_0.35']),
        ('semantic-clips', ['semantic_clips_audit.json', 'data/clips']),
        ('caption', ['captions.json', 'caption_sampling.json', 'caption_provenance.json']),
        ('flow', ['metadata_tx.json', 'flow_sampling.json']),
        ('send', ['transmitter', 'sender_accounting.json']),
        ('channel', ['received', 'channel_accounting.json']), ('rate', ['rate.json']), *tail]


def write_rate(run, video, signature):
    packet = (run / 'transmitter/metadata.bin').read_bytes(); header, _ = unpack(packet)
    size = (run / 'transmitter/visual.c64').stat().st_size
    cfg = video['config']; channel = read(run / 'channel_accounting.json')
    if size % 8 or header['video'] != {k:cfg[k] for k in ('width', 'height', 'fps', 'frames')}:
        raise ValueError('invalid transport size or covered timeline')
    keys = [k['index'] for k in header['keyframes']]
    if (len(keys) < 2 or keys != sorted(set(keys)) or keys[0] != 0 or keys[-1] != cfg['frames']-1
            or not channel['metadata_exact_match'] or channel['bit_errors'] != 0
            or (channel['ldpc_k'], channel['ldpc_n'], channel['bits_per_symbol']) != (6144, 9216, 4)):
        raise ValueError('transport endpoints or digital link differs from BSU')
    ledger = channel_breakdown(packet, size//8, header['video'], channel)
    freeze(run / 'rate.json', dict(status='COMPLETE_TRANSMISSION', signature=signature, id=video['id'],
        frames=cfg['frames'], keyframes=len(header['keyframes']), ledger=ledger,
        total_channel_uses=ledger['complex_channel_uses']['total'],
        cbr_denominator=ledger['cbr']['denominator_source_scalars'], cbr=ledger['cbr']['total'],
        packet_sha256=sha256(run / 'transmitter/metadata.bin'), visual_sha256=sha256(run / 'transmitter/visual.c64')))


def worker(root, video_id, stage):
    from . import workers, codec_transport, etri_60s, fc_lgvsc_qwen35 as fc
    from . import bsu_webvid_base as link, condition_collision as collision
    p = load_plan(root); video = next(v for v in p['videos'] if v['id'] == video_id)
    dest = root / 'videos' / video_id; run = dest / 'run'; cfg = read(run / 'run_config.json')
    if cfg != video['config']: raise ValueError('worker configuration changed')
    if stage in ('select', 'caption', 'flow'):
        getattr(workers, stage)(cfg, REPO, run)
        if stage == 'caption':
            write_json(run / 'caption_provenance.json', dict(provider='PLLaVA-7B', model=cfg['models']['pllava'],
                inference_executed=True, qwen_used=False, clip_policy=cfg['semantic_clip_policy']))
    elif stage in ('send', 'receive'): getattr(codec_transport, stage)(cfg, REPO, run)
    elif stage == 'channel': link.paired_channel(run)
    elif stage == 'rate': write_rate(run, video, p['signature'])
    elif stage in ('input-audit', 'semantic-clips'): etri_60s.worker(stage, run)
    elif stage == 'prepare-text':
        keys = read(run / 'keyframes.json')['indices']
        freeze(dest / 'execution_protocol.json', dict(signature=p['signature'],
            condition_plan=collision.preview(keys, collision.upstream_functions(REPO))))
        fc.prepare_text(run)
        value = read(run / 'receiver_policy.json'); value['default_method'] = POLICY
        write_json(run / 'receiver_policy.json', value)
    elif stage == 'reconstruct': fc.reconstruct(run)
    elif stage == 'output-audit':
        fc.audit(run)
        value = read(run / 'output_audit.json')
        value.update(selector='InternVL2-8B_SKEM_PSSS_0.35', caption_provider='PLLaVA-7B',
            automatic_keyframe_insertion=False, automatic_keyframe_insertion_stage=None,
            full_source_reconstructed=True, bsu_refinement_applied=False)
        write_json(run / 'output_audit.json', value)
    else: raise ValueError('unknown worker stage: ' + stage)


def run_one(root, p, video, stop_after_rate=False, prepare_only=False, on_stage=None):
    dest = root / 'videos' / video['id']; dest.mkdir(parents=True, exist_ok=True)
    stages = Stages(dest, p['signature']); run = dest / 'run'
    stages.step('initialize', ['run'], initialization_files(video['frames']), lambda _: initialize(dest, video))
    if prepare_only: return
    env = dict(environment(video['config']['seed']), PYTHONPATH=str(REPO / 'src'), ETRI_RUN_SIGNATURE=p['signature'])
    for stage, products in jobs():
        event(stage=f'LGVSC {stage} (진행률 미확정)')
        required = ['run/' + f for f in products] + [f'resources/{stage}.json']
        owned = list(required)
        if stage == 'prepare-text': required.append('execution_protocol.json'); owned.append('execution_protocol.json')
        python = p['interpreters']['channel_python' if stage == 'channel' else 'python']
        command = [python, '-m', MODULE, '--root', str(root), '--worker', stage, '--video-id', video['id']]
        if stage == 'evaluate': command = [python, '-m', 'semantic_transmission.research_quality', 'evaluate', str(run)]
        worker_env = dict(env, CUDA_VISIBLE_DEVICES='-1') if stage == 'channel' else env
        stages.step(stage, owned, required, lambda log, cmd=command, e=worker_env, s=stage:
            legacy.launch(cmd, log, e, dest / f'resources/{s}.json', label='LGVSC '+s,
                          decoder_run=run if s == 'reconstruct' else None, interval=1))
        if on_stage: on_stage(stage)
        if stage == 'rate' and stop_after_rate: return read(run / 'rate.json')
    audit = read(run / 'output_audit.json'); quality = read(run / 'quality.json')
    if (audit['status'] != 'PASSED' or quality['status'] != 'PASSED'
            or audit['video_sha256'] != quality['video_sha256']
            or any(quality[b]['frames'] != video['frames'] for b in ('lossless_frames','delivered_mp4'))):
        raise ValueError('LGVSC reconstruction/quality evidence mismatch')
    result = dict(status='COMPLETE_RECONSTRUCTION_QUALITY', signature=p['signature'], id=video['id'],
        frames=video['frames'], video=audit['video'], quality=quality,
        rate=read(run / 'rate.json'), video_sha256=audit['video_sha256'])
    freeze(dest / 'RESULT.json', result)
    return result


def aggregate_rate(p, rows):
    if len(rows) != 14 or [r['id'] for r in rows] != [v['id'] for v in p['videos']]:
        raise ValueError('rate comparison requires all 14 clips in manifest order')
    for v, r in zip(p['videos'], rows):
        if (r['frames'] != v['frames'] or r['cbr_denominator'] != 3*v['width']*v['height']*v['frames']
                or r['total_channel_uses'] <= 0):
            raise ValueError('rate comparison covers different frame counts or invalid symbols')
    denominator = sum(r['cbr_denominator'] for r in rows)
    if denominator != sum(3*v['width']*v['height']*v['frames'] for v in p['videos']):
        raise ValueError('rate comparison covers different frame counts')
    lgvsc = sum(r['total_channel_uses'] for r in rows); bsu_total = sum(v['bsu_channel_uses'] for v in p['videos'])
    if lgvsc <= 0: raise ValueError('invalid LGVSC channel use count')
    return dict(lgvsc_channel_uses=lgvsc, bsu_channel_uses=bsu_total, denominator=denominator,
        lgvsc_cbr=lgvsc/denominator, bsu_cbr=bsu_total/denominator,
        bsu_over_lgvsc=bsu_total/lgvsc, bsu_change_percent=100*(bsu_total/lgvsc-1))


def report_rate(root, p, rows):
    result = dict(status='COMPLETE_RATE_COMPARISON', signature=p['signature'],
                  summary=aggregate_rate(p, rows), videos=rows, scope=p['comparison'])
    freeze(root / 'RATE_RESULT.json', result)
    with (root / 'rate_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id','frames','lgvsc_keys','lgvsc_visual_uses','lgvsc_digital_uses','lgvsc_total_uses',
            'bsu_base_uses','bsu_trajectory_uses','bsu_total_uses','lgvsc_cbr','bsu_cbr','bsu_over_lgvsc'])
        for video, row in zip(p['videos'], rows):
            old = read(video['bsu_result']); uses = row['ledger']['complex_channel_uses']
            writer.writerow([row['id'],row['frames'],row['keyframes'],uses['visual'],uses['digital'],uses['total'],
                old['base_channel_uses'],old['trajectory_channel_uses'],old['total_channel_uses'],row['cbr'],
                old['total_cbr'],old['total_channel_uses']/uses['total']])
    print('REPORT: '+str(root / 'RATE_RESULT.json'), flush=True)
    return result


def quality_summary(rows):
    total = sum(r['frames'] for r in rows)
    return {boundary:{
        'video_mean':{k:sum(r['quality'][boundary][k] for r in rows)/len(rows) for k in batch.KEYS},
        'frame_mean':{k:sum(r['quality'][boundary][k]*r['frames'] for r in rows)/total for k in batch.KEYS}}
        for boundary in ('lossless_frames','delivered_mp4')}


def execute(root, p, mode):
    previous = read(root/'status.json') if (root/'status.json').exists() else {}
    state = dict(status='RUNNING', total=14, transmitted=0, reconstructed=0, action_evaluated=0, mode=mode)
    rates, results = [], []
    try:
        with lock(REPO / '.local/etri_60s_check.lock'):
            for i, video in enumerate(p['videos']):
                event(overall=f'LGVSC 전송 {state["transmitted"]}/14 ({100*state["transmitted"]/14:.1f}%) | '
                              f'복원·화질 {state["reconstructed"]}/14 ({100*state["reconstructed"]/14:.1f}%)',
                      current=f'{video["id"]} ({i+1}/14)')
                write_json(root / 'status.json', dict(state, current=video['id']))
                def completed(stage):
                    state['stage'] = stage
                    if stage == 'rate': state['transmitted'] = i+1
                    if stage == 'evaluate': state['reconstructed'] = i+1
                    write_json(root/'status.json', dict(state, current=video['id']))
                    event(overall=f'LGVSC 전송 {state["transmitted"]}/14 ({100*state["transmitted"]/14:.1f}%) | '
                                  f'복원·화질 {state["reconstructed"]}/14 ({100*state["reconstructed"]/14:.1f}%)')
                item = run_one(root, p, video, stop_after_rate=mode=='rate', prepare_only=mode=='prepare', on_stage=completed)
                if mode == 'prepare': continue
                rates.append(item if mode == 'rate' else item['rate']); state['transmitted'] = i+1
                if mode == 'run': results.append(item); state['reconstructed'] = i+1
                write_json(root / 'status.json', state)
        if mode == 'prepare':
            state['status'] = 'READY_TO_RUN'
            event(overall='입력 준비 14/14 (100.0%)', stage='모델 추론 전 준비 완료', state='준비 완료')
        else:
            rate = report_rate(root, p, rates)
            if mode == 'run':
                from .lgvsc_kinetics_action import make_job, validate_evaluation
                job = make_job(p, results); freeze(root / 'action_job.json', job)
                env = dict(environment(2025), PYTHONPATH=str(REPO/'src'))
                with lock(REPO / '.local/etri_60s_check.lock'):
                    batch.command([p['action_runtime']['python'], '-m', 'semantic_transmission.lgvsc_kinetics_action',
                        '--job', root/'action_job.json'], root/f'logs/action_{time.time_ns()}.log', env)
                evaluated = validate_evaluation(job); old = read(Path(p['bsu_root'])/'RESULT.json')
                q = quality_summary(results)
                comparison = dict(status='COMPLETE_RATE_QUALITY_ACTION_COMPARISON', signature=p['signature'],
                    rate=rate['summary'], quality=dict(lgvsc=q, bsu=old['quality']['refined']),
                    quality_difference_bsu_minus_lgvsc={b:{k:old['quality']['refined'][b]['video_mean'][k]-q[b]['video_mean'][k]
                        for k in batch.KEYS} for b in q},
                    action=dict(old['action'], lgvsc=evaluated['summary']),
                    videos=results, scope=p['comparison'], independent_semantic_review='PENDING')
                freeze(root/'RESULT.json', comparison)
                print('REPORT: '+str(root/'RESULT.json'), flush=True)
                state.update(status='COMPLETE', action_evaluated=14)
            else: state['status'] = 'COMPLETE_RATE_ONLY'
            event(overall='LGVSC 전송 14/14 (100.0%)'+(' | 복원·평가 14/14 (100.0%)' if mode=='run' else ''),
                  stage='비교 결과 저장 완료', state='완료')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
        raise
    finally:
        if state['status'] in ('READY_TO_RUN','COMPLETE_RATE_ONLY') and previous.get('status')=='COMPLETE':
            state = previous
        write_json(root/'status.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='run', choices=('run','rate','prepare','check','status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bsu-root', type=Path, default=bsu.ROOT)
    parser.add_argument('--worker', choices=[s for s,_ in jobs() if s!='evaluate'], help=argparse.SUPPRESS)
    parser.add_argument('--video-id', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.root.resolve(); bsu_root = args.bsu_root.resolve()
    if args.worker:
        if not args.video_id: parser.error('worker needs --video-id')
        return worker(root, args.video_id, args.worker)
    if args.action == 'status':
        print(json.dumps(read(root/'status.json'),ensure_ascii=False,indent=2)); return
    event(overall='BSU와 동일 조건 확인 중', current='Kinetics-400 저자 14편', stage='입력·전송·모델 검증 (진행률 미확정)')
    p = build_plan(root, bsu_root)
    if args.action == 'check':
        print('CHECK: '+json.dumps(dict(status='READY', videos=14, frames=sum(v['frames'] for v in p['videos']),
            snr_db=10, output=str(root), model_inference_started=False, scope=p['comparison']),ensure_ascii=False)); return
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    with lock(root/'execution.lock'):
        freeze(root/'protocol.json', p)
        execute(root, p, args.action)


if __name__ == '__main__': main()
