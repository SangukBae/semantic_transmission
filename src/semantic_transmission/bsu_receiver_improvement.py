"""Frozen local BSU receiver ablation on imported completed WebVid outputs.

Run four policies at 10dB on every completed source. Only a globally qualifying
candidate advances to the remaining 6/8dB conditions. Generation is received-only;
source frames are opened exclusively by the evaluator. Existing results remain
immutable, and every completed generation/evaluation has an artifact receipt.
"""
import argparse
import concurrent.futures
import contextlib
import csv
import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from . import bsu_compare_transfer as transfer, bsu_receiver_compare as common

REPO = common.REPO
ROOT = REPO/'outputs/bsu_receiver_improvement_20261015_v1'
IMPORT = REPO/'outputs/bsu_webvid_completed_20261015'
ARMS = ('combined_context', 'motion_safe_context', 'guarded_motion_context', 'trajectory_warp_context')
CODE = ('receiver_motion_repair.py', 'bsu_receiver_improvement.py', 'receiver_key_context.py',
        'trajectory_receiver.py', 'guarded_trajectory_receiver.py', 'sparse_trajectory.py',
        'bsu_sequence_refinement.py', 'official_quality.py', 'stutter_metrics.py',
        'research_quality.py', 'bsu_receiver.py', 'bsu_compare_transfer.py', 'bsu_receiver_compare.py')
read, event = common.read, common.event
METRICS = ('psnr_db', 'ssim', 'lpips_vgg', 'clip', 'dists')


def case_name(case):
    return f"{case['id']}/snr_{case['snr']:g}"


def folder(root, case, arm):
    return root/'comparison'/case_name(case)/arm


def prepare(root, imported):
    from .receiver_motion_repair import PARAMETERS
    m = transfer.validate_manifest(read(imported/'transfer_manifest.json'))
    receipt = read(imported/'IMPORT_COMPLETE.json')
    if receipt['status'] != 'PASSED' or receipt['manifest_sha256'] != sha256(imported/'transfer_manifest.json'):
        raise ValueError('verified import required')
    event(stage='수신 입력·평가 원본 SHA256 재검증')
    bad = transfer.inspect_files(m, imported/'inputs', common.tick('실험 입력 검증'))
    if bad:
        raise ValueError('imported experiment inputs changed: '+str(bad[:4]))
    sources = {s['id']:s for s in m['sources']}; cases = []
    for c in m['cases']:
        source = sources[c['id']]
        local = lambda name: str(transfer.local_file(imported/'inputs', name)) if name else None
        receiver = dict(frames=c['frames'], fps=24, keys=c['keys'],
            raw_frames=[local(p) for p in c['raw_frames']], current_frames=[local(p) for p in c['current_frames']],
            key_paths=[local(p) for p in c['key_paths']], base_packet=local(c['base_packet']),
            trajectory=local(c['trajectory']))
        cases.append(dict(id=c['id'], snr=c['snr'], frames=c['frames'], keys=c['keys'],
            source_sha256=source['source_sha256'], source_movie=local(source['movie']),
            receiver=receiver, channel_uses=c['base_channel_uses']+c['trajectory_channel_uses']))
    if len(cases) != 108 or len(sources) != 36:
        raise ValueError('this frozen cohort expects 36 sources and 108 completed conditions')
    value = dict(policy='bsu_received_motion_repair_ablation_v1', cases=cases, arms=list(ARMS),
        import_manifest_sha256=receipt['manifest_sha256'], parameters=PARAMETERS,
        code={n:sha256(REPO/'src/semantic_transmission'/n) for n in CODE},
        runtime=common.runtime_identity('cuda'), initial_snr=10,
        criteria=dict(max_psnr_loss_db=.10, max_nonkey_psnr_loss_db=.10,
            max_lpips_relative_increase=.01, max_dists_relative_increase=.01,
            max_flow_relative_increase=.02, max_cadence_relative_increase=.02,
            max_freeze_increase=.005, max_boundary_relative_increase=.05,
            min_psnr_gain_db=.10, min_lpips_relative_gain=.02, min_flow_relative_gain=.05),
        promotion='all guardrails and at least one improvement; lowest LPIPS of passing candidates; '
                  'then 6/8dB on the same 36 sources with the frozen policy',
        criterion_scope='engineering screening, not ETRI acceptance or independent statistics',
        source_only_for_evaluation=True, additional_transmission=0, generator_reused=True,
        training=False, production_defaults_changed=False, independent_semantic_review='PENDING',
        development_scope='All sources already analyzed; no unseen-data or blinded-review claim. '
                          'No per-video parameters or source-based inference decisions.',
        evaluation='full delivered MP4, official five metrics; nonkeys, temporal errors, boundaries; exact PNG keys')
    value['signature'] = transfer.fingerprint(value)
    path = root/'protocol.json'
    if path.exists() and read(path) != value:
        raise ValueError('frozen code/runtime/inputs differ; use a new root')
    if not path.exists():
        write_json(path, value)
        for n in CODE:
            target = root/'frozen_code'/n; target.parent.mkdir(exist_ok=True)
            shutil.copyfile(REPO/'src/semantic_transmission'/n, target)
    return value


def generate(task):
    root_string, case, arm, signature = task
    import cv2
    import numpy as np
    from PIL import Image
    from .bsu_sequence_refinement import decode_packet, validate_keys
    from .receiver_key_context import preserve_context
    from .receiver_motion_repair import adaptive_cadence, warp_to_received_tracks
    from .trajectory_receiver import refine_references
    from .guarded_trajectory_receiver import refine_references as guarded
    from .bsu_receiver import encode_frames
    from .wire import unpack
    dest = folder(Path(root_string), case, arm)
    if common.verify_receipt(dest, 'generation', signature):
        return dict(id=case['id'], snr=case['snr'], arm=arm, reused=True)
    job = case['receiver']
    if set(job) != {'frames','fps','keys','raw_frames','current_frames','key_paths','base_packet','trajectory'}:
        raise ValueError('receiver job must not contain source/evaluation fields')
    cv2.setNumThreads(1); cv2.setRNGSeed(2025)
    validate_keys(job['keys'], job['frames'])
    packet = Path(job['base_packet']).read_bytes(); header, _ = unpack(packet)
    if [k['index'] for k in header['keyframes']] != job['keys'] or header['video']['frames'] != job['frames']:
        raise ValueError('received packet/key timeline differs')
    tracks = decode_packet(Path(job['trajectory']).read_bytes(), packet) if job['trajectory'] else {}
    dest.mkdir(parents=True, exist_ok=True); output = dest/'frames'; output.mkdir(exist_ok=True)
    windows = []; paths = []; started = time.monotonic()
    for index, (a, b) in enumerate(zip(job['keys'], job['keys'][1:])):
        keys = common.pixels(job['key_paths'][index:index+2]); audit = dict(start=a, end=b)
        if arm == 'combined_context':
            current = common.pixels(job['current_frames'][a:b+1])
        else:
            raw = common.pixels(job['raw_frames'][a:b+1])
            track = tracks.get((a, b))
            if arm == 'trajectory_warp_context':
                raw, audit['geometry'] = warp_to_received_tracks(raw, keys, track)
            reference_fn = refine_references if arm == 'motion_safe_context' else guarded
            current, audit['reference'] = reference_fn(raw, keys, track)
            current, audit['cadence'] = adaptive_cadence(current)
        current, audit['context'] = preserve_context(current, keys)
        assert np.array_equal(current[[0, -1]], keys)
        for offset, rgb in enumerate(current):
            path = output/f'{a+offset:05d}.png'
            if index and offset == 0:
                if not np.array_equal(np.asarray(Image.open(path).convert('RGB')), rgb):
                    raise ValueError('shared received key differs')
                continue
            Image.fromarray(rgb).save(path); paths.append(path)
        windows.append(audit)
    movie = dest/'reconstruction.mp4'; timeline = encode_frames(output, movie, job['frames'], job['fps'])
    info = dest/'generation_info.json'
    write_json(info, dict(arm=arm, receiver_reads_source=False, additional_transmission=0,
        frames=job['frames'], received_keys_exact=True, windows=windows,
        timeline=timeline, seconds=time.monotonic()-started))
    common.receipt(dest, 'generation', signature, [*paths, movie, info])
    return dict(id=case['id'], snr=case['snr'], arm=arm, reused=False)


class PersistentMetrics:
    """Exact official batch-one operations; retain model weights between videos."""
    def __init__(self):
        import torch
        import lpips
        import clip
        from DISTS_pytorch import DISTS
        self.torch = torch; self.lpips_package = lpips
        self.lpips = lpips.LPIPS(net='vgg').cuda().eval()
        self.clip, self.preprocess = clip.load('ViT-B/32', device='cuda'); self.clip.eval()
        self.dists = DISTS().cuda().eval()

    def evaluate(self, source, generated, progress=None):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        from .official_quality import pixel_scores, summarize
        if source.shape != generated.shape:
            raise ValueError('evaluation timeline/shape mismatch')
        rows = []
        with self.torch.inference_mode():
            for i, (a, b) in enumerate(zip(source, generated)):
                row = dict(frame=i, **pixel_scores(a, b))
                x, y = (self.lpips_package.im2tensor(v).cuda() for v in (a, b))
                row['lpips_vgg'] = float(self.lpips(x, y).item())
                x, y = (self.preprocess(Image.fromarray(v)).unsqueeze(0).cuda() for v in (a, b))
                sim = self.torch.nn.functional.cosine_similarity(self.clip.encode_image(x), self.clip.encode_image(y), dim=1).item()
                row['clip'] = (sim+1)/2
                x, y = (to_tensor(Image.fromarray(v)).unsqueeze(0).cuda() for v in (a, b))
                row['dists'] = float(self.dists(x, y).item()); rows.append(row)
                if progress and (i == 0 or (i+1) % 32 == 0 or i+1 == len(source)):
                    progress(i+1, len(source))
        return summarize(rows, source.shape), rows


def evaluate(root, case, arm, signature, metrics):
    import cv2
    import numpy as np
    from PIL import Image
    from .research_quality import read_video
    from .stutter_metrics import evaluate as temporal
    from .official_quality import summarize
    cv2.setNumThreads(2)
    dest = folder(root, case, arm)
    if common.verify_receipt(dest, 'evaluation', signature):
        return read(dest/'quality.json')
    if not common.verify_receipt(dest, 'generation', signature):
        raise ValueError('generation missing')
    source = read_video(case['source_movie']); movie = dest/'reconstruction.mp4'; output = read_video(movie)
    quality, rows = metrics.evaluate(source, output, common.tick('전체 프레임 공식 화질'))
    movement = temporal(source, output, common.tick('전체 인접 프레임 동작'))
    keys = set(case['keys']); nonkeys = [r for r in rows if r['frame'] not in keys]
    key_exact = all(np.array_equal(np.asarray(Image.open(dest/f'frames/{k:05d}.png').convert('RGB')),
                   np.asarray(Image.open(path).convert('RGB'))) for k, path in zip(case['keys'], case['receiver']['key_paths']))
    if not key_exact:
        raise ValueError('received key RGB changed')
    result = dict(id=case['id'], snr=case['snr'], arm=arm, frames=case['frames'],
        quality=quality, temporal=movement,
        nonkeys=summarize(nonkeys, (len(nonkeys), 320, 576, 3)) if nonkeys else None,
        boundaries=common.boundary_diagnostics(source, output, case['keys']),
        received_key_png_exact=True, channel_uses=case['channel_uses'],
        video_sha256=sha256(movie), source_movie_sha256=sha256(case['source_movie']))
    qpath = dest/'quality.json'; write_json(qpath, result)
    csvpath = dest/'quality_delivered_mp4.csv'
    with csvpath.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    common.receipt(dest, 'evaluation', signature, [qpath, csvpath])
    return result


def mean_scores(rows):
    return dict(**{k:st.mean(r['quality'][k] for r in rows) for k in METRICS},
        **{k:st.mean(r['temporal'][k] for r in rows) for k in ('flow_error_mean','cadence_error_mean','false_freeze_rate')},
        nonkey_psnr=st.mean(r['nonkeys']['psnr_db'] for r in rows if r['nonkeys']),
        boundary_error=st.mean(st.mean(b['maximum_change_error'] for b in r['boundaries']) for r in rows))


def screening(rows, p):
    means = {arm:mean_scores([r for r in rows if r['arm'] == arm]) for arm in ARMS}
    base = means['combined_context']; c = p['criteria']; decisions = {}
    for arm in ARMS[1:]:
        x = means[arm]
        gates = dict(psnr=x['psnr_db'] >= base['psnr_db']-c['max_psnr_loss_db'],
            nonkeys=x['nonkey_psnr'] >= base['nonkey_psnr']-c['max_nonkey_psnr_loss_db'],
            lpips=x['lpips_vgg'] <= base['lpips_vgg']*(1+c['max_lpips_relative_increase']),
            dists=x['dists'] <= base['dists']*(1+c['max_dists_relative_increase']),
            flow=x['flow_error_mean'] <= base['flow_error_mean']*(1+c['max_flow_relative_increase']),
            cadence=x['cadence_error_mean'] <= base['cadence_error_mean']*(1+c['max_cadence_relative_increase']),
            freeze=x['false_freeze_rate'] <= base['false_freeze_rate']+c['max_freeze_increase'],
            boundary=x['boundary_error'] <= base['boundary_error']*(1+c['max_boundary_relative_increase']))
        gain = (x['psnr_db'] >= base['psnr_db']+c['min_psnr_gain_db']
                or x['lpips_vgg'] <= base['lpips_vgg']*(1-c['min_lpips_relative_gain'])
                or x['flow_error_mean'] <= base['flow_error_mean']*(1-c['min_flow_relative_gain']))
        decisions[arm] = dict(guardrails=gates, improvement=bool(gain), passed=all(gates.values()) and bool(gain))
    passing = [a for a in ARMS[1:] if decisions[a]['passed']]
    winner = min(passing, key=lambda a:means[a]['lpips_vgg']) if passing else None
    return dict(means=means, decisions=decisions, promoted=winner)


def phase(root, p, cases, arms, workers, metrics):
    tasks = [(str(root), c, arm, p['signature']) for c in cases for arm in arms]
    generation_tasks = [(r, {k:c[k] for k in ('id','snr','receiver')}, a, s) for r,c,a,s in tasks]
    event(overall=f'수신부 생성 0/{len(tasks)}', stage='완료 산출물 재사용·후보 생성')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for n, result in enumerate(pool.map(generate, generation_tasks), 1):
            event(overall=f'수신부 생성 {n}/{len(tasks)} ({100*n/len(tasks):.1f}%)',
                  current=f"{result['id']} SNR={result['snr']} {result['arm']}", stage='수신부 생성·영수증 검증 완료')
    rows = []
    for n, (_, case, arm, _) in enumerate(tasks, 1):
        event(overall=f'정량 평가 {n-1}/{len(tasks)} ({100*(n-1)/len(tasks):.1f}%)',
              current=f"{case['id']} SNR={case['snr']} {arm}", stage='전체 프레임 화질·동작·경계 평가')
        rows.append(evaluate(root, case, arm, p['signature'], metrics))
    return rows


def run(root, imported, workers):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    from .research_quality import read_video
    from .official_quality import OfficialMetrics
    import numpy as np
    root.mkdir(parents=True, exist_ok=True)
    with common.lock(root):
        p = prepare(root, imported)
        # Check evaluator equivalence before scoring any candidate.
        metrics = PersistentMetrics()
        parity = root/'metric_parity.json'
        if not parity.exists():
            case = next(c for c in p['cases'] if c['snr'] == 10)
            source = read_video(case['source_movie'])[[0, case['frames']//2, -1]]
            from PIL import Image
            generated = np.stack([np.asarray(Image.open(case['receiver']['current_frames'][i]).convert('RGB'))
                                  for i in (0, case['frames']//2, case['frames']-1)])
            _, a = metrics.evaluate(source, generated); _, b = OfficialMetrics().evaluate(source, generated)
            error = {k:max(abs(x[k]-y[k]) for x,y in zip(a,b)) for k in METRICS}
            if max(error.values()) > 1e-6:
                raise ValueError('persistent metric implementation differs from official metrics')
            write_json(parity, dict(status='PASSED',frames=3,max_error=error))
        primary = phase(root, p, [c for c in p['cases'] if c['snr'] == 10], ARMS, workers, metrics)
        decision = screening(primary, p); write_json(root/'screening_10db.json', decision)
        secondary = []
        if decision['promoted']:
            secondary = phase(root, p, [c for c in p['cases'] if c['snr'] != 10],
                              ('combined_context', decision['promoted']), workers, metrics)
        result = dict(status='COMPLETE_LOCAL_RECEIVER_ABLATION', signature=p['signature'],
            sources=36, primary_conditions=len(primary), secondary_conditions=len(secondary),
            screening_10db=decision, secondary_means={str(snr):{arm:mean_scores([r for r in secondary if r['snr']==snr and r['arm']==arm])
                for arm in ('combined_context', decision['promoted'])} for snr in (6,8)} if secondary else {},
            rows=[{**r, 'temporal':{k:v for k,v in r['temporal'].items() if k!='per_transition'}} for r in primary+secondary],
            independent_semantic_review='PENDING', production_defaults_changed=False,
            model_training=False, generator_reused=True, additional_transmission=0)
        write_json(root/'RESULT.json', result)
        event(overall=f'실험 {len(primary)+len(secondary)}/{len(primary)+len(secondary)} (100%)', stage='정량 비교 완료', state='완료')
        print('REPORT: '+str(root/'RESULT.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--import-root', type=Path, default=IMPORT)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be in [1,4]')
    run(args.root.resolve(), args.import_root.resolve(), args.workers)


if __name__ == '__main__':
    main()
