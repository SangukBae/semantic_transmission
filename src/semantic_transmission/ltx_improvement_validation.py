"""Separate evaluator for training-free LTX changes; originals never enter RX.

The replay manifest records received paths and evaluation sources separately.
Postprocessing evaluates full existing videos. The expensive neural pilot uses
predeclared received-key prefixes and reports that limited scope explicitly.
"""
import argparse
import copy
import csv
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

from . import ltx_conditioning as cond, ltx_receiver as ltx, bsu_ltx_improvements as rx
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages

MODULE = 'semantic_transmission.ltx_improvement_validation'


def cases(manifest, selected=None):
    manifest = Path(manifest).resolve()
    data = ltx.read(manifest)
    result = []
    for row in data['cases']:
        if selected and row['id'] not in selected:
            continue
        row = dict(row)
        for key in ('packet', 'keys', 'trajectory', 'raw', 'source'):
            if row.get(key):
                row[key] = str((manifest.parent/row[key]).resolve())
        row['legacy'] = str(manifest.parent/row['id']/'legacy')
        result.append(row)
    if not result or selected and set(selected) != {r['id'] for r in result}:
        raise ValueError('no cases or unknown requested case')
    return result


def receiver_args(row, root, variant, action, runtime, raw=None, seed=2025):
    return SimpleNamespace(action=action, packet=Path(row['packet']), keys=Path(row['keys']),
        trajectory=Path(row['trajectory']) if row.get('trajectory') else None,
        variant=variant, seed=seed, raw=Path(raw) if raw else None, output=Path(root),
        offload='disk',
        ltx_python=runtime, model_dir=ltx.REPO/'.local/checkpoints/ltx25', vendor=ltx.REPO/'.local/vendor/LTX-2')


def evaluate(source, movie, destination, keys, frame_count=None, pixel_only=False):
    import numpy as np
    from .research_quality import read_video
    from .official_quality import OfficialMetrics, pixel_scores
    from .stutter_metrics import evaluate as temporal
    from .video_io import probe
    from .etri_60s_check import pts_audit
    info = probe(movie)
    source_info = probe(source)
    if info['fps'] != source_info['fps']:
        raise ValueError('evaluation FPS mismatch')
    a, b = read_video(source), read_video(movie)
    if frame_count is not None:
        a = a[:frame_count]
    if a.shape != b.shape:
        raise ValueError('evaluation source and reconstruction coverage mismatch')
    if pixel_only:
        rows = [dict(frame=i, **pixel_scores(x, y)) for i, (x, y) in enumerate(zip(a, b))]
        quality = {k: statistics.mean(r[k] for r in rows) for k in ('psnr_db', 'ssim')}
    else:
        quality, rows = OfficialMetrics().evaluate(a, b,
            lambda n, total: print(f'QUALITY frames {n}/{total} ({100*n/total:.1f}%) ETA=unknown', flush=True)
            if n % 240 == 0 or n == total else None)
    motion = temporal(a, b) if len(a) >= 3 else None
    if motion:
        motion.pop('per_transition', None)
    nonkeys = [r for r in rows if r['frame'] not in keys]
    boundary = [r for r in rows if any(abs(r['frame']-k) <= 2 for k in keys)]
    scores = [k for k in quality if k in ('psnr_db', 'ssim', 'lpips_vgg', 'dists', 'clip')]
    value = dict(quality=quality, temporal=motion, frames=len(a),
        nonkey_mean={k: statistics.mean(r[k] for r in nonkeys) for k in scores} if nonkeys else {},
        boundary_mean={k: statistics.mean(r[k] for r in boundary) for k in scores},
        source_sha256=sha256(source), video_sha256=sha256(movie),
        timeline=pts_audit(movie, len(a), source_info['fps']),
        source_frame_indices=list(range(len(a))), pixel_only=pixel_only,
        independent_semantic_review='PENDING')
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'quality.json', value)
    with (destination/'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return value


def aggregate(records):
    arms = sorted({r['arm'] for r in records})
    result = {}
    for arm in arms:
        selected = [r['metrics'] for r in records if r['arm'] == arm]
        result[arm] = dict(videos=len(selected), quality={}, temporal={})
        for group in ('quality', 'temporal'):
            keys = set.intersection(*(set(r[group] or {}) for r in selected))
            for key in sorted(keys):
                if all(isinstance(r[group][key], (int, float)) and not isinstance(r[group][key], bool) for r in selected):
                    result[arm][group][key] = statistics.mean(r[group][key] for r in selected)
    return result


def prepare_all(args):
    records = []
    for row in cases(args.manifest, args.case):
        variants = {}
        for variant in cond.VARIANTS:
            output = args.output/row['id']/variant
            rx.run(receiver_args(row, output, variant, 'prepare', args.ltx_python))
            value = ltx.read(output/'conditions/request.json')
            variants[variant] = dict(windows=len(value['windows']),
                received_keys=len(value['receiver_improvements']['original_key_indices']),
                guides=sum('strength' in i for w in value['windows'] for i in w['images']),
                suspected_cuts=sum(c['suspected_cut'] for c in value['receiver_improvements']['cuts']))
        records.append(dict(id=row['id'], variants=variants))
    write_json(args.output/'RESULT.json', dict(status='ALL_REAL_RECEIVED_REQUESTS_VALIDATED',
        records=records, training=False, neural_inference_executed=False, generalization_proven=False))


def postprocess(args):
    records = []
    cohort = cases(args.manifest, args.case)
    contract = dict(manifest_sha256=sha256(args.manifest), implementation=rx.code_identity(),
        evaluation_code_sha256=sha256(__file__), pixel_only=args.pixel_only,
        cases=[dict(id=r['id'], source_sha256=sha256(r['source'])) for r in cohort])
    args.output.mkdir(parents=True, exist_ok=True); freeze(args.output/'protocol.json', contract)
    for i, row in enumerate(cohort):
        output = args.output/row['id']/'postprocess'
        rx.run(receiver_args(row, output, 'postprocess', 'run', args.ltx_python, raw=row['raw']))
        value = ltx.read(output/'conditions/request.json')
        stages = Stages(output, fingerprint(contract))
        stages.step('evaluate', ['evaluation'], ['evaluation'], lambda _: evaluate(row['source'],
            output/'refined/reconstruction.mp4', output/'evaluation',
            value['receiver_improvements']['original_key_indices'], pixel_only=args.pixel_only))
        records.append(dict(id=row['id'], arm='adaptive', metrics=ltx.read(output/'evaluation/quality.json')))
        for arm, folder in (('raw', Path(row['raw'])), ('legacy', Path(row['legacy']))):
            old = ltx.read(folder/'quality.json')
            if old['source_sha256'] != sha256(row['source']) or old['video_sha256'] != sha256(folder/'reconstruction.mp4'):
                raise ValueError('cached baseline source or video hash differs')
            records.append(dict(id=row['id'], arm=arm, metrics=dict(quality=old['delivered_mp4'], temporal=old['temporal']),
                                verified_existing_metrics=True))
        print(f'POSTPROCESS videos {i+1}/{len(cohort)} ({100*(i+1)/len(cohort):.1f}%) ETA=unknown', flush=True)
        write_json(args.output/'status.json', dict(status='RUNNING', completed=i+1, total=len(cohort)))
    write_json(args.output/'RESULT.json', dict(status='COMPLETE_FULL_VIDEO_POSTPROCESS_COMPARISON',
        records=records, summary=aggregate(records), training=False, default_adopted=False,
        source_cohort='known 14-video NTSCC development replay', generalization_proven=False,
        independent_semantic_review='PENDING'))
    write_json(args.output/'status.json', dict(status='COMPLETE_FULL_VIDEO_POSTPROCESS_COMPARISON', completed=len(cohort)))


def prefix(value, end):
    """A diagnostic prefix of an unchanged paid packet, not a new transmission."""
    cond.validate(value)
    value = copy.deepcopy(value)
    windows = [w for w in value['windows'] if w['end'] <= end]
    if not windows or windows[-1]['end'] != end:
        raise ValueError('pilot end must match a complete generation window')
    info = value['receiver_improvements']
    info['diagnostic_prefix'] = dict(full_video_frames=value['video']['frames'], first=0, last=end,
        channel_scope='complete paid packet replay; no standalone prefix bitrate claim')
    info['original_key_indices'] = [k for k in info['original_key_indices'] if k <= end]
    value['windows'], value['video']['frames'] = windows, end+1
    value['signature'] = fingerprint({k: v for k, v in value.items() if k != 'signature'})
    cond.validate(value)
    return value


def pilot(args):
    from .bsu_ltx_comparison import command
    import os
    if not args.case:
        raise ValueError('pilot requires predeclared --case IDs')
    cohort = cases(args.manifest, args.case)
    entries = []
    for row in cohort:
        full = ltx.request(row['packet'], row['keys'])
        if len(full['windows']) < 2:
            raise ValueError('context pilot requires at least three received keys')
        end = full['windows'][1]['end']
        if end+1 > args.max_frames:
            raise ValueError('predeclared two-interval prefix exceeds pilot frame limit')
        for seed in args.seeds:
            for variant in cond.VARIANTS:
                dest = args.output/row['id']/f'seed_{seed}'/variant
                value = cond.prepare(row['packet'], row['keys'], dest/'full_conditions', variant, row.get('trajectory'), seed)
                freeze(dest/'request.json', prefix(value, end))
                entries.append(dict(id=row['id'], seed=seed, variant=variant, output=str(dest.resolve()),
                    row=row, frames=end+1, request_sha256=sha256(dest/'request.json')))
    contract = dict(entries=entries, implementation=rx.code_identity(), evaluation_code_sha256=sha256(__file__),
                    manifest_sha256=sha256(args.manifest), pixel_only=args.pixel_only,
                    execution_offload=args.offload,
                    model_dir=str(args.model_dir.resolve()), vendor=str(args.vendor.resolve()))
    freeze(args.output/'pilot.json', contract)
    env = dict(os.environ, PYTHONPATH=str(ltx.REPO/'src'), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    stages = Stages(args.output, fingerprint(contract))
    stages.step('generate_pilot', [], ['generation_complete.json'], lambda log: command([
        args.ltx_python, '-m', MODULE, 'generate-pilot', '--output', args.output], log, env))
    return evaluate_pilot(args)


def generate_pilot(args):
    contract = ltx.read(args.output/'pilot.json')
    if contract['implementation'] != rx.code_identity() or contract['evaluation_code_sha256'] != sha256(__file__):
        raise ValueError('pilot implementation changed')
    assets = dict(ltx.runtime(contract['model_dir'], contract['vendor']), execution_offload=contract['execution_offload'])
    freeze(args.output/'runtime.json', assets)
    generator = None
    def factory(current):
        nonlocal generator
        if generator is None:
            generator = ltx.Generator(current)
        return generator
    entries = [e for e in contract['entries'] if e['variant'] != 'postprocess']
    for i, entry in enumerate(entries):
        dest = Path(entry['output'])
        if sha256(dest/'request.json') != entry['request_sha256']:
            raise ValueError('pilot request changed')
        value = ltx.read(dest/'request.json'); cond.validate(value)
        ltx.reconstruct(value, assets, dest/'raw', factory=factory)
        print(f'PILOT generation jobs {i+1}/{len(entries)} ({100*(i+1)/len(entries):.1f}%) ETA=unknown', flush=True)
    write_json(args.output/'generation_complete.json', dict(status='COMPLETE_REAL_LTX_PILOT_GENERATION', jobs=len(entries)))


def evaluate_pilot(args):
    contract = ltx.read(args.output/'pilot.json')
    records = []
    for entry in contract['entries']:
        dest = Path(entry['output']); row = entry['row']
        if sha256(dest/'request.json') != entry['request_sha256']:
            raise ValueError('pilot request changed')
        value = ltx.read(dest/'request.json')
        raw = dest.parent/'baseline/raw' if entry['variant'] == 'postprocess' else dest/'raw'
        steps = Stages(dest, fingerprint(contract))
        steps.step('refine', ['refined'], ['refined'], lambda _: rx.refinement(value, raw, dest/'refined',
            row['packet'], row['keys'], row.get('trajectory')))
        steps.step('evaluate', ['evaluation'], ['evaluation'], lambda _: evaluate(row['source'],
            dest/'refined/reconstruction.mp4', dest/'evaluation', value['receiver_improvements']['original_key_indices'],
            frame_count=entry['frames'], pixel_only=contract['pixel_only']))
        records.append(dict(id=entry['id'], seed=entry['seed'], arm=entry['variant'],
            codec=row.get('codec', 'recorded in received packet'),
            metrics=ltx.read(dest/'evaluation/quality.json'), frames=entry['frames'],
            guides=sum('strength' in image for w in value['windows'] for image in w['images']),
            windows=len(value['windows'])))
    write_json(args.output/'RESULT.json', dict(status='COMPLETE_REAL_LTX_FOUR_FACTOR_PILOT', records=records,
        summary=aggregate(records), training=False, default_adopted=False, generalization_proven=False,
        scope='predeclared prefixes; codec recorded per case; same full received packet and seeds within each comparison',
        execution_offload=contract['execution_offload'],
        independent_semantic_review='PENDING'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'postprocess', 'pilot', 'generate-pilot', 'evaluate-pilot'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025, 42])
    parser.add_argument('--max-frames', type=int, default=97)
    parser.add_argument('--offload', choices=('cpu', 'disk'), default='disk')
    parser.add_argument('--pixel-only', action='store_true')
    parser.add_argument('--ltx-python', type=Path, default=ltx.REPO/'.local/envs/bsu-ltx25/bin/python')
    parser.add_argument('--model-dir', type=Path, default=ltx.REPO/'.local/checkpoints/ltx25')
    parser.add_argument('--vendor', type=Path, default=ltx.REPO/'.local/vendor/LTX-2')
    args = parser.parse_args()
    if args.action not in ('generate-pilot', 'evaluate-pilot') and args.manifest is None:
        parser.error('--manifest is required')
    return {'prepare': prepare_all, 'postprocess': postprocess, 'pilot': pilot,
            'generate-pilot': generate_pilot, 'evaluate-pilot': evaluate_pilot}[args.action](args)


if __name__ == '__main__':
    main()
