"""Independent post-reconstruction checks; never imported by the receiver.

Compares base/generated/refined frames against the frozen sender inputs. A
historical baseline is accepted only with an explicit same-source hash receipt.
"""
import argparse
from pathlib import Path

import numpy as np

from . import unified_bsu as u
from . import unified_bsu_core as core
from .artifacts import sha256, write_json


def metrics(source, output, fps=24):
    from skimage.metrics import structural_similarity
    if source.shape != output.shape:
        raise ValueError('evaluation frame count/geometry differs')
    # Avoid materializing two full 60/120-second float64 videos and their diffs.
    errors, ssims, temporal = [], [], []
    previous = None
    for x, y in zip(source, output):
        residual = x.astype(np.float32)-y.astype(np.float32)
        errors.append(float(np.square(residual).mean(dtype=np.float64)))
        ssims.append(float(structural_similarity(x, y, channel_axis=2, data_range=255)))
        if previous is not None:
            temporal.append(float(np.abs(residual-previous).mean(dtype=np.float64)))
        previous = residual
    block = max(1, round(fps))
    timeline = [dict(start_frame=i+1, frames=len(temporal[i:i+block]),
                     error=float(np.mean(temporal[i:i+block]))) for i in range(0, len(temporal), block)]
    return dict(psnr_db=float(10*np.log10(255**2/max(float(np.mean(errors)), 1e-12))),
        ssim=float(np.mean(ssims)), temporal_rgb_mae=float(np.mean(temporal)) if temporal else 0.,
        worst_second_temporal_rgb_mae=max((r['error'] for r in timeline), default=0.),
        temporal_timeline=timeline, frames=len(source))


def evaluate(root, full_metrics=False, reference=None):
    from .research_quality import read_video
    p = core.read(root / 'protocol.json'); result = core.read(root / 'RESULT.json')
    if result['request_signature'] != p['signature']:
        raise ValueError('result/protocol mismatch')
    if sha256(result['movie']) != result['movie_sha256']:
        raise ValueError('final movie changed')
    for name, digest in p['source_hashes'].items():
        if sha256(Path(p['source']) / name) != digest:
            raise ValueError('evaluation source changed')
    source = u.load_rgb(p['source'], p['context']['video']['frames'])
    from .etri_60s_check import pts_audit
    rows = {}; artifacts = {}
    evaluator = None
    if full_metrics:
        from .official_quality import OfficialMetrics
        evaluator = OfficialMetrics()
    for arm in ('base', 'generated', 'refined'):
        if not (root / arm / 'frames').exists():
            continue
        movie = root / arm / 'reconstruction.mp4'
        pts_audit(movie, len(source), p['context']['video']['fps'])
        output = read_video(movie)
        row = metrics(source, output, p['context']['video']['fps'])
        if evaluator:
            row['official_quality'], per_frame = evaluator.evaluate(source, output)
            block = max(1, round(p['context']['video']['fps']))
            row['quality_timeline'] = [dict(start_frame=i, frames=len(per_frame[i:i+block]),
                **{k: float(np.mean([r[k] for r in per_frame[i:i+block]]))
                   for k in ('psnr_db', 'ssim', 'lpips_vgg', 'clip', 'dists')})
                for i in range(0, len(per_frame), block)]
        rows[arm] = row
        artifacts[arm] = dict(movie_sha256=sha256(movie),
            frame_hashes={f.name: sha256(f) for f in sorted((root / arm / 'frames').glob('*.png'))})
    comparison = None
    if reference:
        ref = core.read(reference)
        if (ref['source_hashes'] != p['source_hashes'] or ref['fps'] != p['context']['video']['fps']
                or sha256(ref['movie']) != ref['movie_sha256']):
            raise ValueError('reference input/timeline/movie provenance does not match')
        if type(ref['channel_uses']) is not int or ref['channel_uses'] < 1:
            raise ValueError('invalid measured reference transmission cost')
        pts_audit(Path(ref['movie']), len(source), ref['fps'])
        old = read_video(Path(ref['movie'])); row = metrics(source, old, ref['fps'])
        if evaluator:
            row['official_quality'] = evaluator.evaluate(source, old)[0]
        final = rows.get('refined', rows['base'])
        comparison = dict(reference=row, reference_channel_uses=ref['channel_uses'],
            channel_saving_percent=100*(1-result['channel_uses']/ref['channel_uses']),
            psnr_delta_db=final['psnr_db']-row['psnr_db'], ssim_delta=final['ssim']-row['ssim'],
            temporal_rgb_mae_delta=final['temporal_rgb_mae']-row['temporal_rgb_mae'])
    report = dict(status='COMPLETE_REGISTERED_PIXEL_CHECKS', rows=rows, comparison=comparison, artifacts=artifacts,
        channel_uses=result['channel_uses'], budget=result['budget'], budget_met=result['budget_met'],
        full_perceptual_metrics=full_metrics, independent_semantic_review=False, metric_input='decoded_final_mp4',
        source_is_evaluation_only=True, generation_strength_selected_with_source=False)
    write_json(root / 'quality.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--full-metrics', action='store_true')
    parser.add_argument('--reference', type=Path, help='same-source receipt: source_hashes,fps,movie,movie_sha256,channel_uses')
    args = parser.parse_args()
    from .etri_60s_check import lock
    with lock(core.REPO / '.local/etri_60s_check.lock'):
        value = evaluate(args.run.resolve(), args.full_metrics, args.reference)
    print(__import__('json').dumps(value, indent=2))


if __name__ == '__main__':
    main()
