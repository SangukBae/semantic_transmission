"""Released LGVSC frame pairing/aggregation and explicitly scoped rate reports."""
import math

from .official_quality import KEYS

EVALUATION_PROFILE = 'lgvsc_official_same_index_mp4_v2'
AUTHOR_KEYS = ('clip', 'psnr_first100', 'psnr_last100', 'ssim_first100',
               'ssim_last100', 'psnr', 'ssim', 'lpips', 'dists')
METRIC_NAMES = dict(zip(('psnr', 'ssim', 'lpips', 'clip', 'dists'), KEYS))


def author_summary(rows, reference_frames):
    """Match final_score_calc_100.py, including windows based on the reference.

    The release intersects equal frame filenames, without aligning segment
    endpoints. Its last-100 window belongs to the FULL reference video, even
    when the reconstruction ends earlier. An empty window is reported as zero.
    """
    import numpy as np
    if not rows or reference_frames < len(rows):
        raise ValueError('Expected nonempty common-frame rows within the reference')
    if [r['frame'] for r in rows] != list(range(len(rows))):
        raise ValueError('Released evaluation requires contiguous equal frame numbers')
    if any(not math.isfinite(r[k]) for r in rows for k in KEYS):
        raise ValueError('Nonfinite quality metric')
    def mean(group, key):
        return float(np.mean([r[key] for r in group])) if group else 0.0
    first = rows[:100]
    last = [r for r in rows if r['frame'] >= max(0, reference_frames - 100)]
    result = {name: mean(rows, key) for name, key in METRIC_NAMES.items()}
    for label, group in (('first100', first), ('last100', last)):
        result[f'psnr_{label}'] = mean(group, 'psnr_db')
        result[f'ssim_{label}'] = mean(group, 'ssim')
    return {key: result[key] for key in AUTHOR_KEYS}


def evaluate_same_index(source, reconstructed, metrics, progress=None):
    if source.shape[1:] != reconstructed.shape[1:]:
        raise ValueError('Official baseline requires matching video geometry')
    count = min(len(source), len(reconstructed))
    if count < 1:
        raise ValueError('No common video frames')
    _, rows = metrics.evaluate(source[:count], reconstructed[:count], progress=progress)
    summary = author_summary(rows, len(source))
    return summary, rows


def average_videos(videos):
    """The release gives each video equal weight, not each individual frame."""
    import numpy as np
    if not videos or len({r['name'] for r in videos}) != len(videos):
        raise ValueError('Expected unique completed videos')
    if any(not math.isfinite(v[k]) for v in videos for k in AUTHOR_KEYS):
        raise ValueError('Nonfinite video metric')
    return dict(name='Average', **{k: float(np.mean([v[k] for v in videos])) for k in AUTHOR_KEYS})


def rate_breakdown(ntscc, metadata):
    """Keep simulated symbols distinct from the untransmitted rate-index estimate."""
    if ntscc['status'] != 'COMPLETE_OFFICIAL_NTSCC' or metadata['status'] != 'PASSED':
        raise ValueError('Incomplete rate evidence')
    visual = ntscc['visual_complex_channel_uses']
    digital = metadata['complex_channel_uses']
    estimate = ntscc['rate_index_capacity_estimate_channel_uses']
    denominator = ntscc['normalized_source_denominator']
    official_denominator = ntscc['official_denominator']
    if (any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
            for v in (visual, digital, denominator, official_denominator))
            or not math.isfinite(estimate) or estimate < 0
            or not math.isfinite(ntscc['official_reference_cbr'])):
        raise ValueError('Invalid channel accounting')
    if not math.isclose(ntscc['official_reference_cbr'] * official_denominator,
                        visual + estimate, rel_tol=1e-6, abs_tol=1e-3):
        raise ValueError('Author CBR and measured visual/rate-index components disagree')
    measured = visual + digital
    estimated_total = measured + estimate
    return dict(ntscc, caption_flow_ldpc_channel_uses=digital,
        measured_simulation_channel_uses=measured,
        measured_simulation_cbr=measured / denominator,
        estimated_semantic_total_channel_uses=estimated_total,
        estimated_semantic_total_cbr=estimated_total / denominator,
        estimated_semantic_total_cbr_official_denominator=estimated_total / official_denominator,
        actual_complete_transmission_cbr=None,
        accounting_scope={
            'official_reference_cbr': 'visual symbols + capacity-approximated rate-index; author denominator',
            'measured_simulation_cbr': 'visual + CSV LDPC symbols actually simulated; full source denominator; excludes rate-index',
            'estimated_semantic_total_cbr': 'measured simulation + capacity-approximated rate-index; full source denominator',
            'not_implemented': ['physical transmission of NTSCC rate-index', 'physical link/protocol overhead'],
            'paper_table_accounting_identity_verified': False})
