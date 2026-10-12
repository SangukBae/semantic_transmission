"""Temporal support ablation; the frozen spatial RFLOW correction is unchanged."""
import math


def experiment_arms(anchor_plan, strength=.7, fraction=.25):
    """A 2x2 support/strength design, including a matched coefficient-sum pair.

    Equal sums of coefficients are not equal realized correction energy: the
    residual depends on time position and the evolving generated state.
    """
    count = len(anchor_plan['latent_representative_frames'])
    sparse = list(anchor_plan['slots'])
    protected = list(anchor_plan['protected'])
    if not 0 < strength <= 1 or not 0 < fraction <= 1:
        raise ValueError('invalid strength/band')
    if (len(set(sparse)) != len(sparse) or len(set(protected)) != len(protected)
            or any(i < 0 or i >= count for i in sparse + protected)
            or set(sparse) & set(protected) or not sparse):
        raise ValueError('invalid anchor plan')
    dense = [i for i in range(count) if i not in protected]
    reduced = strength * len(sparse) / len(dense)
    rows = []
    for name, slots, coefficient in [('sparse_high_replay', sparse, strength),
                                      ('continuous_high', dense, strength),
                                      ('sparse_low', sparse, reduced),
                                      ('continuous_low', dense, reduced)]:
        rows.append(dict(name=name, support='sparse' if slots == sparse else 'continuous',
                         slots=slots, strength=coefficient, fraction=fraction,
                         positions=len(slots), coefficient_sum=coefficient * len(slots)))
    if not math.isclose(rows[0]['coefficient_sum'], rows[3]['coefficient_sum'], rel_tol=1e-12):
        raise ValueError('coefficient budget mismatch')
    return rows


def brightness_diagnostic(source, generated, temporal, target=35, window=(32, 38)):
    """Source-adjusted grayscale mean changes, plus the existing MAD diagnostic.

    This measures exposure-like flashes, not all local flicker or semantic error.
    Per-frame luma is measured at the same 288x160 size as the temporal metrics.
    """
    import cv2
    import numpy as np
    if source.shape != generated.shape or source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError('expected matching T,H,W,3 frames')
    if not 0 < window[0] <= target < window[1] < len(source):
        raise ValueError('invalid diagnostic frame/window')
    def luma(frames):
        return np.array([cv2.cvtColor(cv2.resize(frame, (288, 160)), cv2.COLOR_RGB2GRAY).mean()
                         for frame in frames])
    a, b = luma(source), luma(generated)
    error = b - a
    step = np.abs(np.diff(error))
    impulse = np.abs(error[1:-1] - .5 * (error[:-2] + error[2:]))
    mad = temporal['per_transition']['generated_mad']
    original_mad = temporal['per_transition']['source_mad']
    largest = int(np.argmax(mad)) + 1
    return dict(target_frame=target, window_frames=list(window),
                target_luma_impulse=float(impulse[target - 1]),
                target_incoming_mad=mad[target - 1], target_outgoing_mad=mad[target],
                window_max_mad=float(max(mad[window[0]:window[1]])),
                max_mad=float(mad[largest - 1]), max_mad_transition=largest,
                source_mad_at_max=original_mad[largest - 1],
                transitions_above_mad10=sum(v > 10 for v in mad),
                max_source_adjusted_luma_step=float(step.max()),
                max_luma_impulse=float(impulse.max()), max_luma_impulse_frame=int(np.argmax(impulse)) + 1,
                per_frame=dict(source_luma=a.tolist(), generated_luma=b.tolist(),
                               luma_error=error.tolist(), luma_impulse=[None, *impulse.tolist(), None]),
                per_transition=dict(source_adjusted_luma_step=step.tolist()))
