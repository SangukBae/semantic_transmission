"""Source-only SPARC anchor planning; semantic decisions are supplied by Qwen.

The low-resolution linear-interpolation error is only a source motion proxy,
not a measurement of RIFE reconstruction quality or a receiver-quality oracle.
"""
import math
from pathlib import Path

from .artifacts import sha256
from .qwen35_semantic_selection import validate_keys


def signals(source, config, progress=None):
    import numpy as np
    from PIL import Image
    frames = source['video']['frames']; proxies, cuts = [], []
    previous = None
    for i in range(frames):
        path = Path(source['directory']) / f'{i}.png'
        if sha256(path) != source['hashes'][path.name]:
            raise ValueError('source frame changed during selection')
        with Image.open(path) as image:
            image = image.convert('RGB')
            current = np.asarray(image, dtype=np.float32)
            proxies.append(np.asarray(image.resize(tuple(config['proxy_size']), Image.Resampling.BOX), dtype=np.float32))
        if previous is not None and float(np.abs(current - previous).mean()) > config['scene_cut_mad_threshold']:
            cuts.append(i)
        previous = current
        if progress: progress(i + 1, frames)
    return np.stack(proxies), cuts


def validate_cuts(cuts, frames):
    if (not isinstance(cuts, list) or any(type(i) is not int or not 0 < i < frames for i in cuts)
            or cuts != sorted(set(cuts))):
        raise ValueError('invalid scene-cut timeline')


def validate_gap(max_gap):
    if max_gap is not None and (type(max_gap) is not int or not 2 <= max_gap <= 4):
        raise ValueError('RIFE maximum gap must be 2..4 or None (no interval limit)')


def validate_anchors(indices, frames, max_gap, cuts=()):
    validate_keys(indices, frames)
    validate_cuts(list(cuts), frames)
    validate_gap(max_gap)
    if max_gap is not None and max(b - a for a, b in zip(indices, indices[1:])) > max_gap:
        raise ValueError('anchor gap exceeds the registered RIFE policy')
    if any(c not in indices or c - 1 not in indices for c in cuts):
        raise ValueError('both sides of each cut must be transmitted')


def adaptive(proxies, semantic_keys, cuts, *, max_gap=4, threshold=6.0):
    import numpy as np
    validate_gap(max_gap)
    if (proxies.ndim != 4 or proxies.shape[-1] != 3 or not np.isfinite(proxies).all()
            or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('invalid adaptive selector inputs')
    frames = len(proxies); validate_keys(semantic_keys, frames); validate_cuts(cuts, frames)
    selected = set(semantic_keys) | {i for cut in cuts for i in (cut - 1, cut)}
    boundaries = sorted(selected)
    pending = list(zip(boundaries, boundaries[1:])); additions = []
    while pending:
        a, b = pending.pop()
        if b - a <= 1:
            continue
        errors = []
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            predicted = proxies[a] * (1 - t) + proxies[b] * t
            errors.append(float(np.abs(proxies[i] - predicted).mean()))
        largest = max(errors)
        if largest > threshold:
            # Deterministic tie break favors the midpoint, then earlier time.
            i = max(range(a + 1, b), key=lambda k: (errors[k-a-1], -abs(k-(a+b)/2), -k))
            reason = 'source_motion_proxy'
        elif max_gap is not None and b - a > max_gap:
            i = (a + b) // 2; reason = 'maximum_gap'
        else:
            continue
        selected.add(i)
        additions.append(dict(index=i, interval=[a, b], reason=reason, proxy_mae=errors[i-a-1]))
        pending.extend(((a, i), (i, b)))
    indices = sorted(selected); validate_anchors(indices, frames, max_gap, cuts)
    return dict(indices=indices, semantic_keys=semantic_keys, cuts=cuts,
        auxiliary_keys=sorted(selected - set(semantic_keys)), additions=additions,
        max_gap=max_gap, proxy_mae_threshold=threshold,
        selection_reads_source_only=True, receiver_quality_used=False)


def uniform_matched(frames, count, cuts, max_gap=4):
    """Exact-count temporal control, retaining cut boundaries but no semantic keys."""
    validate_cuts(cuts, frames)
    validate_gap(max_gap)
    mandatory = sorted({0, frames - 1, *(i for cut in cuts for i in (cut - 1, cut))})
    gaps = [b-a for a, b in zip(mandatory, mandatory[1:])]
    parts = [1 if max_gap is None else math.ceil(g / max_gap) for g in gaps]
    if type(count) is not int or not sum(parts) + 1 <= count <= frames:
        raise ValueError('matched count cannot satisfy the gap/cut constraints')
    while sum(parts) + 1 < count:
        eligible = [i for i, (g, p) in enumerate(zip(gaps, parts)) if p < g]
        i = max(eligible, key=lambda k: (gaps[k] / parts[k], -k))
        parts[i] += 1
    indices = [mandatory[0]]
    for a, b, count_parts in zip(mandatory, mandatory[1:], parts):
        indices.extend(a + round(i * (b-a) / count_parts) for i in range(1, count_parts + 1))
    validate_anchors(indices, frames, max_gap, cuts)
    if len(indices) != count:
        raise ValueError('uniform matched count changed')
    return indices


def plan(proxies, semantic_keys, cuts, config, mode='adaptive'):
    result = adaptive(proxies, semantic_keys, cuts, max_gap=config['max_gap'],
                      threshold=config['proxy_mae_threshold'])
    result['mode'] = mode
    result['adaptive_anchor_count'] = len(result['indices'])
    if mode == 'uniform-matched':
        result['indices'] = uniform_matched(len(proxies), len(result['indices']), cuts, config['max_gap'])
    elif mode == 'fixed-stride2':
        result['indices'] = sorted({*range(0, len(proxies), 2), len(proxies)-1,
                                    *(i for cut in cuts for i in (cut-1, cut))})
    elif mode != 'adaptive':
        raise ValueError('unknown anchor policy')
    validate_anchors(result['indices'], len(proxies), config['max_gap'], cuts)
    result['semantic_keys_preserved'] = set(semantic_keys).issubset(result['indices'])
    if mode == 'adaptive' and not result['semantic_keys_preserved']:
        raise ValueError('adaptive selection dropped a semantic key')
    result['transmitted_auxiliary_keys'] = sorted(set(result['indices']) - set(semantic_keys))
    return result


def scene_groups(indices, cuts):
    """A scene start becomes I; other selected anchors remain predictive P frames."""
    cutset = set(cuts); groups = []
    for index in indices:
        if not groups or index in cutset:
            groups.append([])
        groups[-1].append(index)
    return groups
