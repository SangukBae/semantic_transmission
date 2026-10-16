"""Sender-only motion ranking and preregistered reconstruction guardrails.

Endpoint DIS flow and forward splatting are a shortlist heuristic, not a
replacement for actual BSU reconstruction or evidence of semantic fidelity.
"""
import copy
import math

CRITERIA = dict(psnr_loss_db=.10, ssim_loss=.005, lpips_increase=.01,
                dists_increase=.01, clip_loss=.002, flow_increase=.02,
                cadence_increase=.02, freeze_increase=.005, boundary_increase=.05)
PARAMETERS = dict(removal_fraction=.20, maximum_merged_gap=96,
                  probe_count=7, size=[288, 160], eligible_kinds=['pose', 'position'])


def subset_packet(packet, visual, keys, segments, seed):
    """Retain exact per-key codec bytes; rebuild both offset tables and header."""
    from .wire import pack, unpack
    header, payload = unpack(packet)
    original = [k['index'] for k in header['keyframes']]
    if (keys != sorted(set(keys)) or len(keys) < 2 or keys[0] != original[0]
            or keys[-1] != original[-1] or not set(keys) <= set(original)
            or len(segments) != len(keys)-1):
        raise ValueError('invalid key subset or segment coverage')
    if len(visual) != 8*sum(k['complex_count'] for k in header['keyframes']):
        raise ValueError('visual byte count differs from header')
    replacement = copy.deepcopy(header); records = []; rates = bytearray(); symbols = bytearray()
    for item in header['keyframes']:
        if item['index'] not in keys:
            continue
        r = dict(item, complex_offset=len(symbols)//8, rate_offset=len(rates))
        a, n = item['complex_offset']*8, item['complex_count']*8
        symbols.extend(visual[a:a+n])
        a, n = item['rate_offset'], item['rate_bytes']
        rates.extend(payload[a:a+n]); records.append(r)
    replacement.update(keyframes=records, segments=segments)
    replacement['decoder']['seed'] = seed
    return pack(replacement, rates), bytes(symbols)


def guard(baseline, candidate, criteria=None):
    c = criteria or CRITERIA; checks = {}
    for section in ('quality', 'nonkeys'):
        a, b = baseline[section], candidate[section]
        for metric, limit, relative, higher in (
            ('psnr_db', c['psnr_loss_db'], False, True),
            ('ssim', c['ssim_loss'], False, True),
            ('clip', c['clip_loss'], False, True),
            ('lpips_vgg', c['lpips_increase'], True, False),
            ('dists', c['dists_increase'], True, False)):
            allowance = abs(a[metric])*limit if relative else limit
            change = b[metric]-a[metric]
            checks[f'{section}.{metric}'] = bool(math.isfinite(change) and
                ((change >= -allowance) if higher else (change <= allowance)))
    for metric, limit, relative in (
        ('flow_error_mean', c['flow_increase'], True),
        ('cadence_error_mean', c['cadence_increase'], True),
        ('false_freeze_rate', c['freeze_increase'], False)):
        a, b = baseline['temporal'][metric], candidate['temporal'][metric]
        checks[metric] = bool(math.isfinite(b) and b <= a+(abs(a)*limit if relative else limit))
    checks['fixed_boundary_error'] = candidate['boundary_error'] <= baseline['boundary_error']*(1+c['boundary_increase'])
    checks['fewer_total_symbols'] = candidate['channel_uses'] < baseline['channel_uses']
    return dict(passed=all(checks.values()), checks=checks,
                failures=[k for k, passed in checks.items() if not passed])


def splat(rgb, flow, fraction):
    import numpy as np
    h, w = rgb.shape[:2]; yy, xx = np.mgrid[:h, :w]
    x = xx + flow[..., 0]*fraction; y = yy + flow[..., 1]*fraction
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    out = np.zeros((h, w, 3), np.float64); weight = np.zeros((h, w), np.float64)
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        xi, yi = x0+dx, y0+dy
        z = (1-np.abs(x-xi))*(1-np.abs(y-yi))
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(weight, (yi[valid], xi[valid]), z[valid])
        np.add.at(out, (yi[valid], xi[valid]), rgb[valid]*z[valid, None])
    return out/np.maximum(weight[..., None], 1e-8), weight


def proposals(keys, decisions, frame_loader, parameters=None):
    import cv2
    import numpy as np
    p = parameters or PARAMETERS
    cv2.setNumThreads(2); cv2.setRNGSeed(2025)
    kinds = {d['candidate']: d['result']['kind'] for d in decisions if d['selected']}
    protected = {k for k in keys if kinds.get(k) not in p['eligible_kinds']} | {keys[0], keys[-1]}
    cache = {}; scores = {}
    def frame(i):
        if i not in cache:
            cache[i] = cv2.resize(frame_loader(i), tuple(p['size']), interpolation=cv2.INTER_AREA)
        return cache[i]
    def score(a, k, b):
        if (a, k, b) in scores:
            return scores[a, k, b]
        left, right = frame(a), frame(b)
        gray = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        ab = flow.calc(gray(left), gray(right), None); ba = flow.calc(gray(right), gray(left), None)
        probes = sorted(set(np.linspace(a, b, p['probe_count']+2).round().astype(int).tolist()[1:-1]+[k]))
        errors = []; missing = []
        for t in probes:
            f = (t-a)/(b-a); x, wx = splat(left, ab, f); y, wy = splat(right, ba, 1-f)
            valid = (wx+wy) > .1
            prediction = (x*wx[..., None]*(1-f)+y*wy[..., None]*f)/np.maximum(
                (wx*(1-f)+wy*f)[..., None], 1e-8)
            residual = np.abs(prediction-frame(t)).mean(axis=2)/255
            # Include holes explicitly; neither large moving objects nor new
            # visibility can disappear from the score through an invalid mask.
            residual[~valid] = 1.
            errors.append(float(.5*residual.mean()+.5*np.quantile(residual, .90)))
            missing.append(float((~valid).mean()))
        value = dict(key=k, left=a, right=b, kind=kinds[k], probes=probes,
                     score=max(errors), mean_error=float(np.mean(errors)), max_hole_fraction=max(missing))
        scores[a, k, b] = value
        return value
    current = list(keys); removed = []; audit = []
    target = max(1, math.floor((len(keys)-2)*p['removal_fraction'])) if len(keys)>2 else 0
    for _ in range(target):
        options = [score(a, k, b) for a, k, b in zip(current, current[1:], current[2:])
                   if k not in protected and b-a <= p['maximum_merged_gap']]
        if not options:
            break
        best = min(options, key=lambda row: (row['score'], row['key']))
        current.remove(best['key']); removed.append(best['key']); audit.append(best)
    return dict(original_keys=keys, protected_keys=sorted(protected), removed=removed,
                aggressive=current, single=[k for k in keys if k not in removed[:1]],
                ranking=audit, all_scores=list(scores.values()), parameters=p,
                scope='sender motion proxy only; actual whole-video reconstruction guard required')
