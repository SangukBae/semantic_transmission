"""Budget-conditioned selection with measured, non-additive wire cost.

All visual inputs are sender-side. A candidate is deployable only after the
real codec/channel/receiver evaluator has accepted it within its wire budget.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .sparc_selector_cnn import state_context
from .sparc_selection import adaptive, validate_anchors

ARCHITECTURE = 'sparc_budget_selector_v3'
METRICS = ('mse', 'lpips_vgg', 'temporal_mse', 'local_mae95')


def required(n, keys, cuts):
    return set(keys) | {0, n-1} | {i for c in cuts for i in (c-1, c)}


def seeds(proxies, keys, cuts):
    """The first candidate exactly follows the previous 64x36 QP30 selector."""
    import cv2
    small = np.stack([cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA) for f in proxies])
    values = [adaptive(small, keys, cuts, max_gap=4, threshold=6.)['indices'],
              sorted(required(len(small), keys, cuts))]
    values += [adaptive(small, keys, cuts, max_gap=None, threshold=t)['indices']
               for t in (6., 12., 24.)]
    return list(dict.fromkeys(tuple(s) for s in values))


def proposals(proxies, chosen, keys, cuts, reconstruction_errors=None):
    """Joint removals/additions, relocation and density changes; no gap cap."""
    x = proxies.astype(np.float32)/255
    chosen = sorted(chosen)
    mandatory = required(len(x), keys, cuts)
    remove, add = [], []
    for a, b in zip(chosen, chosen[1:]):
        for i in range(a+1, b):
            estimate = x[a]+(x[b]-x[a])*((i-a)/(b-a))
            error = (float(reconstruction_errors[i]) if reconstruction_errors is not None
                     else float(np.abs(x[i]-estimate).mean()))
            add.append((error, i))
    for k, i in enumerate(chosen[1:-1], 1):
        if i not in mandatory:
            a, b = chosen[k-1], chosen[k+1]
            estimate = x[a]+(x[b]-x[a])*((i-a)/(b-a))
            remove.append((float(np.abs(x[i]-estimate).mean()), i))
    removable = [i for _, i in sorted(remove)]
    additions = [i for _, i in sorted(add, reverse=True)]
    states = set()
    for count in (1, 2, 4, 8):
        if removable:
            states.add(tuple(sorted(set(chosen)-set(removable[:count]))))
        if additions:
            states.add(tuple(sorted(set(chosen)|set(additions[:count]))))
        if removable and additions:
            states.add(tuple(sorted((set(chosen)-set(removable[:count]))|set(additions[:count]))))
    for phase in (0, 1):
        optional = [i for i in chosen if i not in mandatory]
        states.add(tuple(sorted(mandatory | set(optional[phase::2]))))
    # A whole interval can be simplified even when its individual removals score poorly.
    for a, b in sorted(zip(chosen, chosen[1:]), key=lambda ab: ab[1]-ab[0], reverse=True)[:3]:
        if b-a > 1:
            states.add(tuple(sorted(set(chosen)|{(a+b)//2})))
    states.discard(tuple(chosen))
    for s in states:
        validate_anchors(list(s), len(x), None, cuts)
        if not mandatory.issubset(s):
            raise ValueError('proposal removed a protected anchor')
    return sorted(states)


def cbr(observation, pixels):
    """Paper convention: complex channel uses / (3 * F * H * W)."""
    return observation['channel']['complex_channel_uses']/(3*pixels)


def quality_cost(observation, baseline):
    q, b = observation['quality'], baseline['quality']
    # Original-reference distortion, normalized by the previous QP30 reconstruction.
    return sum(w*q[k]/max(b[k], floor) for k, w, floor in
               [('mse', 1., 1e-5), ('lpips_vgg', 1., .01),
                ('temporal_mse', .25, 1e-5), ('local_mae95', .25, .01)])


def order_key(observation, baseline, budget_uses):
    rate = observation['channel']['complex_channel_uses']
    if rate <= budget_uses:
        return (0, quality_cost(observation, baseline), rate)
    return (1, rate/budget_uses, quality_cost(observation, baseline))


def decision(observations, baseline, budget_uses):
    feasible = [o for o in observations if o['channel']['complex_channel_uses'] <= budget_uses]
    if not feasible:
        return dict(status='NO_FEASIBLE_CANDIDATE_FOUND', indices=None,
                    budget_uses=budget_uses, feasible=False)
    best = min(feasible, key=lambda o: order_key(o, baseline, budget_uses))
    return dict(status='MEASURED_WITHIN_BUDGET', indices=best['identity']['indices'],
                budget_uses=budget_uses, feasible=True, observation=best,
                psnr_delta=best['quality']['psnr_db']-baseline['quality']['psnr_db'],
                lpips_delta=best['quality']['lpips_vgg']-baseline['quality']['lpips_vgg'])


class BudgetSelector(nn.Module):
    """Spatial patch attention, local time attention and global allocation context."""
    def __init__(self, feature_dim=1024, motion_dim=19, width=256, heads=8,
                 local_layers=2, global_layers=4, window=24, dropout=.1):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, motion_dim=motion_dim, width=width,
            heads=heads, local_layers=local_layers, global_layers=global_layers,
            window=window, dropout=dropout)
        self.window = window
        self.visual = nn.Linear(feature_dim, width)
        self.position = nn.Parameter(torch.zeros(1, 15, width))
        self.patch_score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.motion = nn.Linear(motion_dim, width)
        self.context = nn.Linear(18, width)
        def layer():
            return nn.TransformerEncoderLayer(width, heads, 4*width, dropout,
                activation='gelu', batch_first=True, norm_first=True)
        self.local = nn.ModuleList([layer() for _ in range(local_layers)])
        self.global_time = nn.ModuleList([layer() for _ in range(global_layers)])
        self.norm = nn.LayerNorm(width)
        self.distortion_rate = nn.Sequential(nn.Linear(width+6, width), nn.GELU(), nn.Linear(width, 5))
        self.log_variance = nn.Linear(width+6, 5)
        self.policy = nn.Sequential(nn.Linear(width+6+2, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, patches, motion, contexts, globals_, scenes, budgets):
        z = self.visual(patches.float()) + self.position
        z = (z*self.patch_score(z).softmax(1)).sum(1)+self.motion(motion.float())
        z = z.unsqueeze(0)+self.context(contexts)
        chunks = []
        # Local attention never mixes unrelated shots. Global tokens allocate across shots.
        for sid in torch.unique_consecutive(scenes):
            ids = torch.nonzero(scenes == sid, as_tuple=True)[0]
            for block in ids.split(self.window):
                h = z[:, block]
                for layer in self.local:
                    h = layer(h)
                chunks.append(h.mean(1))
        h = torch.stack(chunks, 1)
        for layer in self.global_time:
            h = layer(h)
        pooled = torch.cat((self.norm(h).mean(1), globals_), 1)
        prediction = self.distortion_rate(pooled)
        variance = self.log_variance(pooled).clamp(-5, 3)
        # Physical rate/distortion heads do not depend on the requested budget.
        score = self.policy(torch.cat((pooled, budgets), 1)).squeeze(-1)
        return prediction, variance, score


def contexts(n, states, keys, cuts, fps, device):
    data = [state_context(n, s, keys, cuts, fps) for s in states]
    return (torch.as_tensor(np.stack([a[0] for a in data]), device=device),
            torch.as_tensor(np.stack([a[1] for a in data]), device=device),
            torch.as_tensor(data[0][2], device=device))


def budget_inputs(count, target_cbr, baseline_cbr, device):
    if min(target_cbr, baseline_cbr) <= 0:
        raise ValueError('positive absolute budget and baseline rate required')
    return torch.tensor([math.log(target_cbr/.00065), math.log(target_cbr/baseline_cbr)],
                        dtype=torch.float32, device=device).expand(count, -1)


def predict(model, features, states, prepared, target_cbr, baseline_cbr, normalization, device):
    patches, motion = (v.to(device) for v in features)
    c, g, s = contexts(len(patches), states, prepared['keys'], prepared['cuts'],
                       prepared['source']['video']['fps'], device)
    y, var, score = model(patches, motion, c, g, s,
                         budget_inputs(len(states), target_cbr, baseline_cbr, device))
    mean, std = (normalization[k].to(device) for k in ('mean', 'std'))
    return (y*std+mean).exp(), var.exp().sqrt()*std, score


def search(proxies, keys, cuts, budget_uses, evaluate, *, known=(), rank=None,
           rounds=3, beam=4, per_round=4, stop=lambda: None):
    """Bounded beam search; all returned candidates have actual paid-rate evidence.

    Teacher search may reuse measured training candidates. Learned validation
    starts only with source-derived seeds, and never sees teacher candidates.
    A ranker proposes measurements; it cannot authorize an over-budget result.
    """
    measured = {tuple(o['identity']['indices']): o for o in known}
    initial = seeds(proxies, keys, cuts)
    for s in initial:
        stop()
        if s not in measured:
            value = evaluate(list(s))
            if value is not None:
                measured[s] = value
    if initial[0] not in measured:
        raise ValueError('baseline channel failed; quality comparison cannot be certified')
    baseline = measured[initial[0]]
    visited = set(measured)
    for _ in range(rounds):
        stop()
        leaders = sorted(measured, key=lambda s: order_key(measured[s], baseline, budget_uses))[:beam]
        proposed = sorted({p for s in leaders for p in proposals(proxies, s, keys, cuts,
            measured[s]['quality'].get('per_frame', {}).get('mse'))}-visited)
        if not proposed:
            break
        if rank is None:
            # Teacher measures both sparse and dense directions, independent of frame-rate linearity.
            ordered = sorted(proposed, key=lambda s: (len(s), s))
            ids = np.linspace(0, len(ordered)-1, min(per_round, len(ordered))).round().astype(int)
            candidates = [ordered[i] for i in ids]
        else:
            scores = np.asarray(rank(proposed, baseline), dtype=float)
            if scores.shape != (len(proposed),) or not np.isfinite(scores).all():
                raise ValueError('invalid candidate scores')
            candidates = [proposed[i] for i in np.argsort(-scores, kind='stable')[:per_round]]
        for s in candidates:
            stop()
            value = evaluate(list(s))
            if value is not None:
                measured[s] = value
            visited.add(s)
    result = decision(measured.values(), baseline, budget_uses)
    return result, list(measured.values()), baseline
