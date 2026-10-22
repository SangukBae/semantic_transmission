"""V4 sender-side risk prediction; frozen v3 checkpoints remain loadable by v3."""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import sparc_selector_budget as v3

ARCHITECTURE = 'sparc_risk_selector_v4'


class RiskSelector(v3.BudgetSelector):
    """Candidate-conditioned spatial errors survive both spatial and time pooling."""
    def __init__(self, **config):
        super().__init__(**config)
        width = self.config['width']
        self.delta = nn.Linear(width + 6, 2)
        self.delta_log_variance = nn.Linear(width + 6, 2)
        self.violation = nn.Linear(width + 6, 1)
        self.local_error = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, patches, motion, contexts, globals_, scenes, budgets):
        spatial = self.visual(patches.float()) + self.position
        pooled = (spatial * self.patch_score(spatial).softmax(1)).sum(1)
        z = (pooled + self.motion(motion.float())).unsqueeze(0) + self.context(contexts)
        local, chunks = [], []
        for sid in torch.unique_consecutive(scenes):
            ids = torch.nonzero(scenes == sid, as_tuple=True)[0]
            for block in ids.split(self.window):
                h = z[:, block]
                for layer in self.local:
                    h = layer(h)
                local.append(h)
                chunks.append(h.mean(1))
        temporal = torch.cat(local, 1)
        h = torch.stack(chunks, 1)
        for layer in self.global_time:
            h = layer(h)
        summary = torch.cat((self.norm(h).mean(1), globals_), 1)
        spatial = spatial.unsqueeze(0).expand(len(contexts), -1, -1, -1)
        temporal = temporal.unsqueeze(2).expand(-1, -1, spatial.shape[2], -1)
        return dict(absolute=self.distortion_rate(summary), log_var=self.log_variance(summary).clamp(-5, 3),
            delta=self.delta(summary), delta_log_var=self.delta_log_variance(summary).clamp(-5, 3),
            violation_logit=self.violation(summary).squeeze(-1),
            local_log_mse=self.local_error(torch.cat((spatial, temporal), -1)).squeeze(-1),
            score=self.policy(torch.cat((summary, budgets), 1)).squeeze(-1))


def quality_feasible(observation, baseline, goal):
    q, b = observation['quality'], baseline['quality']
    return (q['psnr_db'] >= b['psnr_db'] - goal['maximum_psnr_drop_db']
            and q['lpips_vgg'] <= b['lpips_vgg'] + goal['maximum_lpips_increase'])


def physical(output, normalization):
    """Absolute metrics use log coordinates; signed deltas never pass through exp."""
    device = output['absolute'].device
    n = {k: torch.as_tensor(v, device=device) for k, v in normalization.items()}
    return dict(rate=(output['absolute'][:, 4] * n['std'][4] + n['mean'][4]).clamp(-25, 15).exp(),
        rate_log_sigma=(output['log_var'][:, 4] / 2).exp() * n['std'][4],
        delta=output['delta'] * n['delta_std'] + n['delta_mean'],
        delta_sigma=(output['delta_log_var'] / 2).exp() * n['delta_std'],
        violation_probability=output['violation_logit'].sigmoid(),
        local_mse=(output['local_log_mse'] * n['local_std'] + n['local_mean']).clamp(-25, 5).exp(),
        score=output['score'])


def fit_calibration(records, alpha, *, split='calibration'):
    """One-sided empirical bounds, with one worst residual per source parent.

    These are validation corrections, not a guarantee under distribution shift.
    No test records or model-selection parents may enter this function.
    """
    if split != 'calibration' or not 0 < alpha < 1 or not records:
        raise ValueError('held-out calibration records and 0 < alpha < 1 required')
    parents = {}
    for r in records:
        delta = np.asarray(r['actual_delta'])
        pred = np.asarray(r['delta'])
        sigma = np.maximum(np.asarray(r['delta_sigma']), [.05, .001])
        value = np.array([(math.log(r['actual_rate']) - math.log(r['rate'])) / max(r['rate_log_sigma'], .01),
                          (pred[0] - delta[0]) / sigma[0], (delta[1] - pred[1]) / sigma[1],
                          r['local_log_underprediction']])
        if not np.isfinite(value).all():
            raise ValueError('nonfinite calibration residual')
        parents[r['parent']] = np.maximum(parents.get(r['parent'], value), value)
    count = len(parents)
    rank = math.ceil((count + 1) * (1 - alpha))
    if rank > count:
        raise ValueError('too few independent calibration parents for the requested alpha')
    offsets = np.maximum(0, np.sort(np.stack(list(parents.values())), axis=0)[rank - 1])
    return dict(schema='sparc_risk_calibration_v1', split=split, alpha=alpha,
                parents=sorted(parents), offsets=offsets.tolist(),
                scope='empirical one-sided parent-max corrections; no shifted-domain guarantee')


def conservative(values, calibration):
    if calibration.get('schema') != 'sparc_risk_calibration_v1':
        raise ValueError('v4 held-out calibration is required for selection')
    device = values['rate'].device
    offsets = torch.tensor(calibration['offsets'], device=device)
    if offsets.shape != (4,) or not torch.isfinite(offsets).all() or (offsets < 0).any():
        raise ValueError('invalid calibration offsets')
    sigma = values['delta_sigma'].clamp_min(torch.tensor([.05, .001], device=device))
    return dict(rate=values['rate'] * (offsets[0] * values['rate_log_sigma'].clamp_min(.01)).clamp(max=20).exp(),
        psnr=values['delta'][:, 0] - offsets[1] * sigma[:, 0],
        lpips=values['delta'][:, 1] + offsets[2] * sigma[:, 1],
        local=values['local_mse'] * offsets[3].clamp(max=20).exp())


class Predictor:
    def __init__(self, model, features, prepared, normalization, calibration, goal, device):
        self.model, self.features, self.ready = model, features, prepared
        self.normalization, self.calibration, self.goal, self.device = normalization, calibration, goal, device
        self.baseline_maps = {}

    @torch.inference_mode()
    def values(self, states, target, baseline_cbr):
        features = [f.to(self.device) for f in self.features]
        c, g, s = v3.contexts(len(features[0]), states, self.ready['keys'], self.ready['cuts'],
                              self.ready['source']['video']['fps'], self.device)
        out = self.model(*features, c, g, s, v3.budget_inputs(len(states), target, baseline_cbr, self.device))
        return physical(out, self.normalization)

    def assess(self, states, baseline, pixels, target):
        baseline_cbr = v3.cbr(baseline, pixels)
        key = tuple(baseline['identity']['indices'])
        if key not in self.baseline_maps:
            self.baseline_maps[key] = self.values([key], target, baseline_cbr)['local_mse'][0].cpu()
        maps, scores = [], []
        for start in range(0, len(states), 4):
            values = self.values(states[start:start + 4], target, baseline_cbr)
            bound = conservative(values, self.calibration)
            goal = self.goal
            penalty = (F.relu(-bound['psnr'] - goal['maximum_psnr_drop_db'])
                       + 100 * F.relu(bound['lpips'] - goal['maximum_lpips_increase']))
            score = (values['score'] - 4 * F.relu(bound['rate'] / target - 1)
                     - 4 * penalty - 2 * values['violation_probability'])
            scores.extend(score.cpu().tolist())
            base = self.baseline_maps[key].to(bound['local'].device).clamp_min(1e-6)
            # Excess relative to the same model's fixed adaptive reconstruction.
            maps.extend((bound['local'] / base).amax(-1).cpu().numpy())
        return np.asarray(scores), maps
