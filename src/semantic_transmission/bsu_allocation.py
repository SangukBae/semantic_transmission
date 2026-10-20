"""Joint key/time/rate scoring; sender inputs only, no codec or LTX gradients.

The query-dependent sigmoid gate is applied AFTER attention, before the output
projection (G1-style headwise gating). This is a BSU adaptation, not a claim of
reproducing the Gated Attention paper's language-model results.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 71
    hidden: int = 256
    heads: int = 4
    temporal_layers: int = 3
    candidate_layers: int = 2
    metrics: int = 3
    dropout: float = 0.1
    gated: bool = True

    def __post_init__(self):
        for name in ('feature_dim', 'hidden', 'heads', 'temporal_layers', 'candidate_layers', 'metrics'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.hidden % self.heads or not 0 <= self.dropout < 1:
            raise ValueError('invalid attention dimensions/dropout')
        if type(self.gated) is not bool:
            raise ValueError('gated must be boolean')


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inner = math.ceil((8 * dim / 3) / 32) * 32
        self.up = nn.Linear(dim, 2 * inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, -1)
        return self.down(F.silu(gate) * value)


class GatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden
        self.heads, self.head_dim = config.heads, dim // config.heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.gate = nn.Linear(dim, config.heads) if config.gated else None
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = config.dropout

    def forward(self, query, memory, memory_mask):
        b, n, d = query.shape
        def heads(x):
            return x.reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        q = heads(self.q(query))
        k, v = self.kv(memory).chunk(2, -1)
        # SDPA boolean True means ALLOWED, unlike MultiheadAttention padding.
        result = F.scaled_dot_product_attention(q, heads(k), heads(v),
            attn_mask=memory_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0)
        if self.gate is not None:
            result = result * self.gate(query).sigmoid().transpose(1, 2).unsqueeze(-1)
        return self.out(result.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, config, cross=False):
        super().__init__()
        self.norm = RMSNorm(config.hidden)
        self.attention = GatedAttention(config)
        self.ff_norm = RMSNorm(config.hidden)
        self.ff = SwiGLU(config.hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.cross = GatedAttention(config) if cross else None
        self.cross_norm = RMSNorm(config.hidden)
        self.memory_norm = RMSNorm(config.hidden)

    def forward(self, x, mask, memory=None, memory_mask=None):
        z = self.norm(x)
        x = x + self.dropout(self.attention(z, z, mask))
        if self.cross is not None:
            x = x + self.dropout(self.cross(self.cross_norm(x), self.memory_norm(memory), memory_mask))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x.masked_fill(~mask[..., None], 0)


class JointAllocationPredictor(nn.Module):
    """Score each complete allocation independently of other candidate ordering.

    Frame features [B,T,F]; seconds [B,T]; key_positions/etas [B,C,K].
    Candidate tokens attend to each other AND the original temporal context.
    A summary token prevents all-masked rows for padded candidate slots.
    """
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        d = config.hidden
        self.features = nn.Linear(config.feature_dim, d)
        self.time = nn.Linear(10, d, bias=False)
        self.context = nn.Linear(3, d)  # log budget, SNR, log duration
        self.rate = nn.Linear(2, d)     # eta and log eta per key
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        self.temporal = nn.ModuleList(Block(config) for _ in range(config.temporal_layers))
        self.candidate = nn.ModuleList(Block(config, cross=True) for _ in range(config.candidate_layers))
        self.norm = RMSNorm(d)
        self.quality = nn.Linear(d, config.metrics)
        self.score = nn.Linear(d, 1)

    @staticmethod
    def time_features(times, mask):
        # Actual seconds preserve irregular intervals; normalized time alone
        # cannot distinguish a one-second clip from a ten-second clip.
        times = times.masked_fill(~mask, 0)
        duration = times.amax(-1).clamp_min(1e-6)
        phase = times[..., None] * times.new_tensor([0.25, 0.5, 1., 2.]) * math.pi
        return torch.cat((times[..., None] / duration[:, None, None],
            torch.log1p(times)[..., None], phase.sin(), phase.cos()), -1), duration

    def forward(self, features, times, frame_mask, key_positions, etas, key_mask,
                candidate_mask, budget, snr, baseline_index):
        b, t, _ = features.shape
        _, c, k = key_positions.shape
        if (not frame_mask.any(-1).all() or not candidate_mask.any(-1).all()
                or not torch.isfinite(features[frame_mask]).all()):
            raise ValueError('empty or nonfinite input')
        if (not torch.isfinite(times[frame_mask]).all() or (times[frame_mask] < 0).any()
                or not torch.isfinite(snr).all() or (budget <= 0).any()
                or not torch.isfinite(etas[key_mask]).all() or (etas[key_mask] <= 0).any()
                or (key_positions[key_mask] < 0).any() or (key_positions[key_mask] >= t).any()
                or (key_mask.sum(-1)[candidate_mask] < 2).any()):
            raise ValueError('invalid time/rate/key/budget context')
        if ((baseline_index < 0).any() or (baseline_index >= c).any()
                or not candidate_mask[torch.arange(b, device=features.device), baseline_index].all()):
            raise ValueError('invalid reference baseline')
        tf, duration = self.time_features(times, frame_mask)
        context = self.context(torch.stack((torch.log1p(budget.float()) / 16,
            snr / 20, torch.log1p(duration)), -1))
        x = self.features(features.masked_fill(~frame_mask[..., None], 0)) + self.time(tf)
        for block in self.temporal:
            x = block(x, frame_mask)
        positions = key_positions.masked_fill(~key_mask, 0)
        selected = x[torch.arange(b, device=x.device)[:, None, None], positions]
        safe_rates = etas.masked_fill(~key_mask, 1)
        rate = self.rate(torch.stack((safe_rates, safe_rates.log()), -1))
        keys = (selected + rate).reshape(b*c, k, -1)
        summary = self.summary.expand(b, c, -1) + context[:, None, :]
        y = torch.cat((summary.reshape(b*c, 1, -1), keys), 1)
        mask = torch.cat((torch.ones(b*c, 1, dtype=torch.bool, device=x.device),
                          key_mask.reshape(b*c, k)), 1)
        memory = x[:, None].expand(-1, c, -1, -1).reshape(b*c, t, -1)
        memory_mask = frame_mask[:, None].expand(-1, c, -1).reshape(b*c, t)
        for block in self.candidate:
            y = block(y, mask, memory, memory_mask)
        y = self.norm(y[:, 0]).reshape(b, c, -1)
        quality = self.quality(y)
        # Predict a difference from the explicitly supplied baseline, rather
        # than asking the model to guess which reference produced the labels.
        reference = quality[torch.arange(b, device=x.device), baseline_index]
        return dict(quality=quality-reference[:, None], scores=self.score(y).squeeze(-1))

    def specification(self):
        return asdict(self.config)


def feasible_mask(costs, budget, candidate_mask):
    """Integer comparison, no FP rounding at the hard budget boundary."""
    if costs.dtype != torch.long or budget.dtype != torch.long:
        raise ValueError('actual costs and budgets must be int64')
    if (costs[candidate_mask] <= 0).any() or (budget <= 0).any():
        raise ValueError('costs and budgets must be positive')
    return candidate_mask & (costs <= budget[:, None])


def choose(scores, costs, budget, candidate_mask):
    feasible = feasible_mask(costs, budget, candidate_mask)
    if not feasible.any(-1).all():
        raise ValueError('INFEASIBLE_REGISTERED_BUDGET: no feasible candidate')
    if not torch.isfinite(scores[candidate_mask]).all():
        raise ValueError('nonfinite candidate prediction')
    # Stable input-order tie break. Never fill a budget just for utilization.
    return scores.masked_fill(~feasible, -torch.inf).argmax(-1)


def allocation_loss(prediction, targets, utility, feasible, candidate_mask,
                    regression_weight=1., ranking_weight=1., regret_weight=1., temperature=0.5):
    """Huber metrics + gap-weighted pairwise rank + differentiable expected regret.

    Regret is relative to the BEST MEASURED FEASIBLE candidate, not a global
    oracle. Hard argmax is used at inference, softmax only during training.
    """
    if not feasible.any(-1).all() or temperature <= 0:
        raise ValueError('loss requires a feasible candidate and positive temperature')
    scores = prediction['scores']
    if not (torch.isfinite(targets[candidate_mask]).all()
            and torch.isfinite(utility[candidate_mask]).all()):
        raise ValueError('nonfinite labels')
    regression = F.smooth_l1_loss(prediction['quality'], targets, reduction='none').mean(-1)
    regression = ((regression * candidate_mask).sum(-1) / candidate_mask.sum(-1)).mean()
    gap = utility[:, :, None] - utility[:, None, :]
    pairs = feasible[:, :, None] & feasible[:, None, :] & (gap > 1e-6)
    pair_weight = gap.clamp(min=0, max=10) * pairs
    pair_loss = F.softplus(-(scores[:, :, None] - scores[:, None, :])) * pair_weight
    ranking = (pair_loss.sum((1, 2)) / pair_weight.sum((1, 2)).clamp_min(1e-6)).mean()
    masked = scores.masked_fill(~feasible, -torch.inf)
    probabilities = (masked / temperature).softmax(-1)
    best = utility.masked_fill(~feasible, -torch.inf).amax(-1)
    regret = (best - (probabilities * utility).sum(-1)).mean()
    total = regression_weight*regression + ranking_weight*ranking + regret_weight*regret
    return dict(loss=total, regression=regression, ranking=ranking, regret=regret)
