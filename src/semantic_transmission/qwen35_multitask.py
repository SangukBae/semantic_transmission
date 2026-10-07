"""One frozen Qwen vision encoder, temporal heads, and caption LoRA.

The straight-through subset distribution makes caption CE differentiable with
respect to candidate and selection heads. Forward/inference always uses real,
ordered source frames, with a fixed key budget and both endpoints retained.
"""
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalSelector(nn.Module):
    def __init__(self, width=4096, hidden=128, candidates=8, keys=4):
        super().__init__()
        if not 2 <= keys <= candidates <= 16:
            raise ValueError("Invalid key budget")
        self.norm = nn.LayerNorm(width)
        self.project = nn.Linear(width, hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.candidate = nn.Linear(hidden, 2)  # before/after evidence
        self.selection = nn.Linear(hidden, 1)
        subsets = [(0, *c, candidates - 1) for c in combinations(range(1, candidates - 1), keys - 2)]
        self.register_buffer("subsets", torch.tensor(subsets, dtype=torch.long))
        self.register_buffer("matrices", F.one_hot(self.subsets, candidates).float())

    def forward(self, pooled, temperature=1.0):
        if pooled.ndim != 2 or pooled.shape[0] != self.matrices.shape[-1] or temperature <= 0:
            raise ValueError("Wrong candidate shape/temperature")
        x = F.gelu(self.project(self.norm(pooled.float())))
        x = x + F.gelu(self.temporal(x.T.unsqueeze(0)).squeeze(0).T)
        candidate = self.candidate(x)
        importance = self.selection(x).squeeze(-1) + candidate.sigmoid().mean(-1)
        scores = importance[self.subsets[:, 1:-1]].sum(-1)
        probabilities = (scores / temperature).softmax(-1)
        winner = scores.argmax()
        hard = F.one_hot(winner, len(scores)).to(probabilities)
        straight_through = hard + (probabilities - probabilities.detach())
        matrix = torch.einsum("s,skn->kn", straight_through, self.matrices)
        return {"candidate_logits": candidate, "probabilities": probabilities,
                "matrix": matrix, "indices": self.subsets[winner], "importance": importance}

    def select_tokens(self, tokens, prediction):
        # FP32 weights retain the small selector gradients; cast only final tokens.
        return torch.einsum("kn,npd->kpd", prediction["matrix"], tokens.float()).to(tokens.dtype)


def targets_for(row, device="cpu"):
    candidates = torch.zeros((len(row["frames"]), 2), device=device)
    for e in row["events"]:
        candidates[e["before"], 0] = 1
        candidates[e["after"], 1] = 1
    return candidates


def coverage(row, indices):
    keys = set(int(i) for i in indices)
    events = [bool(keys.intersection(e["before"])) and bool(keys.intersection(e["after"])) for e in row["events"]]
    facts = [bool(keys.intersection(f["evidence"])) for f in row["facts"]]
    return {"event_hits": sum(events), "event_count": len(events),
            "fact_evidence_hits": sum(facts), "fact_count": len(facts)}


def supervised_head_loss(head, prediction, row):
    logits = prediction["candidate_logits"]
    target = targets_for(row, logits.device)
    # Sparse positives get a bounded weight; static windows remain true negatives.
    candidate_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(3.0))
    penalties = []
    for subset in head.subsets.tolist():
        c = coverage(row, subset)
        missed_events = 1 - c["event_hits"] / c["event_count"] if c["event_count"] else 0
        missed_facts = 1 - c["fact_evidence_hits"] / c["fact_count"]
        # Mild spreading only breaks otherwise equivalent evidence coverage.
        gaps = [b - a for a, b in zip(subset, subset[1:])]
        spread = (max(gaps) - min(gaps)) / len(row["frames"])
        penalties.append(2 * missed_events + missed_facts + 0.02 * spread)
    evidence_loss = (prediction["probabilities"] * logits.new_tensor(penalties)).sum()
    return candidate_loss + evidence_loss, {"candidate_bce": float(candidate_loss.detach()), "evidence_loss": float(evidence_loss.detach())}
