"""Risk-guided candidate pools and feedback from measured aggregate allocation."""
import math
import time

import numpy as np

from . import sparc_selector_budget as v3
from .sparc_selector_risk import quality_feasible
from .sparc_selector_train_v3 import allocate_cohort as measured_allocate


def frontier(observations, baseline):
    best, result = float('inf'), []
    for o in sorted(observations, key=lambda o: (o['channel']['complex_channel_uses'], v3.quality_cost(o, baseline))):
        cost = v3.quality_cost(o, baseline)
        if cost < best:
            result.append(o)
            best = cost
    return result


def leaders(observations, baseline, count, goal):
    """Keep the minimum-rate, safest, and intermediate measured trade-offs."""
    safe = [o for o in observations if quality_feasible(o, baseline, goal)]
    pool = frontier(observations, baseline)
    chosen = [pool[0], min(safe or observations, key=lambda o: v3.quality_cost(o, baseline)), baseline]
    chosen += [pool[int(i)] for i in np.linspace(0, len(pool)-1, max(1, count))]
    result = {tuple(o['identity']['indices']): o for o in chosen}
    return list(result.values())[:max(3, count)]


def risk_proposals(proxies, indices, keys, cuts, risk=None, errors=None, *, risk_ratio=2., risky_max_gap=4):
    candidates = set(v3.proposals(proxies, indices, keys, cuts, errors))
    if risk is None:
        return sorted(candidates)
    risk = np.asarray(risk)
    if risk.shape != (len(proxies),) or not np.isfinite(risk).all():
        raise ValueError('finite per-frame reconstruction risk required')
    # Insert anchors at risky *times*. Spatial predictions do not change the wire format.
    for a, b in zip(indices, indices[1:]):
        if b - a > 1 and risk[a + 1:b].max() > risk_ratio:
            i = a + 1 + int(np.argmax(risk[a + 1:b]))
            candidates.add(tuple(sorted(set(indices) | {i})))
    return sorted(s for s in candidates if not any(
        b-a > risky_max_gap and risk[a+1:b].max(initial=0) > risk_ratio
        and any(i not in s for i in indices if a < i < b)
        for a, b in zip(s, s[1:])))


def allocate_cohort(clips, target_cbr, goal):
    safe = [dict(c, observations=[o for o in c['observations'] if quality_feasible(o, c['baseline'], goal)])
            for c in clips]
    budget = math.floor(target_cbr * 3 * sum(c['pixels'] for c in clips))
    missing = [c['id'] for c in safe if not c['observations']]
    if missing:
        return dict(feasible=False, status='NO_QUALITY_FEASIBLE_CANDIDATE_FOUND', budget_uses=budget,
                    missing_quality_candidates=missing, minimum_observed_safe_uses=None, selected=[])
    result = measured_allocate(safe, target_cbr)
    minimum = sum(min(o['channel']['complex_channel_uses'] for o in c['observations']) for c in safe)
    result.update(minimum_observed_safe_uses=minimum, per_video_quality_met=bool(result['feasible']))
    if not result['feasible']:
        result['status'] = 'NO_OBSERVED_QUALITY_AND_BUDGET_FEASIBLE_COHORT'
        result['selected'] = []
    return result


class Session:
    def __init__(self, clip_id, proxies, keys, cuts, pixels, measure, cfg, goal, predictor=None, stop=lambda: None):
        self.id, self.proxies, self.keys, self.cuts, self.pixels = clip_id, proxies, keys, cuts, pixels
        self.measure, self.cfg, self.goal, self.predictor, self.stop = measure, cfg, goal, predictor, stop
        self.measured, self.visited = {}, set()
        self.calls = self.seed_calls = 0
        self.seconds = 0.
        self.exhausted = False

    def observe(self, state):
        state = tuple(state)
        if state in self.visited:
            return False
        self.stop()
        begin = time.monotonic()
        value = self.measure(list(state))
        self.seconds += time.monotonic() - begin
        self.visited.add(state)
        self.calls += 1  # Count rejected attempts and cache hits consistently in controls.
        if value is not None:
            self.measured[state] = value
        return True

    def initialize(self):
        states = v3.seeds(self.proxies, self.keys, self.cuts)
        for s in states:
            self.observe(s)
        self.seed_calls = self.calls
        if states[0] not in self.measured:
            raise ValueError('adaptive baseline failed real channel verification')
        self.baseline = self.measured[states[0]]

    def clip(self):
        return dict(id=self.id, pixels=self.pixels, baseline=self.baseline, observations=list(self.measured.values()))

    def step(self, target, count):
        begin = time.monotonic()
        pool = leaders(list(self.measured.values()), self.baseline, self.cfg['beam'], self.goal)
        risks = [None] * len(pool)
        if self.predictor:
            _, risks = self.predictor.assess([o['identity']['indices'] for o in pool], self.baseline, self.pixels, target)
        candidates = set()
        for o, risk in zip(pool, risks):
            candidates.update(risk_proposals(self.proxies, o['identity']['indices'], self.keys, self.cuts, risk,
                o['quality'].get('per_frame', {}).get('mse'), risk_ratio=self.cfg['risk_ratio'],
                risky_max_gap=self.cfg['risky_max_gap']))
        options = sorted(candidates - self.visited)
        if not options:
            self.exhausted = True
            return 0
        if self.predictor:
            scores, risks = self.predictor.assess(options, self.baseline, self.pixels, target)
            # Candidate-conditioned risks can reveal a newly dangerous gap after removal.
            allowed = [i for i, (s, risk) in enumerate(zip(options, risks)) if not any(
                b-a > self.cfg['risky_max_gap'] and np.max(risk[a+1:b], initial=0) > self.cfg['risk_ratio']
                for a, b in zip(s, s[1:]))]
            chosen = sorted(allowed, key=lambda i: (-scores[i], options[i]))[:count]
        else:
            # Teacher exploration spans densities; actual whole-packet cost decides acceptance.
            options.sort(key=lambda s: (len(s), s))
            chosen = np.linspace(0, len(options)-1, min(count, len(options))).round().astype(int).tolist()
        self.seconds += time.monotonic()-begin
        if not chosen:
            self.exhausted = True
        return sum(self.observe(options[i]) for i in chosen)


def search_cohort(sessions, target, cfg, goal, progress=lambda **k: None):
    """A fixed total measurement allowance, redistributed after initial exploration."""
    start = time.monotonic()
    for s in sessions:
        s.initialize()
        s.step(target, cfg['initial_per_clip'])
        progress(clip=s.id, calls=sum(x.calls for x in sessions))
    cap = sum(s.seed_calls for s in sessions) + cfg['measurements_per_clip'] * len(sessions)
    allocation = allocate_cohort([s.clip() for s in sessions], target, goal)
    rounds = 0
    while sum(s.calls for s in sessions) < cap:
        eligible = [s for s in sessions if not s.exhausted]
        if not eligible:
            break
        picked = {o['clip']: o for o in allocation.get('selected', [])}
        def priority(s):
            safe = [o for o in s.measured.values() if quality_feasible(o, s.baseline, goal)]
            lowest = min((o['channel']['complex_channel_uses'] for o in safe), default=float('inf'))
            chosen = picked.get(s.id)
            difficulty = (v3.quality_cost(chosen, s.baseline) if chosen else lowest/(3*s.pixels*target))
            return difficulty/(1 + max(0, s.calls-s.seed_calls-cfg['initial_per_clip']))
        s = max(eligible, key=lambda s: (priority(s), s.id))
        used_elsewhere = sum(o['channel_uses'] for key, o in picked.items() if key != s.id)
        allocated_target = ((allocation['budget_uses']-used_elsewhere)/(3*s.pixels)
                            if picked else target * cfg['budget_multipliers'][rounds % len(cfg['budget_multipliers'])])
        remaining = cap - sum(x.calls for x in sessions)
        s.step(max(allocated_target, 1e-9), min(cfg['per_round'], remaining))
        rounds += 1
        allocation = allocate_cohort([x.clip() for x in sessions], target, goal)
        progress(clip=s.id, calls=sum(x.calls for x in sessions), allocation_status=allocation['status'])
    return dict(cohort=allocation, clips=[s.clip() for s in sessions],
        measurement_attempts=sum(s.calls for s in sessions), measurement_cap=cap,
        seed_attempts=sum(s.seed_calls for s in sessions), feedback_rounds=rounds,
        elapsed_seconds=time.monotonic()-start, per_video_attempts={s.id:s.calls for s in sessions})
