"""Candidate-specific ablations and source-cluster statistics for the campaign."""
import itertools

import numpy as np

from . import forbidden_state_metric as fso
from .metric_campaign_cases import PRIMARY
from .metric_v5_pipeline import BASELINES, compare_observations
from .event_duration_metric import evaluate_observations, sfr_observed
from .object_metric import score_tracks

ABLATIONS = {"fso_presence": ("fso_presence_raw",),
             "fso_premature": ("fso_premature_raw",),
             "fso_motion": ("fso_motion_raw", "fso_motion_absolute_speed"),
             "fso_heading": ("fso_heading_raw", "fso_heading_source_heading", "fso_heading_absolute_speed"),
             "eoi": ("eoi_missing_penalty", "eoi_kendall", "eoi_time_limited_matching"),
             "uep": ("uep_source_only", "uep_rx_only")}
ALL_METRICS = tuple(dict.fromkeys((*PRIMARY, *BASELINES, *(m for names in ABLATIONS.values() for m in names))))


def component_scores(a, b, values, rx, status):
    """Use the same RGB observations and entity associations for every ablation."""
    out = {}
    for state in ("motion", "heading"):
        absolute, source_heading = [], []
        for r in values["fso_records"]:
            if r["instance"] != state or r["status"] != "observed":
                continue
            anchor = round(r["source_time_s"] * 8)
            ai, bi, window = r["source_entity_tracks"], r["reconstruction_entity_tracks"], r["window_frames"]
            ha = fso.pre_heading(a["tracks"], ai, anchor)
            hb = fso.pre_heading(b["tracks"], bi, anchor)
            # Each worker scores sequentially; always restore the frozen scorer.
            original = fso.PARAMETERS["residual_speed_ratio"]
            try:
                fso.PARAMETERS["residual_speed_ratio"] = 0.
                x = sum(fso.occupancy(a["tracks"], ai, window, state, ha)) / 4
                y = sum(fso.occupancy(b["tracks"], bi, window, state, hb)) / 4
                absolute.append(max(0., y - x))
            finally:
                fso.PARAMETERS["residual_speed_ratio"] = original
            if state == "heading":
                x = sum(fso.occupancy(a["tracks"], ai, window, state, ha)) / 4
                y = sum(fso.occupancy(b["tracks"], bi, window, state, ha)) / 4
                source_heading.append(max(0., y - x))
        out["fso_" + state + "_absolute_speed"] = max(absolute) if absolute else None
        if state == "heading":
            out["fso_heading_source_heading"] = max(source_heading) if source_heading else None
    matches = values["eoi_matches"]
    source_pairs = [(i, j) for i, j in itertools.combinations(range(len(a["events"])), 2)
                    if a["events"][j]["time_s"] - a["events"][i]["time_s"] >= .375 - 1e-9]
    missing = sum(i not in matches or j not in matches for i, j in source_pairs)
    out["eoi_missing_penalty"] = ((values["eoi_inversions"] + missing) / len(source_pairs)
                                    if source_pairs else None)
    matched = [(i, j) for i, j in source_pairs if i in matches and j in matches]
    out["eoi_kendall"] = (sum(b["events"][matches[i]]["time_s"] > b["events"][matches[j]]["time_s"]
                               for i, j in matched) / len(matched) if matched else None)
    out['eoi_time_limited_matching'] = fso.order_inversions(a, b, dict(values['event_pairs']))['eoi']
    if not a["tracks"] or not b["tracks"]:
        out.update(uep_source_only=None, uep_rx_only=None)
    else:
        explained = {j for _, j in values["track_pairs"]}
        out["uep_source_only"] = max((min(1., len(t) / 4) for j, t in enumerate(b["tracks"])
                                       if j not in explained), default=0.)
        if rx is None or not status:
            out["uep_rx_only"] = None
        else:
            evidence = {}
            for i, j in score_tracks(b["tracks"], rx["tracks"]).get("track_pairs", []):
                evidence.setdefault(i, set()).update(t for t in rx["tracks"][j] if t < len(status) and status[t] == "ok")
            out["uep_rx_only"] = max(min(1., len(set(t) - evidence.get(j, set())) / 4)
                                      for j, t in enumerate(b["tracks"]))
    return out


def visual_scores(a, b, rx=None, status=None):
    result = compare_observations(a, b)
    duration = evaluate_observations(a, b)
    result.update({k: duration[k] for k in ("ghost_max", "delay_max")})
    result["sfr_inferred"] = sfr_observed(a, b, duration["entity_groups"])
    result.update(fso.evaluate(a, b))
    if rx is not None:
        result.update(fso.unsupported_presence(a, b, rx, status))
    result.update(component_scores(a, b, result, rx, status))
    return {name: result.get(name) for name in ALL_METRICS}


def finite(value):
    return value is not None and bool(np.isfinite(value))


def auc(positive, negative):
    from scipy.stats import rankdata
    p = [v for v in positive if finite(v)]
    n = [v for v in negative if finite(v)]
    if not p or not n:
        return None
    ranks = rankdata(p + n)
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def interval(values, config):
    if not values:
        return None
    x = np.asarray(values)
    rng = np.random.default_rng(config["natural_seed"])
    return np.quantile(x[rng.integers(len(x), size=(config["bootstrap_iterations"], len(x)))].mean(1), [.025, .975]).tolist()


def detection(rows, target, metric, threshold, config):
    applicable = [r for r in rows if finite(r["truth"].get(target))]
    positive = [r for r in applicable if r["truth"][target] > 0]
    negative = [r for r in applicable if r["truth"][target] == 0]
    def measure(group):
        scores = [r["scores"].get(metric) for r in group]
        decisions = [finite(s) and threshold is not None and s > threshold for s in scores]
        rates = [sum(decisions[i] for i, r in enumerate(group) if r["source_id"] == sid) /
                 sum(r["source_id"] == sid for r in group) for sid in sorted({r["source_id"] for r in group})]
        return {"cases": len(group), "sources": len(rates), "measured": sum(finite(s) for s in scores),
                "coverage": sum(finite(s) for s in scores) / len(group) if group else None,
                "rate": sum(decisions) / len(group) if group else None, "source_95ci": interval(rates, config)}
    p, n = measure(positive), measure(negative)
    area = auc([r["scores"].get(metric) for r in positive], [r["scores"].get(metric) for r in negative])
    sufficient = min(p["sources"], n["sources"]) >= config["minimum_positive_sources"]
    errors = [abs(r['scores'][metric] - r['truth'][target]) for r in applicable if finite(r['scores'].get(metric))]
    passed = bool(sufficient and threshold is not None and area is not None and area >= .9
                  and p["rate"] >= .8 and n["rate"] <= .1 and min(p["coverage"], n["coverage"]) >= .95)
    return {"auc": area, "positive": p, "negative": n, "threshold": threshold,
            "observed_oracle_mean_absolute_error": float(np.mean(errors)) if errors else None,
            "unlabelled_or_inapplicable": len(rows) - len(applicable),
            "status": "PASSED" if passed else "NOT_PASSED" if sufficient else "INSUFFICIENT_TRUTH"}


def paired_auc(rows, target, candidate, baseline, config):
    group = [r for r in rows if finite(r["truth"].get(target))
             and finite(r["scores"].get(candidate)) and finite(r["scores"].get(baseline))]
    differences = []
    for sid in sorted({r["source_id"] for r in group}):
        p = [r for r in group if r["source_id"] == sid and r["truth"][target] > 0]
        n = [r for r in group if r["source_id"] == sid and r["truth"][target] == 0]
        a = auc([r["scores"][candidate] for r in p], [r["scores"][candidate] for r in n])
        b = auc([r["scores"][baseline] for r in p], [r["scores"][baseline] for r in n])
        if a is not None and b is not None:
            differences.append(a - b)
    ci = interval(differences, config)
    return {"sources": len(differences), 'paired_measured_cases': len(group),
            'cases_with_different_scores': sum(r['scores'][candidate] != r['scores'][baseline] for r in group),
            "mean_auc_difference": float(np.mean(differences)) if differences else None,
            "source_95ci": ci, "superiority_supported": bool(len(differences) >= config["minimum_positive_sources"] and ci[0] > 0)}


def analyse(rows, thresholds, config):
    result = {}
    for target in PRIMARY:
        comparisons = tuple(dict.fromkeys((*BASELINES, *ABLATIONS[target])))
        result[target] = {"overall": detection(rows, target, target, thresholds.get(target), config),
                          "by_style": {style: detection([r for r in rows if r["style"] == style], target, target,
                                                         thresholds.get(target), config)
                                       for style in sorted({r["style"] for r in rows})},
                          "comparisons": {name: paired_auc(rows, target, target, name, config) for name in comparisons},
                          "baseline_detection": {name: detection(rows, target, name, thresholds.get(name), config)
                                                 for name in comparisons}}
        result[target]["all_styles_passed"] = all(x["status"] == "PASSED" for x in result[target]["by_style"].values())
    return result
