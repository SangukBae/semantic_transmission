"""Source-separated ERE/STA preparation and mandatory pre-score audits.

No candidate pair scores may be written until all required extractor gates pass.
Ground-truth audit results are never inputs to the RGB event/object extractors.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import cv2
import numpy as np

from .artifacts import sha256
from .automatic_validation import write
from .metric_v3_cases import COUNT, FPS, HEIGHT, WIDTH, annotation_events, audit_truth, public_truth, render_scene, variants

FILES = ("event_metric.py", "attribution_metric.py", "metric_v3_cases.py", "metric_v3_validation.py",
         "object_metric.py", "motion_metric.py", "automatic_metrics.py", "temporal_baselines.py",
         "tracking_baselines.py", "automatic_validation.py", "metric_v2_cases.py")
OLD_RUNS = ("outputs/automatic_validation_20260912_v1", "outputs/mte_otf_20260914_v1")


def read(path):
    return json.loads(Path(path).read_text())


def pixel_sha(frames):
    return hashlib.sha256(frames.tobytes()).hexdigest()


def no_scores(root):
    if list((root / "scores").rglob("*.json")):
        raise ValueError("candidate scores already exist: pre-score changes forbidden")


def verify(root):
    protocol = read(root / "protocol.json")
    for name, digest in protocol["code_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("frozen implementation changed: " + name)
    prepared = read(root / "preparation.json")
    for name in ("protocol.json", "cases.jsonl"):
        if sha256(root / name) != prepared[name + "_sha256"]:
            raise ValueError("frozen artifact changed: " + name)
    for name, digest in protocol["script_sha256"].items():
        if sha256(Path("scripts") / name) != digest:
            raise ValueError("frozen script changed: " + name)
    for name, key in (("ERE_STA_PROTOCOL.frozen.md", "definition_sha256"),
                      ("ERE_STA_EXECUTION_NOTES.frozen.md", "execution_notes_sha256")):
        if sha256(root / name) != protocol[key]:
            raise ValueError("frozen definition changed: " + name)
    if sha256(root / 'development_diagnostics.json') != protocol['development_diagnosis']['sha256']:
        raise ValueError('frozen development diagnosis changed')
    return protocol


def require_gates(root):
    verify(root)
    gate_path = root / "extractor_gate.json"
    if not gate_path.exists() or read(gate_path)["status"] != "PASSED":
        raise ValueError("ERE_STA_PROTOCOL section 6: mandatory extractor gate not passed; no candidate scoring permitted")
    if read(root / "truth_audit.json")["status"] != "PASSED":
        raise ValueError("truth audit not passed")


def prepare(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists() or (root / "inputs").exists():
        raise FileExistsError("prepare requires a fresh run")
    previous = [read(Path(p) / "protocol.json") for p in OLD_RUNS]
    excluded = {x["source_id"] for p in previous for x in p["inventory"]}
    seeds = {x["seed"] for p in previous for x in p["inventory"] if "seed" in x}
    for p in previous:
        for values in p.get("synthetic_seeds", {}).values():
            seeds.update(values)
    inventory = []
    data = args.data / "DAVIS"
    for official, split, count in (("train", "development", 8), ("val", "heldout", 14)):
        names = (data / "ImageSets/2017" / (official + ".txt")).read_text().split()
        names = [n for n in names if "davis/" + n not in excluded]
        names.sort(key=lambda n: hashlib.sha256(("metric-v3-20260914/" + n).encode()).hexdigest())
        if len(names) < count:
            raise ValueError("not enough unused DAVIS sources")
        for name in names[:count]:
            paths = sorted((data / "JPEGImages/480p" / name).glob("*.jpg"))[::3][:COUNT]
            inventory.append({"source_id": "davis/" + name, "domain": "public_video", "split": split,
                "files": [{"path": str(p.resolve()), "sha256": sha256(p),
                           "annotation_path": str(Path(str(p).replace("JPEGImages", "Annotations")).with_suffix(".png").resolve()),
                           "annotation_sha256": sha256(Path(str(p).replace("JPEGImages", "Annotations")).with_suffix(".png"))} for p in paths]})
    for split, values in (("development", range(260915100, 260915116)), ("heldout", range(260915900, 260915932))):
        for seed in values:
            if seed in seeds or f"renderer/{seed}" in excluded:
                raise ValueError("old renderer seed/source reused")
            inventory.append({"source_id": f"renderer/{seed}", "domain": "rendered", "split": split, "seed": seed})
    if any(x["source_id"] in excluded for x in inventory):
        raise ValueError("old source reused")
    for name in ("inputs", "cases", "case_truth", "frozen_source", "scores", "audit"):
        (root / name).mkdir()
    from .event_metric import PARAMETERS
    diagnosis = read(args.diagnostics)
    if diagnosis['status'] != 'COMPLETED' or diagnosis['heldout_sources_used'] != 0 or diagnosis['formal_candidate_scores_observed'] != 0:
        raise ValueError('development-only diagnosis must finish before prepare')
    if diagnosis['event_code_sha256'] != sha256(Path(__file__).with_name('event_metric.py')):
        raise ValueError('event implementation changed after parameter diagnosis')
    protocol = {"schema": "ere-sta-v3.1-20260914", "created_unix": time.time(), "inventory": inventory,
        "fps": FPS, "resolution": [WIDTH, HEIGHT], "max_frames": COUNT,
        "code_sha256": {n: sha256(Path(__file__).with_name(n)) for n in FILES},
        "script_sha256": {n: sha256(Path('scripts') / n) for n in ("run_metric_v3.sh", "report_metric_v3.py", "diagnose_event_v31.py")},
        "definition_sha256": sha256(Path("docs/ERE_STA_PROTOCOL.md")),
        "execution_notes_sha256": sha256(Path("docs/ERE_STA_EXECUTION_NOTES.md")),
        "previous_protocols": {p: sha256(Path(p) / "protocol.json") for p in OLD_RUNS},
        "old_source_overlap": [], "old_seed_overlap": [], "models": previous[1]["models"],
        "event_parameters": {**PARAMETERS, **diagnosis['selected']['parameters']},
        "development_diagnosis": {'path': str(args.diagnostics), 'sha256': sha256(args.diagnostics),
                                  'selected': diagnosis['selected'], 'heldout_sources_used': 0},
        "pre_score_amendment": read(Path('docs/ERE_STA_EXECUTION_AMENDMENTS.json')),
        "scores_observed": 0, "new_human_review": False,
        "truth_inputs": "annotations/renderer state only in truth audit and extractor gates, never RGB candidate inputs",
        "gates": {"event_recall": .8, "event_precision": .8, "turn_recall": .8, "rendered_object_recall": .8,
                  "sta_rule_checks": 'all_six', "public_object_recall": .8,
                  "public_object_policy": "diagnostic_only_predeclared", "public_event_policy": 'diagnostic_only_predeclared'},
        "claim_scope": "pilot; no decoder causal effect, human agreement, or novelty claim"}
    write(root / "protocol.json", protocol)
    for name in FILES:
        shutil.copy2(Path(__file__).with_name(name), root / "frozen_source" / name)
    for name in protocol["script_sha256"]:
        shutil.copy2(Path('scripts') / name, root / "frozen_source" / name)
    shutil.copy2("docs/ERE_STA_PROTOCOL.md", root / "ERE_STA_PROTOCOL.frozen.md")
    shutil.copy2("docs/ERE_STA_EXECUTION_NOTES.md", root / "ERE_STA_EXECUTION_NOTES.frozen.md")
    shutil.copy2(args.diagnostics, root / 'development_diagnostics.json')
    if args.reuse_inputs:
        _reuse_inputs(root, args.reuse_inputs, inventory)
        return
    count = 0
    with (root / "cases.jsonl").open('w') as manifest:
        for item in inventory:
            stem = item["source_id"].replace('/', '__')
            if item["domain"] == "rendered":
                frames, truth = render_scene(item["seed"])
                events = annotation_events(truth["centers"], truth["visible"])
                if len(events) < 4:
                    raise ValueError("renderer source has fewer than four true events")
            else:
                from PIL import Image
                frames = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p["path"]), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA) for p in item["files"]])
                labels = np.stack([cv2.resize(np.asarray(Image.open(p["annotation_path"])), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) for p in item["files"]])
                truth = public_truth(labels)
            np.savez_compressed(root / 'inputs' / (stem + '.npz'), frames=frames)
            np.savez_compressed(root / 'inputs' / (stem + '_truth.npz'), **truth)
            for name, rec, rt, metadata in variants(frames, truth, item.get("seed")):
                cid = stem + '__' + name
                np.savez_compressed(root / 'cases' / (cid + '.npz'), frames=rec)
                # Truth is physically separate from the metric RGB artifacts.
                np.savez_compressed(root / 'case_truth' / (cid + '.npz'), **rt)
                row = {"case_id": cid, "source_id": item["source_id"], "source_stem": stem,
                       "split": item["split"], "domain": item["domain"], "variant": name, **metadata,
                       "source_pixel_sha256": pixel_sha(frames), "reconstruction_pixel_sha256": pixel_sha(rec),
                       "source_truth_sha256": sha256(root / 'inputs' / (stem + '_truth.npz')),
                       "reconstruction_truth_sha256": sha256(root / 'case_truth' / (cid + '.npz'))}
                manifest.write(json.dumps(row) + '\n'); count += 1
            print('prepared', item['source_id'], count, flush=True)
    write(root / 'preparation.json', {"sources": len(inventory), "cases": count,
          "protocol.json_sha256": sha256(root / 'protocol.json'), "cases.jsonl_sha256": sha256(root / 'cases.jsonl')})


def _reuse_inputs(root, previous, inventory):
    """Copy immutable v3.0 pixels/state, verifying every copied RGB/truth payload."""
    if root.resolve() == previous.resolve():
        raise ValueError('old run must not be overwritten')
    old, prep = read(previous / 'protocol.json'), read(previous / 'preparation.json')
    if old['inventory'] != inventory:
        raise ValueError('reuse inventory mismatch')
    for name in ('protocol.json', 'cases.jsonl'):
        if sha256(previous / name) != prep[name + '_sha256']:
            raise ValueError('old run manifest changed')
    if old['code_sha256']['metric_v3_cases.py'] != sha256(Path(__file__).with_name('metric_v3_cases.py')):
        raise ValueError('renderer changed; source data cannot be reused')
    rows = [json.loads(line) for line in (previous / 'cases.jsonl').read_text().splitlines()]
    seen = set()
    for row in rows:
        if row['source_id'] not in seen:
            source = previous / 'inputs' / (row['source_stem'] + '.npz')
            truth = previous / 'inputs' / (row['source_stem'] + '_truth.npz')
            with np.load(source) as f:
                if pixel_sha(f['frames']) != row['source_pixel_sha256']:
                    raise ValueError('old source pixel hash mismatch')
            if sha256(truth) != row['source_truth_sha256']:
                raise ValueError('old source truth mismatch')
            for p in (source, truth):
                shutil.copy2(p, root / 'inputs' / p.name)
            seen.add(row['source_id'])
        case = previous / 'cases' / (row['case_id'] + '.npz')
        truth = previous / 'case_truth' / case.name
        with np.load(case) as f:
            if pixel_sha(f['frames']) != row['reconstruction_pixel_sha256']:
                raise ValueError('old case pixel mismatch')
        if sha256(truth) != row['reconstruction_truth_sha256']:
            raise ValueError('old case truth mismatch')
        shutil.copy2(case, root / 'cases' / case.name)
        shutil.copy2(truth, root / 'case_truth' / truth.name)
    shutil.copy2(previous / 'cases.jsonl', root / 'cases.jsonl')
    write(root / 'preparation.json', {'sources': len(seen), 'cases': len(rows),
          'protocol.json_sha256': sha256(root / 'protocol.json'), 'cases.jsonl_sha256': sha256(root / 'cases.jsonl'),
          'data_reused_from': str(previous), 'all_rgb_and_truth_hashes_verified': True,
          'copy_policy': 'independent file copies, no symlinks or hardlinks',
          'scope': 'same pre-score corpus; not a new independent data sample'})
    print('prepared verified copies', len(seen), len(rows), flush=True)


def truth_audit(args):
    root = args.output; verify(root); no_scores(root)
    rows = [json.loads(s) for s in (root / 'cases.jsonl').read_text().splitlines()]
    audited = []; previous = None
    for row in rows:
        if previous != row['source_id']:
            path = root / 'inputs' / (row['source_stem'] + '_truth.npz')
            if sha256(path) != row['source_truth_sha256']:
                raise ValueError('source truth changed')
            with np.load(path) as f:
                source = dict(f)
            previous = row['source_id']
        path = root / 'case_truth' / (row['case_id'] + '.npz')
        if sha256(path) != row['reconstruction_truth_sha256']:
            raise ValueError('case truth changed')
        with np.load(path) as f:
            result = audit_truth(source, dict(f), row['kind'], row['variant'])
        audited.append({"case_id": row['case_id'], "source_id": row['source_id'], "domain": row['domain'],
                        "split": row['split'], "kind": row['kind'], "variant": row['variant'], **result})
    write(root / 'truth_audit.json', {"status": "PASSED", "scores_observed": 0,
        "total": len(rows), "accepted": sum(x['accepted'] for x in audited),
        "rejected": sum(not x['accepted'] for x in audited),
        "reasons": dict(Counter(x['reason'] for x in audited)), "results": audited,
        "scope": "truth eligibility only; rejected cases cannot become positive or normal samples"})
    print('truth audit', len(audited), Counter(x['reason'] for x in audited), flush=True)


def extractor_audit(args):
    root = args.output; protocol = verify(root); no_scores(root)
    if not (root / 'truth_audit.json').exists():
        raise ValueError('truth audit must run first')
    import torch
    torch.set_num_threads(4); torch.manual_seed(20260914); cv2.setNumThreads(1)
    if args.part == 'event':
        from .event_metric import EventMetric, event_pairs
        extractor = EventMetric(model_root=protocol['models']['root'], cache_root=root / 'object_cache',
                                parameters=protocol['event_parameters'])
    else:
        from .object_metric import ObjectExtractor
        extractor = ObjectExtractor(protocol['models']['root'], root / 'object_cache')
    results = []
    source_rows = {r['source_id']: r for r in (json.loads(s) for s in (root / 'cases.jsonl').read_text().splitlines())}
    for item in protocol['inventory']:
        if item['split'] != 'development':
            continue
        stem = item['source_id'].replace('/', '__')
        dest = root / 'audit' / (args.part + '__' + stem + '.json')
        if dest.exists():
            results.append(read(dest)); continue
        with np.load(root / 'inputs' / (stem + '.npz')) as f:
            frames = f['frames']
        if pixel_sha(frames) != source_rows[item['source_id']]['source_pixel_sha256']:
            raise ValueError('source RGB bytes changed')
        if sha256(root / 'inputs' / (stem + '_truth.npz')) != source_rows[item['source_id']]['source_truth_sha256']:
            raise ValueError('source audit truth changed')
        with np.load(root / 'inputs' / (stem + '_truth.npz')) as f:
            truth = dict(f)
        row = {"source_id": item['source_id'], "domain": item['domain'], "split": item['split']}
        if args.part == 'event':
            extracted = extractor.extract(frames)
            actual = annotation_events(truth['centers'], truth['visible'])
            reference = [{**e, "track_id": e['object_id']} for e in actual]
            pairs = event_pairs(reference, extracted['events'])
            # Audit fragmentation with true identities, separately from RGB extraction.
            associations = {}
            for track_id, track in enumerate(extracted['tracks']):
                for t, region in track.items():
                    labels = truth['masks'][t]
                    overlap = [(int(k), int(np.logical_and(region['mask'], labels == k).sum())) for k in np.unique(labels) if k]
                    if overlap:
                        k, n = max(overlap, key=lambda z: z[1])
                        if n >= .5 * int((labels == k).sum()):
                            associations.setdefault(k, set()).add(track_id)
            splits = sum(max(0, len(v) - 1) for v in associations.values())
            nlinks = sum(len(v) for v in associations.values())
            row.update(reference_events=reference, predicted_events=extracted['events'], event_pairs=pairs,
                       true_events=len(reference), predicted_event_count=len(extracted['events']), matched_events=len(pairs),
                       track_count=extracted['track_count'], fragment_extra_tracks=splits,
                       truth_linked_tracks=nlinks, track_fragmentation_rate=splits / nlinks if nlinks else None,
                       camera_fit_coverage=extracted['camera_fit_coverage'])
            row['matched_by_type'] = dict(Counter(reference[i]['type'] for i, _ in pairs))
            row['true_by_type'] = dict(Counter(e['type'] for e in reference))
        else:
            from scipy.optimize import linear_sum_assignment
            ntruth = npred = hit = 0
            for frame, labels in zip(frames, truth['masks']):
                detections = extractor.frame(frame)
                masks = [labels == k for k in np.unique(labels) if k and k != 255]
                cost = np.full((len(masks), len(detections)), 1e6)
                for i, a in enumerate(masks):
                    for j, d in enumerate(detections):
                        b = d['mask']; overlap = np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())
                        if overlap >= .5:
                            cost[i, j] = 1 - overlap
                hit += sum(cost[i, j] < 1e6 for i, j in zip(*linear_sum_assignment(cost)))
                ntruth += len(masks); npred += len(detections)
            row.update(annotated_object_frames=ntruth, predicted_object_frames=npred, matched_object_frames=int(hit),
                       recall=hit / ntruth if ntruth else None)
        write(dest, row); results.append(row)
        print(args.part, item['source_id'], {k: v for k, v in row.items() if k in ('true_events', 'predicted_event_count', 'matched_events', 'recall')}, flush=True)
    write(root / (args.part + '_extractor_audit.json'), {"stage": args.part, "results": results,
          "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
          "gpu": torch.cuda.get_device_name(), "candidate_scores_observed": 0})


def gate_summary(args):
    root = args.output; verify(root); no_scores(root)
    event = read(root / 'event_extractor_audit.json')['results']
    objects = read(root / 'object_extractor_audit.json')['results']
    result = {}
    for domain in ('rendered', 'public_video'):
        rows = [x for x in event if x['domain'] == domain]
        true = sum(x['true_events'] for x in rows); pred = sum(x['predicted_event_count'] for x in rows); hit = sum(x['matched_events'] for x in rows)
        result[domain + '/event'] = {"sources": len(rows), "true_events": true, "predicted_events": pred, "matched_events": hit,
            "recall": hit / true if true else None, "precision": hit / pred if pred else None,
            "required": domain == 'rendered', "passed": bool(true and pred and hit / true >= .8 and hit / pred >= .8)}
        turns = sum(r['true_by_type'].get('turn', 0) for r in rows)
        matched_turns = sum(r['matched_by_type'].get('turn', 0) for r in rows)
        result[domain + '/turn'] = {'sources': len(rows), 'true_events': turns, 'matched_events': matched_turns,
            'recall': matched_turns / turns if turns else None, 'required': domain == 'rendered',
            'passed': bool(turns and matched_turns / turns >= .8)}
        rows = [x for x in objects if x['domain'] == domain]
        true = sum(x['annotated_object_frames'] for x in rows); hit = sum(x['matched_object_frames'] for x in rows)
        result[domain + '/object'] = {"sources": len(rows), "annotated_object_frames": true, "matched_object_frames": hit,
            "recall": hit / true if true else None, "required": domain == 'rendered',
            "passed": bool(true and hit / true >= .8), "scope": "diagnostic_only" if domain == 'public_video' else 'required_gate'}
    from .attribution_metric import rule_checks
    rules = rule_checks(); write(root / 'sta_rule_checks.json', rules)
    result['sta/rules'] = {'sources': 0, 'cases': rules['cases'], 'matched_events': rules['passed'],
                           'recall': rules['passed'] / rules['cases'], 'required': True, 'passed': rules['status'] == 'PASSED'}
    passed = all(v['passed'] for v in result.values() if v['required'])
    write(root / 'extractor_gate.json', {"status": 'PASSED' if passed else 'NOT_PASSED', "gates": result,
          "scores_observed": 0, "downstream": 'ELIGIBLE' if passed else 'NOT_RUN_GATE_FAILED'})
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'truth_audit', 'extractor_audit', 'gate_summary'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/metric_v2_public'))
    p.add_argument('--part', choices=('event', 'object'))
    p.add_argument('--reuse-inputs', type=Path)
    p.add_argument('--diagnostics', type=Path, default=Path('outputs/ere_sta_v31_development/diagnostics.json'))
    args = p.parse_args()
    if args.stage == 'extractor_audit' and args.part is None:
        p.error('--part required')
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
