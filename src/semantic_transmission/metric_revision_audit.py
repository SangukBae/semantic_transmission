"""Read-only reuse of v5 inputs; all v6 development evidence has a new root."""
import argparse
from collections import OrderedDict
import gzip
from pathlib import Path
import pickle
import time

import cv2
import numpy as np
from PIL import Image

from . import metric_revision as metric
from .artifacts import sha256
from .metric_campaign import REPO, read, save, digest, check_files, file_inventory, campaign_lock
from .metric_campaign_real import real_inputs
from .metric_campaign_scoring import detection, finite
from .metric_v3_features import pixels
from .metric_v3_cases import public_truth
from .metric_v2_cases import temporal_variant


class Cache:
    def __init__(self, base, out):
        declaration = read(base / 'campaign.json')
        signature = digest([declaration['files'], sha256(base / '01_cross/calibration.json')])
        self.root = base / 'visual_cache' / signature
        self.out, self.recent = out, OrderedDict()
        self.receipt = out / 'cache_inputs.json'
        self.files = read(self.receipt) if self.receipt.exists() else {}

    def __call__(self, key):
        if key in self.recent:
            self.recent.move_to_end(key)
            return self.recent[key]
        path = self.root / (key + '.pkl.gz')
        actual = sha256(path)
        if str(path) in self.files and self.files[str(path)] != actual:
            raise ValueError('Original observation cache changed')
        self.files[str(path)] = actual
        with gzip.open(path, 'rb') as stream:
            raw = pickle.load(stream)  # trusted local experiment cache only
        if raw['pixel_sha256'] != key:
            raise ValueError('Wrong cached pixel identity')
        value = metric.observe(raw)
        self.recent[key] = value
        while len(self.recent) > 4:
            self.recent.popitem(last=False)
        return value

    def checkpoint(self):
        save(self.receipt, self.files)


def freeze(base, out):
    if base == out or base in out.parents:
        raise ValueError('Use a separate output directory outside the frozen campaign')
    files = dict(read(base / 'campaign.json')['files'])
    for receipt in base.glob('*/receipts/*.json'):
        files.update(read(receipt)['artifacts'])
        files[str(receipt)] = sha256(receipt)
    files.update(file_inventory([base / 'campaign.json', base / 'summary.json']))
    check_files(files)
    source_paths = [Path(__file__), Path(metric.__file__), REPO / 'docs/METRIC_REVISION_PROTOCOL.md',
                    REPO / 'scripts/run_metric_revision.sh', REPO / 'scripts/prepare_metric_revision_review.py']
    declaration = dict(schema='metric-revision-v6-development', base=str(base), parameters=metric.PARAMETERS,
                       code=file_inventory(source_paths), original_files=files,
                       scientific_status='DEVELOPMENT_ONLY', calibrated=False,
                       prior_campaign_results_already_observed=True)
    path = out / 'declaration.json'
    if path.exists() and read(path) != declaration:
        raise ValueError('Frozen v6 inputs/code changed; use a new revision directory')
    if not path.exists():
        save(path, declaration)
    return digest(declaration)


def natural_annotations(base):
    result = {}
    for item in read(base / 'campaign.json')['natural_inventory']:
        masks = []
        for path in item['labels']:
            with Image.open(path) as image:
                masks.append(cv2.resize(np.asarray(image), (320, 192), interpolation=cv2.INTER_NEAREST))
        masks = np.stack(masks)
        relabelled = np.zeros_like(masks)
        for j, label in enumerate([x for x in np.unique(masks) if 0 < x < 255], 1):
            relabelled[masks == label] = j
        gt = public_truth(relabelled)
        result[item['source_id']] = (gt, relabelled)
    return result


def natural_truth(row, gt):
    if row['kind'] == 'addition':
        return {'uep': row['truth']['uep']}, [], None
    source = metric.annotation_observation(gt['centers'], gt['visible'])
    changed = {k: v.copy() for k, v in gt.items()}
    if row['kind'] != 'control':
        severity = float(row['case_id'].split('__')[-1].split('_')[1])
        _, meta = temporal_variant(np.zeros((len(gt['centers']), 2, 2, 3), np.uint8), row['kind'], severity)
        changed = {k: v[meta['source_index_map']] for k, v in gt.items()}
    if row['style'] == 'camera':
        for t in range(len(changed['centers'])):
            matrix = cv2.getRotationMatrix2D((160, 96), 1.5*np.sin(t*.4), 1.)
            matrix[:, 2] += [2*np.sin(t*.8), 2*np.cos(t*.6)]
            changed['centers'][t] = changed['centers'][t] @ matrix[:, :2].T + matrix[:, 2]
    target = metric.annotation_observation(changed['centers'], changed['visible'])
    truth, records = metric.annotation_truth(source, target)
    truth['uep'] = row['truth'].get('uep')
    return truth, records, source


def query_audit(observed, labels, truth_records, result):
    """Post-score audit only: associate predicted queries to labelled objects.

    Annotation masks never alter candidate extraction, events or score values.
    Unmatched truth queries remain in the coverage denominator.
    """
    identity = {}
    for i, track in enumerate(observed['tracks']):
        candidates = []
        for j in range(1, int(labels.max()) + 1):
            overlaps = []
            for t, d in track.items():
                mask = labels[t] == j
                union = np.logical_or(mask, d['mask']).sum()
                if mask.any():
                    overlaps.append(np.logical_and(mask, d['mask']).sum() / max(1, union))
            candidates.append(np.mean(overlaps) if overlaps else 0.)
        if candidates and max(candidates) >= .5:
            identity[i] = int(np.argmax(candidates))
    used, output = set(), []
    for truth in truth_records:
        candidates = [(abs(r['anchor'] - truth['anchor']), k, r) for k, r in enumerate(result['fso_records'])
                      if k not in used and r['metric'] == truth['metric'] and
                      identity.get(r['source_track']) == truth['source_track'] and abs(r['anchor'] - truth['anchor']) <= 2]
        candidates.sort(key=lambda x: (x[0], x[1]))
        chosen = candidates[0] if candidates else None
        if chosen:
            used.add(chosen[1])
        output.append(dict(metric=truth['metric'], object_id=truth['source_track']+1,
                           truth_anchor=truth['anchor'], truth=truth['score'],
                           score=chosen[2]['score'] if chosen else None,
                           predicted_anchor=chosen[2]['anchor'] if chosen else None,
                           query_found=bool(chosen)))
    return output


def _write_once(path, record):
    if path.exists():
        if read(path) != record:
            raise ValueError('Refusing to overwrite changed revision artifact: ' + str(path))
    else:
        save(path, record)


def process(base, out, phase, cache, signature, annotations):
    real = phase == '04_existing_reconstructions'
    input_path = base / '03_natural/real_pairs.json' if real else base / phase / 'cases.json'
    rows = read(input_path)['rows']
    results = []
    for n, row in enumerate(rows, 1):
        path = out / phase / 'rows' / (row['case_id'] + '.json')
        input_signature = digest([signature, row])
        if path.exists():
            record = read(path)
            if record['input_signature'] != input_signature or record['result_sha256'] != digest(record['result']):
                raise ValueError('Corrupt resumed revision row')
        else:
            if real:
                aa, bb, rr, status, alignment = real_inputs(row)
                keys = [pixels(v) for v in (aa, bb, rr)]
            else:
                keys = [row[k]['pixel_sha256'] for k in ('source', 'reconstruction', 'rx')]
                status, alignment = row['rx_status'], None
            a, b, rx = [cache(key) for key in keys]
            values = metric.evaluate(a, b, rx, status)
            old_phase = '03_natural' if real else phase
            old_operation = 'real_visual' if real else 'visual'
            old = read(base / old_phase / 'scores' / old_operation / (row['case_id']+'.json'))['result']['scores']
            truth, truth_records, oracle_source = {}, [], None
            scope = 'same prior oracle; changed kinematics, diagnostic comparison only'
            if phase == '03_natural':
                gt, labels = annotations[row['source_id']]
                truth, truth_records, oracle_source = natural_truth(row, gt)
                scope = 'DAVIS selected objects; image-plane contract; query audit only for FSO/EOI'
            elif not real:
                truth = row['truth']
            query = query_audit(a, labels, truth_records, values) if truth_records else []
            result = dict(case_id=row['case_id'], source_id=row['source_id'], style=row['style'],
                          kind=row.get('kind'), steps=row.get('steps'), scores={k: values[k] for k in metric.PRIMARY},
                          previous_scores={k: old.get(k) for k in metric.PRIMARY}, diagnostics=values,
                          truth=truth, truth_scope=scope, alignment=alignment, query_audit=query,
                          source_tracks_before=a['original_tracks'], source_tracks_after=len(a['tracks']),
                          reconstruction_tracks_before=b['original_tracks'], reconstruction_tracks_after=len(b['tracks']))
            record = dict(input_signature=input_signature, result=result, result_sha256=digest(result))
            _write_once(path, record)
        results.append(record['result'])
        if n % 25 == 0 or n == len(rows):
            cache.checkpoint()
            print(phase, n, '/', len(rows), flush=True)
    config = read(base / 'campaign.json')['config']
    report = dict(execution_status='COMPLETED', scientific_status='DEVELOPMENT_ONLY', cases=len(results),
                  independent_sources=len({r['source_id'] for r in results}), calibrated=False,
                  parameters=metric.PARAMETERS, metrics={})
    for target in metric.PRIMARY:
        stats = detection(results, target, target, 0., config)
        stats['diagnostic_gate'] = stats.pop('status')
        stats['status'] = 'DEVELOPMENT_ONLY'
        if target != 'uep':
            stats['scope_warning'] = ('old oracle uses a different kinematic contract' if phase == '01_cross' else
                                      'selected-object truth is not a whole-scene truth' if phase == '03_natural' else
                                      'no independent scalar truth')
        report['metrics'][target] = stats
    if phase == '03_natural':
        query_rows = [dict(source_id=r['source_id'], truth={q['metric']: q['truth']}, scores={q['metric']: q['score']})
                      for r in results for q in r['query_audit']]
        report['selected_object_query_diagnostics'] = {k: detection(query_rows, k, k, 0., config)
                                                       for k in metric.PRIMARY if k.startswith('fso_')}
        for v in report['selected_object_query_diagnostics'].values():
            v['diagnostic_gate'] = v.pop('status')
            v['status'] = 'DEVELOPMENT_ONLY'
    controls = [r for r in results if r['truth'].get('uep') == 0]
    report['uep_known_negatives'] = {style: dict(cases=sum(r['style']==style for r in controls),
           previous_warnings=sum(r['style']==style and finite(r['previous_scores']['uep']) and r['previous_scores']['uep'] > 0 for r in controls),
           revised_warnings=sum(r['style']==style and finite(r['scores']['uep']) and r['scores']['uep'] > 0 for r in controls))
           for style in sorted({r['style'] for r in controls})}
    report['rows'] = results
    _write_once(out / phase / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=REPO/'outputs/metric_validation_20260916_v1')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/metric_revision_20260916_v6')
    args = parser.parse_args(argv)
    base, out = args.base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with campaign_lock(out):
        started = time.time()
        signature = freeze(base, out)
        cache, annotations = Cache(base, out), natural_annotations(base)
        reports = {}
        # Real outputs first: no new GPU inference or reconstruction is invoked.
        for phase in ('04_existing_reconstructions', '03_natural', '02_components', '01_cross'):
            reports[phase] = process(base, out, phase, cache, signature, annotations)
        cache.checkpoint()
        check_files(read(out/'declaration.json')['original_files'])
        save(out/'summary.json', dict(execution_status='COMPLETED', scientific_status='DEVELOPMENT_ONLY',
             seconds=time.time()-started, declaration_sha256=sha256(out/'declaration.json'),
             reports={k: sha256(out/k/'report.json') for k in reports},
             original_campaign_files_unchanged=True, generated_new_reconstructions=0))
        print('Completed:', out, flush=True)


if __name__ == '__main__':
    main()
