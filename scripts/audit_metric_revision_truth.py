"""Compare v5/v6 on a common, explicitly image-plane synthetic oracle.

This is a supplementary development audit after the v6 run. It does not
overwrite either run and does not reinterpret the old world-state protocol.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from semantic_transmission import metric_revision as metric
from semantic_transmission.metric_campaign import read, digest, file_inventory
from semantic_transmission.metric_campaign_scoring import detection
from semantic_transmission.metric_revision_audit import _write_once
from semantic_transmission.metric_v5_cases import scene, trajectory, SWAP


def aligned_truth(row):
    if row['kind'] == 'addition':
        return {'uep': row['truth']['uep']}
    family, seed = row['source_id'].split('/')
    state = scene(int(seed), family)
    ac, av = trajectory(state)
    name = row['case_id'].rsplit('_'+row['style'], 1)[0].split('__')[-1]
    if row['kind'] == 'control':
        bc, bv = ac.copy(), av.copy()
    elif row['kind'] == 'order_swap':
        times = dict(state['times'])
        x, y = SWAP[family]
        times[x], times[y] = times[y], times[x]
        bc, bv = trajectory(state, times)
    else:
        bc, bv = trajectory(state, override={'kind': row['kind'], 'delay': int(name.rsplit('_', 1)[1])})
    if row['style'] == 'camera':
        for t in range(len(bc)):
            matrix = cv2.getRotationMatrix2D((160, 96), 1.5*np.sin(t*.4), 1.)
            matrix[:, 2] += [2*np.sin(t*.8), 2*np.cos(t*.6)]
            bc[t] = bc[t] @ matrix[:, :2].T + matrix[:, 2]
    a, b = metric.annotation_observation(ac, av), metric.annotation_observation(bc, bv)
    truth, _ = metric.annotation_truth(a, b)
    truth['uep'] = row['truth'].get('uep')
    return truth


def audit(base, out):
    manifest = base/'01_cross/cases.json'
    report_path = out/'01_cross/report.json'
    revised = {r['case_id']: r for r in read(report_path)['rows']}
    results = []
    for row in read(manifest)['rows']:
        scores = revised[row['case_id']]
        results.append(dict(case_id=row['case_id'], source_id=row['source_id'], style=row['style'],
                            truth=aligned_truth(row), scores=scores['scores'], previous_scores=scores['previous_scores']))
    config = read(base/'campaign.json')['config']
    metrics = {}
    for key in metric.PRIMARY:
        old = [{**r, 'scores': r['previous_scores']} for r in results]
        metrics[key] = {'v5': detection(old, key, key, 0., config), 'v6': detection(results, key, key, 0., config)}
        for value in metrics[key].values():
            value['diagnostic_gate'] = value.pop('status')
            value['status'] = 'DEVELOPMENT_ONLY'
    _write_once(out/'aligned_synthetic_truth.json', dict(schema='metric-revision-aligned-truth-v1',
        scientific_status='DEVELOPMENT_ONLY', score_threshold=0., calibrated=False,
        scope='same explicit image-plane truth for both scorers; not world-coordinate appearance invariance',
        provenance=file_inventory([Path(__file__).resolve(), Path(metric.__file__).resolve(), manifest, report_path]),
        cases=len(results), metrics=metrics, rows=results, rows_sha256=digest(results)))
    print('Aligned independent renderer trajectories:', len(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.base.resolve(), args.output.resolve())
