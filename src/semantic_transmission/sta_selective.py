"""Development-only five-class STA classifier with explicit abstention.

Unknown is an operational status, not a semantic error class. Rejections count
as wrong in unconditional accuracy and are always reported with coverage.
"""
import numpy as np

CLASSES = ('normal', 'edr', 'clr', 'gfr', 'ghr')


def fit(rows, names):
    x = np.asarray([[r['scores'][n] for n in names] for r in rows], float)
    if not np.isfinite(x).all():
        raise ValueError('filter and report unmeasurable development rows before fit')
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.
    z = (x - mean) / scale
    labels = [r['label'] for r in rows]
    centers = []
    for label in CLASSES:
        # Equal weight per source within a class; no label counts at prediction.
        by_source = [z[[i for i,r in enumerate(rows) if r['label'] == label and r['source_id'] == source]].mean(0)
                     for source in sorted({r['source_id'] for r in rows if r['label'] == label})]
        if not by_source:
            raise ValueError('missing development class ' + label)
        centers.append(np.mean(by_source, axis=0))
    return {'names': list(names), 'mean': mean.tolist(), 'scale': scale.tolist(), 'centers': np.asarray(centers).tolist(),
            'training_sources': sorted({r['source_id'] for r in rows})}


def distances(scores, fitted):
    values = [scores.get(n) for n in fitted['names']]
    if any(v is None or not np.isfinite(v) for v in values):
        return None
    z = (np.asarray(values) - fitted['mean']) / fitted['scale']
    return np.mean((np.asarray(fitted['centers']) - z) ** 2, axis=1)


def calibrate(rows, fitted):
    # Separate development calibration sources; heldout is never read here.
    radii, margins = {}, []
    for label in CLASSES:
        values = []
        for row in rows:
            d = distances(row['scores'], fitted)
            if d is None or row['label'] != label:
                continue
            values.append(float(d[CLASSES.index(label)]))
            ordered = np.sort(d)
            if CLASSES[int(np.argmin(d))] == label:
                margins.append(float(ordered[1] - ordered[0]))
        radii[label] = float(np.quantile(values, .95, method='higher')) if values else 0.
    return {'radii': radii, 'minimum_margin': float(np.quantile(margins, .05, method='lower')) if margins else 0.,
            'calibration_sources': sorted({r['source_id'] for r in rows})}


def predict(scores, fitted, calibration, observable=True):
    d = distances(scores, fitted) if observable else None
    if d is None:
        return {'prediction': 'abstain', 'raw_prediction': None, 'reason': 'unobservable'}
    order = np.argsort(d); index = int(order[0]); label = CLASSES[index]
    margin = float(d[order[1]] - d[index])
    reason = ('outside_development_radius' if d[index] > calibration['radii'][label] else
              'ambiguous_classes' if margin < calibration['minimum_margin'] else None)
    return {'prediction': 'abstain' if reason else label, 'raw_prediction': label, 'reason': reason,
            'distance': float(d[index]), 'margin': margin}
