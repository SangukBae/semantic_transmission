import json

from semantic_transmission.artifacts import sha256
from semantic_transmission import metric_v3_statistics as stats
from semantic_transmission.metric_v3_formal import SIGNS


def test_development_freeze_and_complete_heldout_reporting(tmp_path, monkeypatch):
    """Exercise the full statistics path before waiting for the GPU corpus."""
    root = tmp_path / 'run'; root.mkdir()
    base = tmp_path / 'base'; base.mkdir()
    protocol = {'base_run': str(base), 'domains': {'rendered': 'primary'}, 'sta_simulator': 'test fixture'}
    (root / 'protocol.json').write_text(json.dumps(protocol))
    digest = sha256(root / 'protocol.json')
    monkeypatch.setattr(stats, 'verify', lambda _: (protocol, {}))
    cases, sta, truth = [], [], []
    events = [{'type': 'start', 'time_s': .5, 'object_id': 1, 'direction': 0, 'cell': [1, 1]},
              {'type': 'stop', 'time_s': 2., 'object_id': 1, 'direction': 0, 'cell': [1, 1]}]
    for split in ('development', 'heldout'):
        for index in range(2):
            sid = split + str(index)
            common = dict(source_id=sid, domain='rendered', split=split, source_pixel_sha256=sid)
            for variant in ('identity', 'restyle_texture'):
                cases.append({**common, 'case_id': sid + variant, 'variant': variant, 'kind': 'control',
                              'target': 'control', 'family': 'pixel' if variant == 'identity' else 'rerender', 'severity': 0.})
            for severity in (.125, .25, .5):
                cases.append({**common, 'case_id': sid + 'reverse' + str(severity), 'variant': 'reverse',
                              'kind': 'reverse', 'target': 'motion', 'family': 'pixel', 'severity': severity})
            for label in ('normal', 'edr', 'clr', 'gfr', 'ghr'):
                sta.append({**common, 'case_id': sid + label, 'kind': label, 'label': label,
                            'severity': .25 if label != 'normal' else 0., 'eligible': True})
    for row in cases:
        truth.append({**row, 'accepted': True, 'reference_events': events, 'reconstruction_events': events})
    (base / 'truth_audit.json').write_text(json.dumps({'results': truth}))
    for name, rows in (('cases.jsonl', cases), ('sta_cases.jsonl', sta)):
        (root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    def populate(split):
        for corpus, rows in (('ere', cases), ('sta', sta)):
            for row in rows:
                if row['split'] != split:
                    continue
                positive = row.get('target') == 'motion' or row.get('label') in ('edr', 'clr', 'gfr', 'ghr')
                scores = {n: sign * (.7 if positive else 0.) for n, sign in SIGNS.items()}
                scores.update({name: ([0., 1., 0.] if positive else [0., 0., 0.]) for name in stats.CURVES})
                scores['reference_events'] = events
                scores['sta'] = {n: float(row.get('label') == n) for n in ('edr', 'clr', 'gfr', 'ghr')}
                for part in ('visual', 'pixel'):
                    directory = root / 'scores' / part / corpus; directory.mkdir(parents=True, exist_ok=True)
                    (directory / (row['case_id'] + '.json')).write_text(json.dumps({**row, 'scores': scores, 'protocol_sha256': digest}))
    populate('development')
    stats.calibrate(root)
    calibration_digest = sha256(root / 'calibration.json')
    assert (root / 'calibration.sha256').read_text().strip() == calibration_digest
    populate('heldout')
    stats.summarize(root)
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'COMPLETED'
    assert summary['ere_adoption_gate'] is True
    assert summary['ere_superiority_all_comparators'] is False
    assert summary['sta']['macro_accuracy'] == 1.
    assert summary['sta']['normal_false_alarm_rate'] == 0.
    assert summary['heldout_source_extractor_audit']['rendered']['matched'] == 4
    assert sha256(root / 'calibration.json') == calibration_digest
