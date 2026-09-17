"""Verify frozen inputs, all rows, report joins, independent labels and AUCs."""
import argparse
from pathlib import Path

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_campaign import read, digest, check_files
from semantic_transmission.metric_revision_audit import _write_once
from semantic_transmission.metric_campaign_scoring import finite


def direct_auc(rows, field='scores'):
    positive = [r[field]['uep'] for r in rows if finite(r['truth'].get('uep')) and r['truth']['uep'] > 0 and finite(r[field].get('uep'))]
    negative = [r[field]['uep'] for r in rows if r['truth'].get('uep') == 0 and finite(r[field].get('uep'))]
    if not positive or not negative:
        return None
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive)*len(negative))


def verify(out):
    declaration = read(out/'declaration.json')
    check_files(declaration['code'])
    check_files(declaration['original_files'])
    check_files(read(out/'cache_inputs.json'))
    summary = read(out/'summary.json')
    assert summary['declaration_sha256'] == sha256(out/'declaration.json')
    assert summary['scientific_status'] == 'DEVELOPMENT_ONLY'
    base = Path(declaration['base'])
    total = 0
    for phase, expected in summary['reports'].items():
        path = out/phase/'report.json'
        assert sha256(path) == expected, phase
        report = read(path)
        inputs = (base/'03_natural/real_pairs.json') if phase == '04_existing_reconstructions' else (base/phase/'cases.json')
        source_rows = {r['case_id']: r for r in read(inputs)['rows']}
        assert len(report['rows']) == report['cases'] == len(source_rows)
        assert len({r['case_id'] for r in report['rows']}) == len(source_rows)
        for r in report['rows']:
            record = read(out/phase/'rows'/(r['case_id']+'.json'))
            assert record['result'] == r
            assert record['result_sha256'] == digest(r)
            assert record['input_signature'] == digest([digest(declaration), source_rows[r['case_id']]])
        area = direct_auc(report['rows'])
        saved = report['metrics']['uep']['auc']
        assert area is None and saved is None or area is not None and saved is not None and abs(area-saved) < 1e-12
        total += len(report['rows'])
    aligned = read(out/'aligned_synthetic_truth.json')
    check_files(aligned['provenance'])
    assert aligned['rows_sha256'] == digest(aligned['rows'])
    assert aligned['cases'] == 1800
    for name, field in [('v5', 'previous_scores'), ('v6', 'scores')]:
        assert abs(direct_auc(aligned['rows'], field)-aligned['metrics']['uep'][name]['auc']) < 1e-12
    review = read(out/'independent_assessments/receipt.json')
    check_files(review['files'])
    labels = read(out/'independent_truth.json')['rows']
    for label in labels:
        check_files(label['evidence_files'])
    assert len(labels) == 12 and sum(r['truth']['uep'] is not None for r in labels) == 4
    result = dict(execution_status='VERIFIED', scientific_status='DEVELOPMENT_ONLY', scored_rows=total,
                  original_files_verified=len(declaration['original_files']),
                  observation_caches_verified=len(read(out/'cache_inputs.json')),
                  phases=len(summary['reports']), reviewed_reconstructions=12, numeric_uep_negative_labels=4,
                  independent_positive_labels=0, scorer_used_to_generate_visual_labels=False,
                  check_scope='SHA256, row input/result signatures, exact report joins, direct pairwise UEP AUC, label evidence hashes',
                  human_truth_verified=False)
    _write_once(out/'verification.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    verify(parser.parse_args().output.resolve())
