"""Bind score-independent visual notes to exact pairs; keep null labels null."""
import argparse
from pathlib import Path

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_campaign import read, save, check_files, file_inventory
from semantic_transmission.metric_campaign_real import attach_truth
from semantic_transmission.metric_campaign_worker import scored, generated_report
from semantic_transmission.metric_revision_audit import _write_once
from semantic_transmission.metric_revision import PRIMARY


def validate_notes(notes, index):
    if notes.get('schema') != 'metric-revision-ai-visual-notes-v1':
        raise ValueError('Unsupported visual note schema')
    by_id = {r['review_id']: r for r in notes['rows']}
    if len(by_id) != len(notes['rows']) or set(by_id) != {r['review_id'] for r in index['rows']}:
        raise ValueError('Missing, unknown or duplicate reviewed cases')
    for row in notes['rows']:
        if row['uep'] is not None and (type(row['uep']) not in (int, float) or row['uep'] != 0):
            raise ValueError('This coarse review can certify only explicit negatives; positive occupancy needs framewise evidence')
        if row['uep'] is not None and row['uncertain_frames']:
            raise ValueError('Uncertain cases cannot have a finite whole-scene label')
        if not row.get('observations') or not row.get('fso_eoi_unavailable_reason'):
            raise ValueError('Every judgement and abstention needs a reason')
        for key in ('source_visible', 'reconstruction_visible', 'uncertain_frames'):
            for start, end in row[key]:
                if not (type(start) is int and type(end) is int and 0 <= start < end <= 32):
                    raise ValueError('Invalid sampled-frame interval')
    return by_id


def assess(base, out, notes_path):
    index_path = out / 'review/index.json'
    notes, index = read(notes_path), read(index_path)
    by_id = validate_notes(notes, index)
    labels = []
    for item in index['rows']:
        note = by_id[item['review_id']]
        check_files(item['evidence_files'])
        evidence = dict(item['evidence_files'])
        evidence.update(file_inventory([notes_path, index_path, out/'review/received_slots.png']))
        labels.append(dict(case_id=item['case_id'], source_sha256=item['source_sha256'],
                           reconstruction_sha256=item['reconstruction_sha256'], alignment_sha256=item['alignment_sha256'],
                           evidence_files=evidence,
                           truth={k: note['uep'] if k == 'uep' else None for k in PRIMARY},
                           provenance='AI visual review of raw frame evidence; v5 aggregate outcomes previously known; '
                                      'not independently human verified; recognizable-entity semantic labels, not exhaustive SAM regions',
                           uep_evidence_scope='received_keyframes_only', visual_note=note))
    path = out / 'independent_truth.json'
    _write_once(path, dict(schema='metric-campaign-independent-truth-v1', rows=labels,
                          review_method=notes, independence_scope='no SAM2/DINO/RAFT/candidate score used to generate labels',
                          certified_human_ground_truth=False))
    old_rows = attach_truth(scored(base, '03_natural', real=True), path)
    config = read(base/'campaign.json')['config']
    thresholds = read(base/'01_cross/calibration.json')['thresholds']
    prior = generated_report(old_rows, thresholds, config)
    revised_by_id = {r['case_id']: r for r in read(out/'04_existing_reconstructions/report.json')['rows']}
    revised_rows = [{**r, 'scores': revised_by_id[r['case_id']]['scores']} for r in old_rows]
    revised = generated_report(revised_rows, {k: 0. for k in PRIMARY}, config)
    for report in (prior, revised):
        report['review_status'] = 'AI_VISUAL_REVIEW_PARTIAL_TRUTH'
        report['scientific_status'] = 'DEVELOPMENT_ONLY'
        report['limitations'] = ['No human verification or new independent dataset',
                                'Only four UEP-negative clips from two sources have numeric labels',
                                'No positive labels; AUC and improvement ranking cannot be established',
                                'Semantic entities differ from arbitrary segmentation regions']
    _write_once(out/'independent_assessments/v5.json', prior)
    _write_once(out/'independent_assessments/v6.json', revised)
    _write_once(out/'independent_assessments/receipt.json', dict(schema='metric-revision-review-receipt-v1',
        files=file_inventory([Path(__file__).resolve(), notes_path, index_path, path,
                              out/'04_existing_reconstructions/report.json',
                              out/'independent_assessments/v5.json', out/'independent_assessments/v6.json']),
        reviewed_pairs=len(labels), numeric_uep_labels=sum(r['truth']['uep'] is not None for r in labels)))
    print('Reviewed 12 pairs; numeric UEP negatives: 4; positive examples: 0; model ranking unavailable')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    args = parser.parse_args()
    assess(args.base.resolve(), args.output.resolve(), args.notes.resolve())
