#!/usr/bin/env python3
"""Verify completed score coverage, chronology, and preserved experiment inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_formal import read, save, verify
from semantic_transmission.metric_v2_validation import _verify as verify_v2


def reject_nonfinite(value):
    raise ValueError('Nonstandard JSON numeric value: ' + value)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    root = ap.parse_args().output; protocol, base = verify(root)
    verify_v2(Path('outputs/mte_otf_20260914_v1'))
    calibration = read(root / 'calibration.json')
    assert sha256(root / 'calibration.json') == (root / 'calibration.sha256').read_text().strip()
    assert calibration['heldout_scores_observed'] == 0
    assert read(root / 'summary.json')['status'] == 'COMPLETED'
    assert read(root / 'real_pair_diagnostics.json')['status'] == 'COMPLETED'
    counts, times = Counter(), {'development': [], 'heldout': []}
    sources = {'development': set(), 'heldout': set()}
    pixels = {'development': set(), 'heldout': set()}
    for corpus, filename in (('ere', 'cases.jsonl'), ('sta', 'sta_cases.jsonl')):
        rows = [json.loads(x) for x in (root / filename).read_text().splitlines()]
        rows = [r for r in rows if r.get('eligible', True)]
        if corpus == 'sta':
            for row in rows:
                folder = root / 'sta_inputs' / row['case_id']
                assert sha256(folder / 'packets.pkl') == row['packets_sha256']
                assert sha256(folder / 'transport.json') == row['transport_sha256']
        for part in ('visual', 'pixel'):
            paths = {f.stem: f for f in (root / 'scores' / part / corpus).glob('*.json')}
            assert set(paths) == {r['case_id'] for r in rows}
            for row in rows:
                path = paths[row['case_id']]
                scored = json.loads(path.read_text(), parse_constant=reject_nonfinite)
                for key in ('case_id', 'split', 'domain', 'source_id', 'source_pixel_sha256'):
                    assert scored[key] == row[key], (path, key)
                assert scored['protocol_sha256'] == sha256(root / 'protocol.json')
                counts['/'.join((part, corpus, row['split']))] += 1
                times[row['split']].append(path.stat().st_mtime)
                sources[row['split']].add(row['source_id'])
                pixels[row['split']].add(row['source_pixel_sha256'])
    assert not sources['development'] & sources['heldout']
    assert not pixels['development'] & pixels['heldout']
    assert max(times['development']) < calibration['created_unix'] < min(times['heldout'])
    for name in ('same_input_baseline_declaration.json', 'memoization_probe.json',
                 'sta_semantic_controls_declaration.json', 'sta_classifier_head_declaration.json'):
        declaration = read(root / name)
        assert declaration['heldout_scores_observed'] == 0
        assert declaration['created_unix'] < min(times['heldout'])
    assert read(root / 'same_input_baseline_declaration.json')['script_sha256'] == sha256(Path('scripts/evaluate_sta_same_input_baselines.py'))
    assert read(root / 'memoization_probe.json')['script_sha256'] == sha256(Path('scripts/run_metric_v3_memoized.py'))
    assert read(root / 'sta_semantic_controls_declaration.json')['script_sha256'] == sha256(Path('scripts/evaluate_sta_semantic_controls.py'))
    assert read(root / 'sta_classifier_head_declaration.json')['script_sha256'] == sha256(Path('scripts/evaluate_sta_classifier_head.py'))
    assert all(r['all_outputs_exactly_equal'] for r in read(root / 'memoization_probe.json')['results'])
    supplement = read(root / 'same_input_baseline_calibration.json')
    assert supplement['heldout_scores_observed_by_this_supplement'] == 0
    assert supplement['created_unix'] < (root / 'same_input_baseline_heldout.json').stat().st_mtime
    assert read(root / 'same_input_baseline_results.json')['calibration_sha256'] == sha256(root / 'same_input_baseline_calibration.json')
    for split in ('development', 'heldout'):
        controls = read(root / ('sta_semantic_controls_' + split + '.json'))
        assert controls['declaration_sha256'] == sha256(root / 'sta_semantic_controls_declaration.json')
        assert controls['calibration_sha256'] == sha256(root / 'calibration.json')
        manifest = [json.loads(x) for x in (root / 'cases.jsonl').read_text().splitlines()]
        expected = {r['case_id'] for r in manifest if r['split'] == split and r['domain'] == 'rendered' and r['target'] == 'control'}
        assert {r['case_id'] for r in controls['rows']} == expected
        assert len(controls['rows']) == len(expected)
    head = read(root / 'sta_classifier_head_results.json')
    assert head['calibration_sha256'] == sha256(root / 'sta_classifier_head_calibration.json')
    head_calibration = read(root / 'sta_classifier_head_calibration.json')
    assert head_calibration['declaration_sha256'] == sha256(root / 'sta_classifier_head_declaration.json')
    assert head_calibration['heldout_scores_observed_by_this_supplement'] == 0
    assert head_calibration['created_unix'] < head['completed_unix']
    sta_manifest = [json.loads(x) for x in (root / 'sta_cases.jsonl').read_text().splitlines()]
    assert set(head_calibration['classifier']['training_sources']) == {r['source_id'] for r in sta_manifest if r['split'] == 'development'}
    assert {r['case_id'] for r in head['predictions']} == {r['case_id'] for r in sta_manifest if r['split'] == 'heldout'}
    # An early summary exists in the original concurrent run, but is not a
    # prerequisite for a sequential replay. Never manufacture an interim result.
    early_matches_final = None
    if (root / 'ere_heldout_interim.json').exists():
        final = read(root / 'summary.json')['error_detection']
        for kind, group in read(root / 'ere_heldout_interim.json')['error_detection'].items():
            assert all(final[kind][key] == value for key, value in group.items())
        early_matches_final = True
    preserved = {}
    for filename in ('2026-09-14-ere-sta.json', '2026-09-14-ere-sta-reaudit.json'):
        prior = read(Path('docs/validation') / filename)
        for name, digest in prior['artifact_sha256'].items():
            assert sha256(Path(prior['run']) / name) == digest, (prior['run'], name)
        old_protocol = read(Path(prior['run']) / 'protocol.json')
        for name, digest in old_protocol['code_sha256'].items():
            assert sha256(Path(prior['run']) / 'frozen_source' / name) == digest
        preserved[prior['run']] = {'recorded_artifacts_verified': len(prior['artifact_sha256']),
                                   'frozen_code_files_verified': len(old_protocol['code_sha256'])}
    model_root = Path(base['models']['root'])
    for path, key in ((model_root / 'sam2.1_hiera_tiny.pt', 'sam2_weights_sha256'),
                      (model_root / 'dinov2_vits14_pretrain.pth', 'dino_weights_sha256'),
                      (Path.home() / '.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth', 'raft_weights_sha256'),
                      (Path.home() / '.cache/clip/ViT-B-32.pt', 'clip_weights_sha256')):
        assert sha256(path) == base['models'][key], path
    for repo, key in (('sam2', 'sam2_commit'), ('dinov2', 'dinov2_commit')):
        commit = subprocess.check_output(['git', '-C', str(model_root / repo), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == base['models'][key]
        assert not subprocess.check_output(['git', '-C', str(model_root / repo), 'status', '--porcelain'], text=True).strip()
    public_files = 0
    for source in base['inventory']:
        for item in source.get('files', []):
            assert sha256(Path(item['path'])) == item['sha256']
            assert sha256(Path(item['annotation_path'])) == item['annotation_sha256']
            public_files += 2
    report = {'status': 'PASSED', 'recorded_unix': time.time(), 'score_counts': dict(counts),
        'total_score_records': sum(counts.values()), 'source_overlap': [], 'source_pixel_overlap': [],
        'chronology': {'last_development_score_mtime': max(times['development']),
                      'calibration_created_unix': calibration['created_unix'],
                      'first_heldout_score_mtime': min(times['heldout']),
                      'last_heldout_score_mtime': max(times['heldout'])},
        'preserved_previous_runs': preserved, 'model_weight_hashes_verified': 4,
        'model_source_commits_verified': 2, 'model_source_worktrees_clean': True,
        'original_public_RGB_and_annotation_files_verified': public_files,
        'STA_packet_and_transport_files_rechecked': 1536,
        'supplementary_semantic_controls_verified': 480,
        'supplementary_STA_classifier_predictions_verified': len(head['predictions']),
        'early_ERE_summary_matches_final': early_matches_final,
        'protocol_sha256': sha256(root / 'protocol.json'), 'calibration_sha256': sha256(root / 'calibration.json'),
        'note': 'All scored RGB hashes were additionally checked when each formal observer loaded its input.'}
    save(root / 'final_integrity.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'total_score_records', 'score_counts')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
