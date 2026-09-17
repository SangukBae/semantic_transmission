import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_transmission import metric_campaign as c
from semantic_transmission import metric_campaign_cases as cases
from semantic_transmission.metric_campaign_real import attach_truth, export_source, load_array, reconstruct
from semantic_transmission.metric_campaign_scoring import detection, visual_scores
from semantic_transmission.metric_campaign_worker import generated_report, model_change_agreement, rx_counterfactual


def config():
    cfg = c.read(c.REPO / 'configs/metric_validation.json')
    return {**cfg, 'synthetic_sources_per_family': 1, 'rx_sources_per_family': 1,
            'bootstrap_iterations': 20, 'minimum_positive_sources': 2}


@pytest.fixture
def local_campaign_repo(tmp_path, monkeypatch):
    """Orchestration fixtures must not read a developer's ignored installation."""
    import shutil
    import sys
    from semantic_transmission import metric_campaign_real as real
    repo = tmp_path / 'fixture_repo'
    shutil.copytree(c.REPO / 'configs', repo / 'configs')
    c.save(repo / '.local/settings.json', {'python': sys.executable,
                                         'channel_python': sys.executable})
    c.save(repo / '.local/model_paths.json', {'fixture_model': str(repo / 'models')})
    monkeypatch.setattr(c, 'REPO', repo)
    monkeypatch.setattr(real, 'REPO', repo)
    monkeypatch.setattr(real, 'git_state', lambda root: {'commit': 'test-fixture', 'dirty': False})
    return repo


def test_worker_invokes_virtualenv_path_without_dereferencing_python(tmp_path, local_campaign_repo):
    import os
    import subprocess
    import sys
    base = subprocess.check_output([sys.executable, '-c', 'import sys; print(sys.base_prefix)'], text=True).strip()
    venv = tmp_path / 'evaluation'
    (venv / 'bin').mkdir(parents=True)
    binary = venv / 'bin/python'
    binary.symlink_to(Path(sys.executable).resolve())
    (venv / 'pyvenv.cfg').write_text(f'home = {base}/bin\ninclude-system-site-packages = false\n')
    cfg = {**config(), 'evaluation_python': str(binary)}
    for operation in ('calibrate', 'visual', 'real_visual'):
        command = c.worker_command(cfg, tmp_path, '01_cross', operation)
        assert command[0] == str(binary)
        assert Path(command[0]).resolve() != Path(command[0])
    env = {**os.environ, 'PYTHONNOUSERSITE': '1'}
    prefix = subprocess.check_output([command[0], '-c', 'import sys; print(sys.prefix)'], env=env, text=True).strip()
    assert prefix == str(venv)


def test_preflight_detects_missing_sam2_with_the_worker_environment(local_campaign_repo):
    cfg = config()
    def missing(command, **kwargs):
        assert command[0] == c.worker_command(cfg, c.REPO, '01_cross', 'visual')[0]
        assert kwargs['env'] == c.worker_env(cfg)
        return SimpleNamespace(returncode=1, stdout='', stderr="ModuleNotFoundError: No module named 'sam2'")
    with pytest.raises(RuntimeError, match='sam2'):
        c.probe_evaluation_python(cfg, runner=missing)


def fake_runner(root, called, fail=None):
    def run(command, **kwargs):
        operation, phase = command[3], command[-1]
        called.append((phase, operation))
        if (phase, operation) == fail:
            return SimpleNamespace(returncode=3)
        artifact = root / phase / (operation + '.json')
        c.save(artifact, {'status': 'NOT_PASSED'})
        c.save(c.receipt(root, phase, operation), {'execution_status': 'COMPLETED',
                                                  'artifacts': c.file_inventory([artifact])})
        return SimpleNamespace(returncode=0)
    return run


def test_sequence_resumes_and_scientific_failure_does_not_stop(tmp_path, local_campaign_repo):
    called = []
    c.execute(tmp_path, config(), runner=fake_runner(tmp_path, called))
    assert called == c.tasks(config())
    assert c.read(tmp_path / 'summary.json')['scientific_status'] == 'REVIEW_REQUIRED'
    called.clear()
    c.execute(tmp_path, config(), runner=fake_runner(tmp_path, called))
    assert called == []
    assert c.read(tmp_path / 'state.json')['execution_status'] == 'COMPLETED'


def test_failure_stops_before_following_stage_and_resume_detects_tampering(tmp_path, local_campaign_repo):
    called = []
    with pytest.raises(RuntimeError, match='exited 3'):
        c.execute(tmp_path, config(), runner=fake_runner(tmp_path, called, ('01_cross', 'visual')))
    assert called == c.tasks(config())[:3]
    assert c.read(tmp_path / 'state.json')['execution_status'] == 'FAILED'
    (tmp_path / '01_cross/prepare.json').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        c.execute(tmp_path, config(), runner=fake_runner(tmp_path, []))


def test_empty_receipt_and_duplicate_process_cannot_count_as_success(tmp_path):
    path = tmp_path / 'receipt.json'
    c.save(path, {'execution_status': 'COMPLETED', 'artifacts': {}})
    with pytest.raises(ValueError, match='Incomplete'):
        c.verify_receipt(path)
    with c.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another process'):
            with c.campaign_lock(tmp_path):
                pass


def test_error_crosses_every_style_and_truth_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, 'FAMILIES', cases.FAMILIES[:1])
    cfg = config()
    cases.cross_corpus(tmp_path, cfg)
    rows = c.read(tmp_path / '01_cross/cases.json')['rows']
    assert len(rows) == 15 * 5
    groups = {}
    for row in rows:
        groups.setdefault(row['case_id'].rsplit('_' + row['style'], 1)[0], []).append(row)
    for group in groups.values():
        assert {r['style'] for r in group} == set(cfg['styles'])
        assert len({c.digest(r['truth']) for r in group}) == 1
    for target in cases.PRIMARY:
        assert any((r['truth'].get(target) or 0) > 0 for r in rows)
        assert any(r['truth'].get(target) == 0 for r in rows)
    nonidentity = next(r for r in rows if r['style'] == 'palette')
    assert nonidentity['source']['pixel_sha256'] != nonidentity['reconstruction']['pixel_sha256']
    descriptor = rows[0]['source']
    Path(descriptor['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        load_array(descriptor)


def test_receiver_only_changes_truth_with_fixed_source_and_reconstruction(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, 'FAMILIES', cases.FAMILIES[:1])
    cases.rx_corpus(tmp_path, config())
    rows = c.read(tmp_path / '02_components/cases.json')['rows']
    assert len(rows) == 15
    for length in (2, 4, 8):
        group = [r for r in rows if r['counterfactual_group'].endswith('/' + str(length))]
        assert len({c.digest([r['source'], r['reconstruction']]) for r in group}) == 1
        truth = {r['kind']: r['truth']['uep'] for r in group}
        assert truth['supported'] == 0
        assert truth['absent'] == truth['corrupt'] == min(1., length / 4)
        assert truth['sparse'] > 0
        for row in group:
            row['scores'] = {'uep': row['truth']['uep'], 'uep_source_only': min(1., length / 4),
                             'uep_rx_only': row['truth']['uep']}
    report = rx_counterfactual(rows, config())
    assert report['uep']['source_mean_direction_agreement'] == 1
    assert report['uep_source_only']['source_mean_direction_agreement'] == 0


def test_unlabelled_unmeasured_and_too_few_sources_are_not_passes():
    rows = [{'source_id': str(i), 'style': 'identity', 'truth': {'uep': truth},
             'scores': {'uep': score}} for i, truth, score in [(0, 1, None), (1, 0, 0), (2, None, 0)]]
    value = detection(rows, 'uep', 'uep', 0, config())
    assert value['status'] == 'INSUFFICIENT_TRUTH'
    assert value['positive']['coverage'] == 0 and value['positive']['rate'] == 0
    assert value['unlabelled_or_inapplicable'] == 1
    report = generated_report([{**r, 'truth': {}} for r in rows], {}, config())
    assert report['status'] == 'REQUIRES_INDEPENDENT_RECONSTRUCTION_TRUTH'
    assert not report['model_ranking_validated']


def test_truth_requires_exact_pair_timeline_and_external_evidence(tmp_path):
    row = {'case_id': 'a', 'source_sha256': 's', 'reconstruction_sha256': 'r', 'alignment': {'fps': 8}, 'truth': {}}
    evidence = tmp_path / 'annotations.json'
    c.save(evidence, {'external_annotation': True})
    label = {**row, 'alignment_sha256': c.digest(row['alignment']), 'truth': {'uep': .5},
             'provenance': 'independent annotated tracks', 'evidence_files': c.file_inventory([evidence]),
             'uep_evidence_scope': 'received_keyframes_only'}
    path = tmp_path / 'truth.json'
    c.save(path, {'schema': 'metric-campaign-independent-truth-v1', 'rows': [label]})
    assert attach_truth([row], path)[0]['truth']['uep'] == .5
    changed = {**row, 'alignment': {'fps': 4}}
    with pytest.raises(ValueError, match='timeline'):
        attach_truth([changed], path)
    evidence.write_text('edited')
    with pytest.raises(ValueError, match='changed'):
        attach_truth([row], path)


def test_ablation_scoring_executes_without_truth_or_rgb_models():
    from test_metric_v5 import observation, track
    from semantic_transmission.motion_metric import describe_flow
    a = observation([track(range(20))], [('stop', 4, 0), ('exit', 12, 0)])
    b = copy.deepcopy(a)
    for x in (a, b):
        x['flow'] = np.zeros((19, 24, 32, 2), np.float32)
        x['motion'] = describe_flow(x['flow'])
    scores = visual_scores(a, b, a, ['ok'] * 20)
    assert scores['uep'] == scores['uep_source_only'] == 0
    assert scores['eoi'] == scores['eoi_kendall'] == 0


def test_seed_overlap_and_invalid_size_rejected():
    cfg = config()
    cfg['rx_seed'] = cfg['synthetic_seed']
    with pytest.raises(ValueError, match='overlap'):
        c.validate_config(cfg)
    cfg = config()
    cfg['natural_frames'] = 1
    with pytest.raises(ValueError, match='16..128'):
        c.validate_config(cfg)


def test_natural_truth_uses_masks_and_edit_indices_and_does_not_label_unknown_objects(tmp_path):
    import cv2
    from PIL import Image
    cfg = {**config(), 'styles': ['identity'], 'natural_sources': 1, 'reconstruction_sources': 1}
    a, truth = cases.draw(cases.scene(123, cases.FAMILIES[0]))
    images, labels = [], []
    for i, (frame, mask) in enumerate(zip(a, truth['masks'])):
        image, label = tmp_path / f'{i}.jpg', tmp_path / f'{i}.png'
        cv2.imwrite(str(image), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        palette = Image.fromarray(mask.astype(np.uint8)).convert('P')
        palette.putpalette([component for k in range(256) for component in (k, (k * 3) % 256, (k * 7) % 256)])
        palette.save(label)
        images.append(str(image)); labels.append(str(label))
    cases.natural_corpus(tmp_path, cfg, [{'source_id': 'davis/test', 'images': images, 'labels': labels}])
    rows = c.read(tmp_path / '03_natural/cases.json')['rows']
    assert len(rows) == 15
    assert next(r for r in rows if r['kind'] == 'control')['truth']['uep'] == 0
    assert all(r['truth']['uep'] is None for r in rows if r['kind'] in ('reverse', 'freeze', 'lag', 'swap'))
    for r in rows:
        if r['kind'] == 'addition':
            assert r['truth']['uep'] > 0 and not r['truth'].get('fso_presence')


def test_export_preserves_declared_duration_and_reconstruction_resume(tmp_path, monkeypatch, local_campaign_repo):
    from semantic_transmission import metric_campaign_real as real
    from semantic_transmission.video_io import probe
    corpus = cases.Corpus(tmp_path, '03_natural')
    rgb = np.full((16, 192, 320, 3), 127, np.uint8)
    descriptor = corpus.put(rgb)
    dest = tmp_path / 'probe.mp4'
    export_source(descriptor, dest)
    info = probe(dest)
    assert (info['frames'], info['width'], info['height'], info['fps']) == (48, 576, 320, 24)
    c.save(tmp_path / '03_natural/sources.json', [{'source_id': 'davis/test', 'array': descriptor}])
    cfg = {**config(), 'reconstruction_sources': 1, 'reconstruction_steps': [30]}
    # Inject only the costly model execution and its old verifier. The outer
    # adapter still exports, freezes, hashes and reuses its actual artifacts.
    calls = []
    def runner(attempt, profile, source, config):
        calls.append(attempt)
        run = attempt / 'davis__test'
        video = run / 'receiver/reconstruction/sample.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'fake model output for orchestration test')
        c.save(attempt / 'profile.json', {'test': True})
        c.save(attempt / 'batch_manifest.json', {'test': True})
        return SimpleNamespace(returncode=0)
    def validator(attempt, profile, source):
        run = attempt / source['id']
        return {'path': run} if (run / 'receiver/reconstruction/sample.mp4').exists() else None
    monkeypatch.setattr(real, 'run_attempt', runner)
    monkeypatch.setattr(real, 'validated_attempt', validator)
    reconstruct(tmp_path, cfg)
    assert len(calls) == 1
    reconstruct(tmp_path, cfg)
    assert len(calls) == 1
    pairs = c.read(tmp_path / '03_natural/real_pairs.json')['rows']
    assert len(pairs) == 1 and pairs[0]['truth'] == {}
    Path(pairs[0]['reconstruction']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        reconstruct(tmp_path, cfg)


def test_model_change_is_compared_against_truth_not_assumed_from_step_count():
    rows = []
    for sid in range(3):
        for step, truth, score in [(30, .1, .1), (10, .8, .8)]:
            rows.append({'case_id': f'{sid}_{step}', 'source_id': str(sid), 'steps': step,
                         'truth': {'uep': truth}, 'scores': {'uep': score}})
    report = model_change_agreement(rows, config())['uep']
    assert report['status'] == 'PASSED' and report['direction_agreement'] == 1
    for row in rows:
        row['scores']['uep'] = 1 - row['scores']['uep']
    assert model_change_agreement(rows, config())['uep']['status'] == 'NOT_PASSED'


def test_actual_lgvsc_stage_order_and_driver_environment_are_preserved(tmp_path, local_campaign_repo):
    from semantic_transmission.metric_campaign_real import run_attempt
    from semantic_transmission.resume import STAGES
    cfg = config()
    profile = c.read(c.resolve(cfg['reconstruction_profile']))
    source = {'id': 'test', 'path': '/test/input.mp4', 'sha256': 'fixture',
              'width': 576, 'height': 320, 'frames': 96, 'fps': 24., 'duration': 4.}
    seen = []
    def runner(command, **kwargs):
        stage = command[3]
        seen.append(stage)
        assert kwargs['env']['LD_LIBRARY_PATH'] == str(c.resolve(cfg['driver_library']))
        if stage == 'channel':
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '-1'
        return SimpleNamespace(returncode=0)
    attempt = tmp_path / 'a/b/c/d/attempt_001'
    run_attempt(attempt, profile, source, cfg, runner=runner)
    assert seen == STAGES
    record = c.read(attempt / 'test/run_manifest.json')
    assert all(s['status'] == 'PASSED' and s['returncode'] == 0 for s in record['stages'])
    seen.clear()
    def fails(command, **kwargs):
        seen.append(command[3])
        return SimpleNamespace(returncode=1 if command[3] == 'select' else 0)
    failed = tmp_path / 'a/b/c/d/attempt_002'
    with pytest.raises(RuntimeError, match='select failed'):
        run_attempt(failed, profile, source, cfg, runner=fails)
    assert seen == ['prepare', 'select']
    assert c.read(failed / 'test/run_manifest.json')['status'] == 'FAILED'
