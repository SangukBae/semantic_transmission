from pathlib import Path

import pytest

from semantic_transmission import lgvsc_kinetics_recovery as repair
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid_ablation import snapshot


def signed(value):
    return repair.action.base.signed(value)


def test_repair_is_pinned_to_the_tested_validator():
    assert sha256(repair.run.REPO/repair.ACTION_FILE) == repair.FIXED_SHA


def plan_fixture(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; root = repo/'output'; root.mkdir(parents=True)
    current = repair.run.REPO/repair.ACTION_FILE
    target = repo/repair.ACTION_FILE; target.parent.mkdir(parents=True); target.write_bytes(current.read_bytes())
    other = repo/'other.py'; other.write_bytes(b'unchanged')
    inputs = repo/'inputs.json'; inputs.write_text('{}')
    model = repo/'model.bin'; model.write_bytes(b'weights')
    stat = model.stat()
    p = signed(dict(policy=repair.run.POLICY, root=str(root), videos=[dict(id=f'v{i}',frames=2) for i in range(14)],
        inputs={str(inputs):sha256(inputs)}, checkpoints={}, model_files={str(model):dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)},
        code={repair.ACTION_FILE:repair.LEGACY_SHA, 'other.py':sha256(other)}))
    write_json(root/'protocol.json', p); monkeypatch.setattr(repair.run, 'REPO', repo)
    return root, p


@pytest.mark.parametrize('mutation', [None, 'input', 'code', 'model', 'unknown_legacy'])
def test_recovery_accepts_only_the_known_fix_and_unchanged_provenance(tmp_path, monkeypatch, mutation):
    root, p = plan_fixture(tmp_path, monkeypatch)
    before = (root/'protocol.json').read_bytes()
    if mutation == 'input': (root.parent/'inputs.json').write_text('{"changed":true}')
    elif mutation == 'code': (root.parent/'other.py').write_bytes(b'changed')
    elif mutation == 'model': (root.parent/'model.bin').write_bytes(b'different weights')
    elif mutation == 'unknown_legacy':
        p['code'][repair.ACTION_FILE] = 'unknown'
        write_json(root/'protocol.json', signed({k:v for k,v in p.items() if k != 'signature'}))
    if mutation is None:
        assert repair.verify_plan(root) == p
        assert (root/'protocol.json').read_bytes() == before
    else:
        with pytest.raises(ValueError): repair.verify_plan(root)


def completed_fixture(tmp_path, monkeypatch):
    root, p = plan_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(repair.run, 'initialization_files', lambda _:['run/input.bin'])
    monkeypatch.setattr(repair.run, 'jobs', lambda:[('evaluate',['quality.json','output_audit.json','rate.json','video.mp4'])])
    for v in p['videos']:
        dest = root/'videos'/v['id']; work = dest/'run'; work.mkdir(parents=True)
        (work/'input.bin').write_bytes(b'input'); (work/'video.mp4').write_bytes(v['id'].encode())
        audit = dict(status='PASSED',video=str(work/'video.mp4'),video_sha256=sha256(work/'video.mp4'))
        quality = dict(status='PASSED',video_sha256=audit['video_sha256'],lossless_frames=dict(frames=2),delivered_mp4=dict(frames=2))
        write_json(work/'quality.json',quality); write_json(work/'output_audit.json',audit); write_json(work/'rate.json',{})
        write_json(dest/'resources/evaluate.json',dict(returncode=0))
        init = dict(status='PASSED',identity=p['signature'],dependencies={},required=['run/input.bin'],artifacts=snapshot(dest,['run/input.bin']))
        write_json(dest/'stages/initialize.json',init)
        required = ['run/'+f for f in repair.run.jobs()[0][1]]+['resources/evaluate.json']
        receipt = dict(status='PASSED',identity=p['signature'],dependencies=dict(initialize=repair.fingerprint(init)),
            required=required,artifacts=snapshot(dest,required))
        write_json(dest/'stages/evaluate.json',receipt)
        write_json(dest/'RESULT.json',dict(status='COMPLETE_RECONSTRUCTION_QUALITY',signature=p['signature'],id=v['id'],frames=2,
            video=audit['video'],video_sha256=audit['video_sha256'],quality=quality,rate={}))
    return root, p


@pytest.mark.parametrize('mutation', [None, 'movie', 'dependency', 'missing', 'summary'])
def test_recovery_audits_completed_stages_without_rewriting_or_launching(tmp_path, monkeypatch, mutation):
    root, p = completed_fixture(tmp_path, monkeypatch)
    folder = root/'videos/v0'
    if mutation == 'movie': (folder/'run/video.mp4').write_bytes(b'corrupt')
    elif mutation == 'dependency':
        path = folder/'stages/evaluate.json'; r = repair.read(path); r['dependencies'] = {}; write_json(path,r)
    elif mutation == 'missing': (folder/'stages/evaluate.json').unlink()
    elif mutation == 'summary':
        path = folder/'RESULT.json'; r = repair.read(path); r['frames'] = 3; write_json(path,r)
    before = {str(f):sha256(f) for f in root.rglob('*') if f.is_file()}
    monkeypatch.setattr(repair.run.batch,'command',lambda *_:pytest.fail('must not run model stages during audit'))
    if mutation is None:
        rows, receipts = repair.completed_results(root,p)
        assert len(rows) == 14 and len(receipts) == 42
    else:
        with pytest.raises((ValueError,FileNotFoundError)): repair.completed_results(root,p)
    assert before == {str(f):sha256(f) for f in root.rglob('*') if f.is_file()}
