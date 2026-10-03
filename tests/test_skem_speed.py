"""Protect sparse time coverage and downstream reuse boundaries."""
import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path=Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_scan_keeps_endpoints_and_off_grid_transition_neighbors():
    module=load("benchmark_skem_speed")
    values=module.candidates(26,6,[3,25])
    assert values[0]==0 and values[-1]==25
    assert {2,3,4,24,25}.issubset(values)
    assert values==sorted(set(values))
    assert max(b-a for a,b in zip(values,values[1:]))<=6


def test_sparse_sampling_rejects_invalid_time_axis():
    module=load("benchmark_skem_speed")
    with pytest.raises(ValueError): module.candidates(1,6)
    with pytest.raises(ValueError): module.candidates(25,0)


def test_downstream_plan_never_accepts_partial_selector_output(tmp_path):
    module=load("validate_skem_speed")
    (tmp_path / "protocol.json").write_text(json.dumps({"windows":{"people":{}},"modes":["baseline"]}))
    folder=tmp_path / "selection/baseline"
    folder.mkdir(parents=True)
    (folder / "people.json").write_text(json.dumps({"status":"RUNNING","indices":[0,24]}))
    with pytest.raises(ValueError,match="incomplete"):
        module.plan(tmp_path)


def test_common_visual_frame_noise_does_not_depend_on_other_selected_frames():
    import numpy as np
    module=load("validate_skem_speed")
    a=[dict(index=0,complex_offset=0,complex_count=7),dict(index=12,complex_offset=7,complex_count=5)]
    b=[dict(index=0,complex_offset=0,complex_count=7),dict(index=6,complex_offset=7,complex_count=3),
       dict(index=12,complex_offset=10,complex_count=5)]
    first=module.frame_paired_noise(np.zeros(12,dtype='<c8'),a,217,42,10)
    second=module.frame_paired_noise(np.zeros(15,dtype='<c8'),b,217,42,10)
    np.testing.assert_array_equal(first[:7],second[:7])
    np.testing.assert_array_equal(first[7:],second[10:])
    changed=module.frame_paired_noise(np.zeros(12,dtype='<c8'),a,217,43,10)
    assert not np.array_equal(first,changed)


def test_historical_reference_rejects_changed_source(tmp_path):
    module=load("validate_skem_speed")
    source=tmp_path / "source.log"
    source.write_text("audited source")
    imported=tmp_path / "imported.json"
    imported.write_text('{"indices":[0,24]}')
    (tmp_path / "protocol.json").write_text('{"signature":"test-protocol"}')
    policy={"selection_protocol":"test-protocol", "source_files":{str(source):module.sha256(source)},
            "records":{"people":{"path":"imported.json","sha256":module.sha256(imported)}}}
    (tmp_path / "baseline_reference.json").write_text(json.dumps(policy))
    assert module.baseline_reference(tmp_path)==policy
    source.write_text("changed source")
    with pytest.raises(ValueError,match="source changed"):
        module.baseline_reference(tmp_path)


def test_historical_reference_rejects_changed_import(tmp_path):
    module=load("validate_skem_speed")
    imported=tmp_path / "imported.json"
    imported.write_text('{"indices":[0,24]}')
    (tmp_path / "protocol.json").write_text('{"signature":"test-protocol"}')
    policy={"selection_protocol":"test-protocol", "source_files":{},
            "records":{"people":{"path":"imported.json","sha256":module.sha256(imported)}}}
    (tmp_path / "baseline_reference.json").write_text(json.dumps(policy))
    imported.write_text('{"indices":[0,12,24]}')
    with pytest.raises(ValueError,match="imported baseline changed"):
        module.baseline_reference(tmp_path)
