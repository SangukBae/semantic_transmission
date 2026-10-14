import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from semantic_transmission import bsu_motion_retention_experiment as experiment
from semantic_transmission import etri_60s_check, receiver_motion_retention
from semantic_transmission.artifacts import sha256
from semantic_transmission.wire import pack


def test_source_fields_rejected(tmp_path):
    with pytest.raises(ValueError,match='source fields'):
        experiment.generate({'source_frames':[]},tmp_path,'frozen')


def test_cached_window_selection_failure_resume_and_input_integrity(tmp_path,monkeypatch):
    paths={name:[] for name in ('context','motion')}
    for name in paths:
        for i in range(9):
            value=40+i+(20 if name=='motion' and i in (1,3,5,7) else 0)
            path=tmp_path/f'{name}_{i}.png'
            Image.fromarray(np.full((320,576,3),value,np.uint8)).save(path);paths[name].append(str(path))
    packet=tmp_path/'packet';packet.write_bytes(pack(dict(video=dict(frames=9,fps=24,width=576,height=320),
        keyframes=[dict(index=i) for i in (0,4,8)],segments=[])))
    keys=[paths['context'][i] for i in (0,4,8)]
    job=dict(frames=9,fps=24,keys=[0,4,8],context_paths=paths['context'],motion_paths=paths['motion'],
        windows=[dict(endpoints=[dict(radius=2),dict(radius=2)])]*2,key_paths=keys,base_packet=str(packet))
    inputs={p:sha256(Path(p)) for collection in paths.values() for p in collection}
    calls=[]
    def encode(command,**kwargs):
        calls.append(command)
        if len(calls)==1:raise subprocess.CalledProcessError(9,command)
        Path(command[-1]).write_bytes(b'encoded')
    # The selection algorithm has separate motion-based tests. Here, route one
    # complete window through each cache and check boundaries and safe resume.
    monkeypatch.setattr(receiver_motion_retention,'select_motion',lambda c,m,t:(t.start==1,{}))
    monkeypatch.setattr(subprocess,'run',encode)
    monkeypatch.setattr(etri_60s_check,'pts_audit',lambda *a:dict(frames=9,fps=24))
    folder=tmp_path/'output'
    with pytest.raises(subprocess.CalledProcessError):experiment.generate(job,folder,'frozen')
    assert not (folder/'generation.json').exists()
    experiment.generate(job,folder,'frozen')
    assert experiment.runner.common.verify_receipt(folder,'generation','frozen')
    for i in range(9):
        origin=paths['motion' if i in (1,5) else 'context'][i]
        assert sha256(folder/f'frames/{i:05d}.png')==sha256(Path(origin))
    assert inputs=={p:sha256(Path(p)) for p in inputs}
    (folder/'frames/00001.png').unlink()
    with pytest.raises(ValueError,match='completed artifact changed'):
        experiment.runner.common.verify_receipt(folder,'generation','frozen')
