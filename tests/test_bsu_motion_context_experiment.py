from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_motion_context_experiment as experiment
from semantic_transmission import etri_60s_check
from semantic_transmission.wire import pack


def test_receiver_rejects_original_evaluation_frames(tmp_path):
    with pytest.raises(ValueError, match='received inputs only'):
        experiment.generate({'source_frames': []},tmp_path,'frozen')


def test_multisegment_failure_resume_keeps_exact_keys_and_input_files(tmp_path,monkeypatch):
    paths=[];keys=[]
    for i in range(9):
        a=np.full((320,576,3),40+i,np.uint8)
        path=tmp_path/f'input_{i}.png';Image.fromarray(a).save(path);paths.append(str(path))
    for i in (0,4,8):
        path=tmp_path/f'key_{i}.png';Image.fromarray(np.full((320,576,3),70+i,np.uint8)).save(path);keys.append(str(path))
    packet=tmp_path/'metadata.bin';packet.write_bytes(pack(dict(video=dict(frames=9,fps=24,width=576,height=320),keyframes=[dict(index=i) for i in (0,4,8)],segments=[])))
    inputs={p:experiment.sha256(Path(p)) for p in paths+keys};folder=tmp_path/'output';calls=[]
    def encode(command,**kwargs):
        calls.append(command)
        if len(calls)==1:raise subprocess.CalledProcessError(9,command)
        Path(command[-1]).write_bytes(b'encoded')
    monkeypatch.setattr(subprocess,'run',encode)
    monkeypatch.setattr(etri_60s_check,'pts_audit',lambda *a:dict(frames=9,fps=24))
    job=dict(frames=9,fps=24,keys=[0,4,8],frames_paths=paths,key_paths=keys,base_packet=str(packet),strength=1.,align=True)
    with pytest.raises(subprocess.CalledProcessError):experiment.generate(job,folder,'frozen')
    assert not (folder/'generation.json').exists()
    experiment.generate(job,folder,'frozen')
    assert experiment.common.verify_receipt(folder,'generation','frozen')
    assert inputs=={p:experiment.sha256(Path(p)) for p in inputs}
    for i,key in zip(job['keys'],keys):assert np.array_equal(np.asarray(Image.open(folder/f'frames/{i:05d}.png')),np.asarray(Image.open(key)))
    (folder/'frames/00001.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):experiment.common.verify_receipt(folder,'generation','frozen')
