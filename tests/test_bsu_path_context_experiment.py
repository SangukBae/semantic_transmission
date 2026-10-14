from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image
import pytest
from semantic_transmission import bsu_path_context_experiment as experiment
from semantic_transmission import etri_60s_check
from semantic_transmission.receiver_key_context import preserve_context
from semantic_transmission.wire import pack
from semantic_transmission.artifacts import write_json,sha256


def test_receiver_rejects_source_pixels(tmp_path):
    with pytest.raises(ValueError,match='received inputs only'):
        experiment.generate({'source_frames':[]},tmp_path,'fixed')


def test_generation_failure_resume_baseline_identity_and_keys(tmp_path,monkeypatch):
    cv2.setNumThreads(2);paths=[];keys=[]
    frames=np.stack([np.full((320,576,3),40+i,np.uint8) for i in range(9)])
    for i,a in enumerate(frames):
        p=tmp_path/f'in_{i}.png';Image.fromarray(a).save(p);paths.append(str(p))
    for i in (0,4,8):
        p=tmp_path/f'key_{i}.png';Image.fromarray(np.full((320,576,3),70+i,np.uint8)).save(p);keys.append(str(p))
    root=tmp_path/'new';old=tmp_path/'old';name='video/snr_10'
    write_json(root/'protocol.json',dict(parent=str(old),cases=[dict(name=name)]))
    donor=old/'comparison'/name/'context_only';donor.mkdir(parents=True)
    output=frames.copy()
    for j,(a,b) in enumerate(((0,4),(4,8))):
        output[a:b+1],_=preserve_context(frames[a:b+1],[np.asarray(Image.open(p)) for p in keys[j:j+2]])
    baseline_paths=[]
    for i,a in enumerate(output):
        p=donor/f'{i}.png';Image.fromarray(a).save(p);baseline_paths.append(str(p))
    write_json(donor/'generation_info.json',dict(frames_paths=baseline_paths))
    packet=tmp_path/'packet';packet.write_bytes(pack(dict(video=dict(frames=9,fps=24,width=576,height=320),
        keyframes=[dict(index=i) for i in (0,4,8)],segments=[])))
    job=dict(frames=9,fps=24,keys=[0,4,8],frames_paths=paths,key_paths=keys,base_packet=str(packet),strength=1.,align=True)
    inputs={p:sha256(Path(p)) for p in paths+keys+baseline_paths};calls=[]
    def encode(command,**kwargs):
        calls.append(command)
        if len(calls)==1:raise subprocess.CalledProcessError(9,command)
        Path(command[-1]).write_bytes(b'encoded')
    monkeypatch.setattr(subprocess,'run',encode)
    monkeypatch.setattr(etri_60s_check,'pts_audit',lambda *args:dict(frames=9,fps=24))
    folder=root/'comparison'/name/'path_context'
    with pytest.raises(subprocess.CalledProcessError):experiment.generate(job,folder,'fixed')
    assert not (folder/'generation.json').exists()
    experiment.generate(job,folder,'fixed')
    assert experiment.common.verify_receipt(folder,'generation','fixed')
    info=experiment.read(folder/'generation_info.json')
    assert all(t['baseline_exact'] and t['seconds']['path_context']>0 for t in info['timings'])
    assert info['timings'][0]['order']!=info['timings'][1]['order']
    for i,p in zip(job['keys'],keys):
        assert np.array_equal(np.asarray(Image.open(folder/f'frames/{i:05d}.png')),np.asarray(Image.open(p)))
    assert inputs=={p:sha256(Path(p)) for p in inputs}
    # A corrupted baseline is rejected rather than used as a fresh timing reference.
    Image.fromarray(np.zeros((320,576,3),np.uint8)).save(baseline_paths[1])
    with pytest.raises(ValueError,match='timed baseline differs'):
        experiment.generate(job,folder,'fixed')
