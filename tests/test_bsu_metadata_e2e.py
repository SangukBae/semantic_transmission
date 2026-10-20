import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_transmission import bsu_metadata_e2e as experiment, metadata_channel
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.packets import accounting
from semantic_transmission.qwen35_pairs_comparison import frame_noise
from semantic_transmission.webvid_ablation import snapshot
from semantic_transmission.wire import pack


def baseline(tmp_path):
    base=tmp_path/'baseline';tx=base/'run/transmitter';rx=base/'run/received'
    tx.mkdir(parents=True);rx.mkdir()
    header=dict(visual_codec=dict(name='NTSCC+',rate_index_encoding='uint8'),
        video=dict(width=576,height=320,frames=9,fps=24),decoder={},
        keyframes=[dict(index=i,rate_count=864,rate_bytes=864,rate_offset=n*864,
                       complex_offset=n*12,complex_count=12,average_power=1.) for n,i in enumerate((0,8))],
        segments=[dict(text='A person moves.',flow=1.,path='sample/0_8.mp4')])
    packet=pack(header,bytes([7])*1728)
    (tx/'metadata.bin').write_bytes(packet);(rx/'metadata.bin').write_bytes(packet)
    values=np.ones(24,dtype='<c8');values.tofile(tx/'visual.c64')
    for n,i in enumerate((0,8)):values[n*12:(n+1)*12]+=frame_noise(i,12,42,10)
    values.tofile(rx/'visual.c64')
    write_json(base/'run/run_config.json',dict(snr_db=10,channel_seed=42,seed=2025))
    row=dict(id='fixture',baseline=str(base),frames=9,video=header['video'],keys=[0,8],
             baseline_total_channel_uses=24+accounting(len(packet))['complex_channel_uses'])
    dest=tmp_path/'new';dest.mkdir()
    return base,row,dest


def test_channel_restores_actual_decoder_bytes_and_preserves_visual_awgn(monkeypatch,tmp_path):
    base,row,dest=baseline(tmp_path);calls=[]
    def transmit(packet,snr,seed):
        calls.append(dict(packet=packet,snr=snr,seed=seed))
        return packet,accounting(len(packet))|dict(bit_errors=0)
    monkeypatch.setattr(metadata_channel,'transmit',transmit)
    experiment.prepare_sender(row,dest)
    assert snapshot(dest,['trajectory_sender'])  # Empty trajectory is still a complete sender artifact.
    experiment.channel(row,dest)
    assert calls[0]['packet']!=(base/'run/transmitter/metadata.bin').read_bytes()
    assert (dest/'run/received/metadata.bin').read_bytes()==(base/'run/received/metadata.bin').read_bytes()
    assert (dest/'run/received/visual.c64').read_bytes()==(base/'run/received/visual.c64').read_bytes()
    report=experiment.read(dest/'run/channel_accounting.json')
    assert report['total_with_trajectory']<row['baseline_total_channel_uses']
    assert report['wire_packet_bytes']==len(calls[0]['packet'])
    assert report['decoder_metadata_bytes']>report['wire_packet_bytes']


def test_corrupt_compressed_channel_never_writes_decoder_input(monkeypatch,tmp_path):
    _,row,dest=baseline(tmp_path);experiment.prepare_sender(row,dest)
    def transmit(packet,snr,seed):
        return packet[:-1]+bytes([packet[-1]^1]),accounting(len(packet))|dict(bit_errors=1)
    monkeypatch.setattr(metadata_channel,'transmit',transmit)
    with pytest.raises(ValueError):experiment.channel(row,dest)
    assert not (dest/'run/received/metadata.bin').exists()


def test_visual_noise_change_is_not_misreported_as_compression_effect(monkeypatch,tmp_path):
    base,row,dest=baseline(tmp_path);experiment.prepare_sender(row,dest)
    monkeypatch.setattr(metadata_channel,'transmit',lambda packet,*_: (packet,accounting(len(packet))|dict(bit_errors=0)))
    (base/'run/received/visual.c64').write_bytes(b'wrong baseline noise')
    with pytest.raises(ValueError,match='AWGN realization'):experiment.channel(row,dest)


def test_conditions_compare_content_and_timing_independently_of_path():
    before=dict(profile=dict(seed=2025),video=dict(frames=9),source_pixels_available=False,
        windows=[dict(start=0,end=8,frames=9,generated_frames=9,prompt='moves',
            images=[dict(path='/old/key.png',frame_idx=0)])],inputs={'/old/key.png':'pixels'})
    after=json.loads(json.dumps(before));after['windows'][0]['images'][0]['path']='/new/key.png'
    after['inputs']={'/new/key.png':'pixels'}
    assert experiment.condition_identity(before)==experiment.condition_identity(after)
    after['inputs']['/new/key.png']='different pixels'
    assert experiment.condition_structure(before)==experiment.condition_structure(after)
    assert experiment.condition_identity(before)!=experiment.condition_identity(after)
    after['windows'][0]['images'][0]['frame_idx']=1
    assert experiment.condition_identity(before)!=experiment.condition_identity(after)


def test_pixel_and_metric_comparison_records_runtime_difference(tmp_path):
    base,row,dest=baseline(tmp_path)
    quality=dict(source_sha256='source',evaluation_profile='frozen',
        lossless_frames=dict(frames=9,**{k:1. for k in experiment.previous.KEYS}),
        delivered_mp4=dict(frames=9,**{k:1. for k in experiment.previous.KEYS}),
        temporal={k:0. for k in ('flow_error_mean','cadence_error_mean','motion_retention_ratio','false_freeze_rate')})
    row.update(baseline_quality=quality,baseline_raw_quality=quality)
    for kind in ('raw','refined'):
        for parent in (base,dest):
            frames=parent/kind/'frames';frames.mkdir(parents=True)
            for i in range(9):Image.fromarray(np.zeros((320,576,3),dtype=np.uint8)).save(frames/f'{i:05d}.png')
            (parent/kind/'reconstruction.mp4').write_bytes(b'test video hash')
        write_json(dest/kind/'quality.json',quality)
    pixels=np.zeros((320,576,3),dtype=np.uint8);pixels[0,0,0]=2
    Image.fromarray(pixels).save(dest/'raw/frames/00003.png')
    experiment.compare_outputs(row,dest)
    result=experiment.read(dest/'comparison.json')
    assert result['boundaries']['raw']['exact_rgb_frames']==8
    assert result['boundaries']['raw']['max_rgb_difference']==2
    assert result['boundaries']['refined']['exact_rgb_frames']==9
    assert all(v==0 for v in result['boundaries']['refined']['delta']['delivered_mp4'].values())
    with pytest.raises(ValueError,match='incomplete cohort'):experiment.finish(tmp_path,dict(rows=[row]),[])


def test_status_counts_passed_intervals_not_attempts(monkeypatch,tmp_path):
    monkeypatch.setattr(experiment.subprocess,'check_output',lambda *a,**k:'0 %, 0 MiB, 16376 MiB')
    write_json(tmp_path/'protocol.json',dict(rows=[{}],intervals=2))
    write_json(tmp_path/'status.json',dict(status='INTERRUPTED',stage='generate',completed=0,error='test stop'))
    write_json(tmp_path/'videos/v/raw/stages/interval_00000.json',dict(status='PASSED'))
    write_json(tmp_path/'videos/v/raw/stages/interval_00001.json',dict(status='FAILED'))
    lines=experiment.status_lines(tmp_path)
    assert any(line.startswith('Intervals') and '1/2 (50.0%)' in line for line in lines)
    assert any('Failure' in line and 'test stop' in line for line in lines)


def test_primary_report_uses_new_paired_control_not_historical_difference(tmp_path):
    rows=[dict(id='v__control'),dict(id='v__compressed')]
    p=dict(rows=rows,source_videos=1,frames=9,generated_frames=18,intervals=2,signature='test')
    results=[]
    for variant,delta,cost in [('control',7.,100),('compressed',0.,90)]:
        boundaries={kind:dict(exact_rgb_frames=9,max_rgb_difference=0,
            delta=dict(delivered_mp4={key:delta for key in experiment.previous.KEYS}),
            temporal_delta=dict(flow_error_mean=delta)) for kind in ('raw','refined')}
        r=dict(id='v__'+variant,variant=variant,channel=dict(total_with_trajectory=cost),
            comparison=dict(boundaries=boundaries))
        write_json(tmp_path/'videos'/r['id']/'RESULT.json',r);results.append(r)
    experiment.finish(tmp_path,p,results)
    r=experiment.read(tmp_path/'RESULT.json')
    assert r['baseline_total_channel_uses']==100 and r['new_total_channel_uses']==90
    assert r['videos']==1 and r['variants']==2 and r['paired_input_conditions_equal']
    assert r['summary']['refined']['delivered_mp4_video_mean_delta']['psnr_db']==0
    assert r['summary']['refined']['temporal_video_mean_delta']['flow_error_mean']==0
