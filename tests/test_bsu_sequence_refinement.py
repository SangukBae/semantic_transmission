import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_sequence_refinement as sequence
from semantic_transmission.packets import accounting
from semantic_transmission.sparse_trajectory import encode
from semantic_transmission.wire import pack, unpack


def packets():
    video = dict(frames=13, fps=24, width=48, height=32)
    base = pack(dict(video=video, keyframes=[dict(index=i) for i in [0,6,12]]))
    xy = np.array([[[5,5],[25,5],[5,25]], [[8,5],[28,5],[8,25]]], dtype=np.float32)
    encoded = encode(xy, np.ones((2,3),bool), np.zeros(3), [0,6], 7)
    candidates = [[dict(start=a,end=b,trajectory=encoded) for _ in sequence.TRIALS] for a,b in [(0,6),(6,12)]]
    packet,audit = sequence.choose_packet(candidates,video,hashlib.sha256(base).hexdigest(),2304)
    return base,packet,audit,candidates


def test_all_window_trajectories_are_paid_in_one_fec_packet():
    base,packet,audit,_ = packets()
    assert audit['sent_windows']==2
    assert set(sequence.decode_packet(packet,base))=={(0,6),(6,12)}
    assert accounting(len(packet))['complex_channel_uses']==2304


def test_insufficient_budget_bypasses_without_free_untransmitted_tracks():
    base,_,_,candidates=packets()
    header,_=unpack(base)
    packet,audit=sequence.choose_packet(candidates,header['video'],hashlib.sha256(base).hexdigest(),2303)
    assert packet is None and audit['sent_windows']==0


def test_qpsk_budget_counts_twice_the_complex_symbols():
    base,_,_,candidates=packets()
    header,_=unpack(base)
    assert sequence.choose_packet(candidates,header['video'],hashlib.sha256(base).hexdigest(),2304,2)[0] is None
    packet,_=sequence.choose_packet(candidates,header['video'],hashlib.sha256(base).hexdigest(),4608,2)
    assert packet and accounting(len(packet),bits_per_symbol=2)['complex_channel_uses']==4608


@pytest.mark.parametrize('mutation',['crc','base_identity','overlap','duplicate'])
def test_receiver_rejects_corruption_and_wrong_timeline(mutation):
    base,packet,_,_=packets()
    header,_=unpack(packet)
    if mutation=='crc': packet=packet[:-1]+bytes([packet[-1]^1])
    else:
        if mutation=='base_identity':header['base_metadata_sha256']='0'*64
        if mutation=='overlap':header['windows'][0]['end']=5
        if mutation=='duplicate':header['windows'].append(header['windows'][0])
        packet=pack(header)
    with pytest.raises(ValueError):sequence.decode_packet(packet,base)


@pytest.mark.parametrize('keys',[[0,6],[0,6,6,12],[1,6,12],[0,12,6]])
def test_invalid_keys_cannot_silently_truncate_or_duplicate_video(keys):
    with pytest.raises(ValueError):sequence.validate_keys(keys,13)


@pytest.mark.parametrize('mode',['none','cadence','combined'])
def test_multikey_full_timeline_and_shared_endpoints_are_preserved(tmp_path,mode):
    frames=tmp_path/'base';keys=tmp_path/'received';out=tmp_path/'out'
    frames.mkdir();keys.mkdir()
    rng=np.random.default_rng(12)
    texture=rng.integers(0,255,(32,48,3),dtype=np.uint8)
    arrays=[np.roll(texture,i//2,axis=1) for i in range(13)]
    paths=[]
    for i,a in enumerate(arrays):
        p=frames/f'{i:05d}.png';Image.fromarray(a).save(p);paths.append(p)
        if i in [0,6,12]:Image.fromarray(a).save(keys/f'{i}.png')
    # There is no source-pixel parameter at this receiver boundary.
    audits=sequence.refine_windows(paths,[0,6,12],keys,{},out,mode)
    assert [p.name for p in sorted(out.glob('*.png'))]==[f'{i:05d}.png' for i in range(13)]
    for i in [0,6,12]:assert np.array_equal(np.asarray(Image.open(out/f'{i:05d}.png')),arrays[i])
    assert [(a['start'],a['end']) for a in audits]==[(0,6),(6,12)]
    if mode=='combined':assert all(a['reference']['bypassed'] for a in audits)
    if mode=='none':
        for i in range(13):assert np.array_equal(np.asarray(Image.open(out/f'{i:05d}.png')),arrays[i])


def test_sender_tracks_use_multiple_intervals_without_originals_at_receiver(tmp_path):
    rng=np.random.default_rng(1)
    texture=rng.integers(0,255,(160,288,3),dtype=np.uint8)
    frames=np.stack([np.roll(texture,i,axis=1) for i in range(9)])
    options=sequence.sender_candidates(frames,20,28)
    assert len(options)==len(sequence.TRIALS)
    assert options[0]['start']==20 and options[0]['end']==28
    from semantic_transmission.sparse_trajectory import decode
    decoded=decode(options[0]['trajectory'],9)
    assert decoded['xy'].shape[1]>=3
    assert decoded['valid'].any()
