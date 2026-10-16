import numpy as np
from semantic_transmission.sparse_trajectory import encode
from semantic_transmission.trajectory_allocation import allocate
from semantic_transmission.bsu_sequence_refinement import choose_packet
from semantic_transmission.packets import accounting


def windows():
    result=[]
    rng=np.random.default_rng(12)
    for i in range(12):
        xy=rng.uniform([10,10],[220,130],size=(9,16,2)).astype(np.float32)
        row=dict(start=i*48,end=(i+1)*48,trajectory=encode(xy,np.ones((9,16),bool),np.zeros(16,int),np.linspace(0,48,9).astype(int),49))
        result.append([row]*7)
    return result


def test_legacy_packet_that_fits_is_byte_identical():
    c=windows()[:1];v=dict(frames=49,fps=24,height=320,width=576)
    old,_=choose_packet(c,v,'a'*64,50000)
    new,a=allocate(c,v,'a'*64,50000)
    assert old==new and a['legacy_preserved']


def test_partial_coverage_can_fit_when_entire_video_cannot():
    c=windows();v=dict(frames=577,fps=24,height=320,width=576)
    old,_=choose_packet(c,v,'a'*64,4608)
    new,a=allocate(c,v,'a'*64,4608)
    assert old is None and new is not None
    assert 0<a['sent_windows']<len(c)
    assert accounting(len(new))['complex_channel_uses']<=4608


def test_cannot_overcome_minimum_channel_block():
    packet,a=allocate(windows(),dict(frames=577),'a'*64,100)
    assert packet is None and a['reason']=='budget below one LDPC block'
