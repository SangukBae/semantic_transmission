import json

import numpy as np
import pytest

from semantic_transmission import dense_temporal_receiver as temporal
from semantic_transmission import sparc, sparc_selection as selection, sparc_transport as rt
from semantic_transmission import sparc_uf_rate_transport as uf
from semantic_transmission.wire import pack


def test_static_source_keeps_only_semantic_keys_and_cut_boundaries():
    pixels = np.zeros((81,2,2,3),np.float32)
    config = dict(max_gap=None,proxy_mae_threshold=6.)
    result = selection.plan(pixels,[0,20,80],[45],config)
    assert result['indices'] == [0,20,44,45,80]
    assert result['additions'] == []
    assert result['max_gap'] is None
    assert max(np.diff(result['indices'])) > 4
    selection.validate_anchors(result['indices'],81,None,[45])
    with pytest.raises(ValueError):
        selection.validate_anchors([0,20,45,80],81,None,[45])


def test_no_gap_limit_still_captures_interior_motion_excursion():
    pixels = np.zeros((81,2,2,3),np.float32)
    pixels[40] = 100
    result = selection.adaptive(pixels,[0,80],[],max_gap=None)
    assert 40 in result['indices']
    assert all(a['reason']=='source_motion_proxy' for a in result['additions'])


def test_uf_packet_transmits_variable_gap_policy_and_retains_timestamp_checks():
    video = dict(frames=81,height=16,width=16,fps=24)
    coded = [([0,20,80],[30,30],b'encoded'*100)]
    weights = {'image':'test','hts':'test'}
    packet = uf.make(video,[0,80],[],coded,weights,max_gap=None)
    header,payload = uf.parse(packet,weights)
    assert header['max_gap'] is None
    assert header['temporal']['policy'] == temporal.VARIABLE_POLICY
    assert header['chunks'][0]['indices'] == [0,20,80]
    temporal.validate_indices([0,20,80],81,None)
    with pytest.raises(ValueError):
        temporal.validate_indices([0,20,80],81,4)
    header['temporal'] = rt.temporal_identity(4)
    with pytest.raises(ValueError):
        uf.parse(pack(header,payload),weights)
    with pytest.raises(ValueError):
        uf.make(video,[0,80],[],coded,weights)


def test_rt_packet_roundtrip_preserves_explicit_no_gap_policy():
    video = dict(frames=41,height=16,width=16,fps=24)
    coded = pack(dict(rt.CODEC,indices=[0,40],height=16,width=16),b'encoded'*100)
    wire = rt.compress(rt.make(video,[coded],[],None))
    header,_ = rt.parse(rt.decompress(wire))
    assert header['max_gap'] is None
    assert list(rt.decoder_packets(rt.decompress(wire))) == [coded]


@pytest.mark.parametrize('indices',[[0,20,20,80],[0,81],[1,80],[0,40],[0,True,80]])
def test_no_gap_receiver_rejects_invalid_timestamps(indices):
    with pytest.raises(ValueError):
        temporal.validate_indices(indices,81,None)


@pytest.mark.parametrize('gap',[True,False,0,-1,5,4.0,'none'])
def test_invalid_gap_values_still_rejected(gap):
    with pytest.raises(ValueError):
        rt.temporal_identity(gap)


def test_cli_exposes_opt_in_without_changing_legacy_profile(capsys):
    sparc.main(['profile','--no-max-gap'])
    assert json.loads(capsys.readouterr().out)['max_gap'] is None
    assert sparc.profile()['max_gap'] == 4
