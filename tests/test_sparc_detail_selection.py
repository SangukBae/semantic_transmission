import numpy as np
import pytest

from semantic_transmission import sparc_detail_selection as detail


def test_static_long_sequence_has_no_spacing_limit():
    images = np.zeros((151,32,32,3),np.uint8)
    result = detail.choose(images,[0,150],[],detail.config())
    assert result['indices']==[0,150]
    assert result['max_gap'] is None
    assert not result['receiver_quality_used']
    assert result['codec_calls_during_selection']==0


def test_small_local_event_not_hidden_by_global_average():
    images = np.zeros((21,64,64,3),np.uint8)
    images[10,16:24,16:24] = 160
    assert images[10].mean()<6
    result = detail.choose(images,[0,20],[],detail.config())
    assert 10 in result['indices']
    assert all(row['score']<=1 for row in result['interval_checks'])


def test_semantic_keys_and_scene_cut_sides_survive_pruning():
    images = np.zeros((51,32,32,3),np.uint8)
    result = detail.choose(images,[0,13,50],[26],detail.config())
    assert result['indices']==[0,13,25,26,50]
    assert not result['removals']


def test_linear_fade_does_not_require_uniform_samples():
    images = np.stack([np.full((32,32,3),i*4,np.uint8) for i in range(51)])
    result = detail.choose(images,[0,50],[],detail.config())
    assert result['indices']==[0,50]


def test_all_remaining_auxiliaries_needed_under_source_criterion():
    rng = np.random.default_rng(5)
    images = rng.integers(0,256,(17,32,32,3),dtype=np.uint8)
    settings = detail.config()
    result = detail.choose(images,[0,16],[],settings)
    signal = detail.Signals(images,settings)
    for a,i,b in zip(result['indices'],result['indices'][1:],result['indices'][2:]):
        assert signal.interval(a,b)['score']>1


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf'),True,'1'])
def test_reject_invalid_threshold_scale(scale):
    with pytest.raises(ValueError):
        detail.config(scale)
