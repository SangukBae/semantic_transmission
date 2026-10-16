import numpy as np
from semantic_transmission.learned_trajectory import candidates_from_tracks
from semantic_transmission.sparse_trajectory import decode


def test_learned_tracks_keep_the_existing_quantized_endpoint_contract():
    xy=np.zeros((49,20,2),np.float32)
    xy[:,:,0]=np.arange(20)[None]*10+15+np.arange(49)[:,None]*.25
    xy[:,:,1]=40
    valid=np.ones((49,20),bool);valid[15:22,0]=False
    rows=candidates_from_tracks([(xy,valid),(xy,valid)],100,148)
    assert len(rows)==7
    for w in rows:
        d=decode(w['trajectory'],49)
        assert d['xy'].shape[0]==49
        assert set(d['origins'])=={0,1}
        assert w['start']==100 and w['end']==148
        assert np.isfinite(d['xy']).all()
