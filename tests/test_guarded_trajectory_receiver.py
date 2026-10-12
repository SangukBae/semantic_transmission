import numpy as np
from semantic_transmission.guarded_trajectory_receiver import refine_references


def test_incompatible_silhouette_is_not_overlaid():
    keys=[np.zeros((160,288,3),np.uint8) for _ in range(2)]
    frames=np.zeros((9,160,288,3),np.uint8);frames[:,:,130:150]=255
    points=np.array([[30,30],[140,30],[250,30],[30,130],[140,130],[250,130]]*2,np.float32)
    trajectory=dict(frames=9,xy=np.repeat(points[None],9,axis=0),valid=np.ones((9,12),bool),origins=np.repeat([0,1],6))
    output,audit=refine_references(frames,keys,trajectory)
    assert np.array_equal(output,frames)
    assert audit['mean_rejected_area']>0


def test_compatible_small_color_error_is_corrected():
    keys=[np.full((160,288,3),100,np.uint8) for _ in range(2)]
    frames=np.full((9,160,288,3),108,np.uint8)
    points=np.array([[30,30],[140,30],[250,30],[30,130],[140,130],[250,130]]*2,np.float32)
    trajectory=dict(frames=9,xy=np.repeat(points[None],9,axis=0),valid=np.ones((9,12),bool),origins=np.repeat([0,1],6))
    output,_=refine_references(frames,keys,trajectory)
    assert output[4].mean()<frames[4].mean()
    assert np.array_equal(output[[0,-1]],frames[[0,-1]])
