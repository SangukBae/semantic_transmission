"""Mask-IoU IDF1 and framewise detection F1 on the same predicted tracks.

IDF1 follows global identity counting (Ristani et al., ECCVW 2016); the reference
tracks here are predicted, not annotated truth. Mask IoU >= .5 replaces box IoU.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .object_metric import iou


def identity_baselines(a, b, threshold=.5):
    na, nb = sum(map(len,a)), sum(map(len,b))
    if not na:
        return {"idf1_mask_error":None,"frame_mask_error":None}
    counts=np.zeros((len(a),len(b)))
    per_frame={}
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            for t in set(x)&set(y):
                if iou(x[t]["mask"],y[t]["mask"])>=threshold:
                    counts[i,j]+=1
                    per_frame.setdefault(t,np.zeros_like(counts))[i,j]=1
    ids=linear_sum_assignment(-counts)
    shared=counts[ids].sum()
    frame_matches=sum(matrix[linear_sum_assignment(-matrix)].sum() for matrix in per_frame.values())
    return {"idf1_mask_error":float(1-2*shared/(na+nb)),"frame_mask_error":float(1-2*frame_matches/(na+nb)),
            "idtp":int(shared),"idfn":int(na-shared),"idfp":int(nb-shared)}
