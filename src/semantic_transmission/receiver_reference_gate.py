"""Small learned receiver policy: preserve history or partially re-denoise it.

Inference features use only received keys and previously generated pixels.
Offline targets measure actual counterfactual reconstruction utility. This is
not end-to-end diffusion training, a semantic verifier, or output interpolation.
"""
import math

EDIT_RATIO = .35
FEATURES = ('anchor_l1','anchor_rmse','anchor_edge_error','generated_contrast',
            'received_contrast','history_mad_mean','history_mad_p95',
            'history_flow_mean','history_flow_p95','last_transition_mad',
            'anchor_brightness_error','received_next_l1','previous_gap_seconds','next_gap_seconds')


def features(history, received_current, received_next, gaps, fps):
    import cv2
    import numpy as np
    if (history.ndim != 4 or len(history) != 17 or history.shape[-1] != 3
            or received_current.shape != history.shape[1:] or received_next.shape != history.shape[1:]
            or history.dtype != np.uint8 or received_current.dtype != np.uint8 or received_next.dtype != np.uint8
            or len(gaps) != 2 or min(gaps) <= 0 or not math.isfinite(fps) or fps <= 0):
        raise ValueError('invalid receiver-only feature inputs')
    def gray(im):
        return cv2.cvtColor(cv2.resize(im,(144,80),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
    hist=[gray(x) for x in history]
    current,nxt=gray(received_current),gray(received_next)
    a,b=hist[-1].astype(float)/255,current.astype(float)/255
    edges=lambda x: cv2.Sobel(x,cv2.CV_64F,1,0)+cv2.Sobel(x,cv2.CV_64F,0,1)
    mad=[float(np.abs(y.astype(float)-x).mean()/255) for x,y in zip(hist,hist[1:])]
    motion=[]
    for x,y in zip(hist,hist[1:]):
        flow=cv2.calcOpticalFlowFarneback(x,y,None,.5,3,15,3,5,1.2,0)
        motion.append(float(np.linalg.norm(flow,axis=2).mean()))
    result=[float(np.abs(a-b).mean()),float(np.sqrt(np.square(a-b).mean())),
        float(np.abs(edges(a)-edges(b)).mean()),float(a.std()),float(b.std()),
        float(np.mean(mad)),float(np.percentile(mad,95)),
        float(np.log1p(np.mean(motion))),float(np.log1p(np.percentile(motion,95))),mad[-1],
        abs(float(a.mean()-b.mean())),float(np.abs(current.astype(float)-nxt).mean()/255),
        gaps[0]/fps,gaps[1]/fps]
    if len(result)!=len(FEATURES) or not np.isfinite(result).all(): raise ValueError('nonfinite gate features')
    return result


def relax_history(mask, record, ratio):
    """Only older generated-history slots may change; newest and end key stay fixed."""
    import torch
    if not math.isfinite(ratio) or not 0 <= ratio <= 1: raise ValueError('invalid edit ratio')
    result=mask.clone()
    if record['loop']==0:
        return result,[]
    history=next(op for op in record['operations'] if op['length']==5)
    end=next(op for op in record['operations'] if op['length']==1)
    slots=list(range(history['target'],history['target']+history['length']-1))
    if end['target'] in slots or torch.any(mask[:,slots]!=0): raise ValueError('history/end condition overlap')
    result[:,slots]=ratio
    if torch.any(result[:,end['target']]!=0) or torch.any(result[:,history['target']+4]!=0):
        raise ValueError('hard anchors must remain fixed')
    return result,slots


def utility(baseline, edited):
    """Predeclared motion gain with non-key fidelity regression penalties."""
    gain=(baseline['flow_error']-edited['flow_error'])/max(baseline['flow_error'],.01)
    penalty=max(0.,(edited['lpips_vgg']-baseline['lpips_vgg'])/.01)
    penalty+=max(0.,(baseline['psnr_db']-edited['psnr_db'])/.3)
    return max(-2.,min(2.,gain-penalty))


def assert_source_split(cases):
    train={c['source_id'] for c in cases if c['split']=='train'}
    test={c['source_id'] for c in cases if c['split']=='test'}
    if not train or not test or train & test:
        raise ValueError('training and evaluation must use different original videos')
    # Byte identity is an additional safeguard, not a substitute for source IDs:
    # crops/normalizations of one original can have different video hashes.
    a={c['source_sha256'] for c in cases if c['split']=='train'}
    b={c['source_sha256'] for c in cases if c['split']=='test'}
    if a & b: raise ValueError('source bytes overlap across splits')


def network():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(len(FEATURES),16),torch.nn.SiLU(),
        torch.nn.Linear(16,8),torch.nn.SiLU(),torch.nn.Linear(8,1))


def fit(records, steps=400, seed=31415, progress=None):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    x=torch.tensor([r['features'] for r in records],dtype=torch.float32)
    y=torch.tensor([r['utility'] for r in records],dtype=torch.float32).view(-1,1)
    groups=[r['source_id'] for r in records]
    weights=torch.tensor([1/groups.count(g) for g in groups],dtype=torch.float32).view(-1,1)
    weights/=weights.sum()
    mean=(x*weights).sum(0)
    scale=((x-mean).square()*weights).sum(0).sqrt().clamp_min(1e-4)
    model=network()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    for step in range(steps):
        prediction=model((x-mean)/scale)
        loss=(torch.nn.functional.smooth_l1_loss(prediction,y,reduction='none')*weights).sum()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if progress and (step==0 or (step+1)%100==0):progress(step+1,steps,float(loss))
    model.eval().requires_grad_(False)
    return dict(state_dict=model.state_dict(),mean=mean,scale=scale,feature_names=list(FEATURES),
        steps=steps,seed=seed,edit_ratio=EDIT_RATIO,decision_threshold=.05,
        training_source_ids=sorted(set(groups)),parameter_count=sum(p.numel() for p in model.parameters())),float(loss)


def predict(checkpoint, values):
    import torch
    if checkpoint['feature_names'] != list(FEATURES) or len(values)!=len(FEATURES):
        raise ValueError('gate feature schema mismatch')
    with torch.random.fork_rng():
        model=network()
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.inference_mode():
        x=torch.tensor(values,dtype=torch.float32)
        prediction=float(model((x-checkpoint['mean'])/checkpoint['scale']).squeeze())
    if not math.isfinite(prediction): raise ValueError('nonfinite gate output')
    return prediction,checkpoint['edit_ratio'] if prediction>checkpoint['decision_threshold'] else 0.
