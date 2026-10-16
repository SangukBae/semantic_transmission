"""Sender-side constrained temporal/QP allocation; no action labels or clip rules."""
import copy
import math
import statistics

from .key_reduction_selection import CRITERIA, guard

POLICY = 'bsu_adaptive_temporal_qp_v1'
ARMS = ('temporal', 'quantization', 'joint')


def windows(frames, size=24):
    if type(frames) is not int or frames < 3 or size < 4 or size % 12:
        raise ValueError('timeline and a multiple-of-12 window required')
    return [(a, min(a+size, frames-1)) for a in range(0, frames-1, size)]


def schedule(frames, choices, cuts=()):
    """Continuous predictive stream; shared boundaries retain the larger QP."""
    if not choices or choices[0]['start'] != 0 or choices[-1]['end'] != frames-1:
        raise ValueError('incomplete schedule')
    values = {}
    previous = 0
    for block in choices:
        a, b, stride, qp = (block[k] for k in ('start','end','stride','qp'))
        if (any(type(x) is not int for x in (a,b,stride,qp)) or a != previous or
                not a < b < frames or stride not in (2,3,4) or not 0 <= qp <= 63):
            raise ValueError('invalid adaptive block')
        previous = b
        for i in [*range(a,b,stride), b]:
            values[i] = max(values.get(i,0), qp)
    for i in cuts:
        if type(i) is not int or not 0 <= i < frames:
            raise ValueError('invalid cut timestamp')
        values[i] = 32
    values[0] = 32
    indices = sorted(values)
    if any(b-a > 4 for a,b in zip(indices,indices[1:])):
        raise ValueError('uncovered interpolation gap')
    return indices, [values[i] for i in indices]


def metric_checks(base, candidate, margin=1.0):
    """Same registered quality guards, allowing equal cost only for prediction."""
    criteria = {k:v*margin for k,v in CRITERIA.items()}
    b = copy.deepcopy(candidate)
    b['channel_uses'] = base['channel_uses']-1
    return guard(base,b,criteria)


def predicted_quality(base, frame_rows, transitions, original_keys):
    q = copy.deepcopy(base)
    for section, rows in [('quality',frame_rows),
                          ('nonkeys',[r for i,r in enumerate(frame_rows) if i not in original_keys])]:
        for key in ('psnr_db','ssim','clip','lpips_vgg','dists'):
            q[section][key] = statistics.mean(float(r[key]) for r in rows)
    for name, key in [('flow_error_mean','flow_error'),('cadence_error_mean','cadence_error')]:
        q['temporal'][name] = statistics.mean(x for x in transitions[key] if x is not None)
    q['temporal']['false_freeze_rate'] = sum(transitions['false_freeze_cells'])/max(1,base['temporal']['active_motion_cells'])
    return q


def allocate(base, baseline_rows, candidates, frames, keys, arm, size=24, margin=.75):
    """Greedy block search using measured candidate distortions, then re-encode.

    Candidate rates are only proposals; final admission uses whole-stream bytes
    and an independent full reconstruction. All thresholds are clip-independent.
    """
    if arm not in ARMS or len(baseline_rows) != frames:
        raise ValueError('invalid allocation inputs')
    blocks = [dict(start=a,end=b,stride=2,qp=32,candidate='baseline') for a,b in windows(frames,size)]
    proposals = []
    for name,c in candidates.items():
        stride,qp = c['stride'],c['qp']
        if (arm == 'temporal' and qp != 32) or (arm == 'quantization' and stride != 2):
            continue
        if len(c['frame_rows']) != frames:
            raise ValueError('candidate frame count differs')
        saving = base['channel_uses']-c['quality']['channel_uses']
        if saving <= 0:
            continue
        for bi,block in enumerate(blocks):
            a,b = block['start'],block['end']
            # Rank by measured quality damage per estimated saved symbol.
            damage = 0.
            for key,limit,higher in [('psnr_db',.1,True),('ssim',.005,True),
                    ('clip',.002,True),('lpips_vgg',.01,False),('dists',.01,False)]:
                av = statistics.mean(float(r[key]) for r in baseline_rows[a:b])
                bv = statistics.mean(float(r[key]) for r in c['frame_rows'][a:b])
                allowance = limit if higher else max(abs(av)*limit,1e-9)
                damage += max(0,(av-bv if higher else bv-av)/allowance)
            benefit = saving*(b-a)/frames
            proposals.append(((damage+.01)/max(benefit,1),bi,name))
    rows=copy.deepcopy(baseline_rows)
    transitions=copy.deepcopy(base['temporal']['per_transition'])
    selected=set()
    for _,bi,name in sorted(proposals):
        if bi in selected:
            continue
        c=candidates[name];a,b=blocks[bi]['start'],blocks[bi]['end']
        trial=copy.deepcopy(rows);trial[a:b]=copy.deepcopy(c['frame_rows'][a:b])
        tt=copy.deepcopy(transitions)
        for key in ('flow_error','cadence_error','false_freeze_cells'):
            tt[key][max(0,a-1):b]=c['quality']['temporal']['per_transition'][key][max(0,a-1):b]
        predicted=predicted_quality(base,trial,tt,set(keys))
        if metric_checks(base,predicted,margin)['passed']:
            blocks[bi].update(stride=c['stride'],qp=c['qp'],candidate=name)
            rows,transitions=trial,tt;selected.add(bi)
    return blocks


def rollback(blocks):
    """Deterministic bounded conservative fallback, independent of action labels."""
    result=copy.deepcopy(blocks)
    active=[i for i,b in enumerate(result) if b['stride'] != 2 or b['qp'] != 32]
    for i in active[::2]:
        result[i].update(stride=2,qp=32,candidate='baseline')
    return result
