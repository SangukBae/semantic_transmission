"""Optional partial-coverage fallback using exact serialized packet costs.

Never changes a legacy packet that already fits. Utility uses sender geometry,
not reconstruction scores. An empty interval is an explicit receiver fallback.
"""
import json
import math
import numpy as np
from .bsu_sequence_refinement import POLICY, choose_packet
from .packets import accounting
from .sparse_trajectory import decode
from .wire import pack

ALLOCATION_POLICY='sender_geometry_multiple_choice_knapsack_fallback_v1'


def utility(window):
    d=decode(window['trajectory'],window['end']-window['start']+1)
    visible=d['valid'];xy=d['xy']
    adjacent=visible[1:]&visible[:-1]
    speed=np.linalg.norm(np.diff(xy,axis=0),axis=-1)
    motion=float(speed[adjacent].mean()) if adjacent.any() else 0.
    # Bounded activity weighting avoids spending everything on camera motion.
    coverage=float(visible.any(axis=1).mean())
    support=min(1.,xy.shape[1]/12)
    return (window['end']-window['start'])*coverage*(1+min(motion,4.))*support


def allocate(candidates,video,base_digest,max_uses,bits_per_symbol=4):
    if type(max_uses) is not int or max_uses<0:raise ValueError('invalid channel budget')
    old,info=choose_packet(candidates,video,base_digest,max_uses,bits_per_symbol)
    if old is not None:return old,dict(info,allocation_policy=ALLOCATION_POLICY,legacy_preserved=True)
    one=accounting(1,bits_per_symbol=bits_per_symbol)['complex_channel_uses']
    blocks=max_uses//one
    if not blocks:return None,dict(info,allocation_policy=ALLOCATION_POLICY,reason='budget below one LDPC block',legacy_preserved=True)
    header=dict(policy=POLICY,video=video,base_metadata_sha256=base_digest,windows=[])
    capacity=blocks*6144//8-len(pack(header))
    if capacity<1:return None,dict(info,allocation_policy=ALLOCATION_POLICY,reason='header exceeds budget')
    # charge one comma per selected window, and add one byte of capacity because
    # the first window has no leading comma. All JSON fields remain unchanged.
    capacity+=1
    states={0:(0.,[])}
    for index,options in enumerate(candidates):
        choices=[]
        for level,w in enumerate(options):
            cost=len(json.dumps(w,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())+1
            value=utility(w)
            if math.isfinite(value) and value>0 and cost<=capacity:choices.append((cost,value,level,w))
        merged=dict(states)
        for cost,(value,path) in states.items():
            for extra,gain,level,w in choices:
                total=cost+extra
                if total<=capacity and (total not in merged or value+gain>merged[total][0]):
                    merged[total]=(value+gain,path+[(index,level,w)])
        # Drop dominated states: higher cost without greater utility cannot win.
        best=-1.;states={}
        for cost,(value,path) in sorted(merged.items()):
            if value>best:
                states[cost]=(value,path);best=value
    _,(value,path)=max(states.items(),key=lambda row:(row[1][0],-row[0]))
    if not path:return None,dict(info,allocation_policy=ALLOCATION_POLICY,reason='no individual useful window fits')
    header['windows']=[w for i,l,w in path];packet=pack(header)
    rate=accounting(len(packet),bits_per_symbol=bits_per_symbol)
    if rate['complex_channel_uses']>max_uses:raise AssertionError('serialized allocation exceeds channel cap')
    return packet,dict(allocation_policy=ALLOCATION_POLICY,legacy_preserved=False,
        selected=[dict(index=i,level=l,start=w['start'],end=w['end']) for i,l,w in path],
        sent_windows=len(path),omitted_windows=len(candidates)-len(path),sender_utility=value,
        packet_bytes=len(packet),complex_channel_uses=rate['complex_channel_uses'],
        budget_cap=max_uses,optimality='exact multiple-choice additive-byte knapsack for the declared sender utility')
