"""Provenance checks for reusing source pixels, semantic calls and old BSU tracks."""
from pathlib import Path

from .artifacts import sha256
from .webvid5 import fingerprint
from .webvid_ablation import snapshot
from . import bsu_predict as base
from . import sparc, sparc_semantics
from . import sparc_motion_transport as transport


def audit_reference(root):
    record=base.read(root/'protocol.json');deps={}
    if record['signature']!=fingerprint({k:v for k,v in record.items() if k!='signature'}):
        raise ValueError('reference protocol changed')
    for stage in sparc.STAGES:
        receipt=base.read(root/f'stages/{stage}.json')
        if (receipt['status']!='PASSED' or receipt['identity']!=record['signature'] or
                receipt['dependencies']!=deps or snapshot(root,receipt['required'])!=receipt['artifacts']):
            raise ValueError('reference stage changed: '+stage)
        deps[stage]=fingerprint(receipt)
    result=base.read(root/'RESULT.json')
    if (result['status']!=sparc.COMPLETE or result['protocol_sha256']!=sha256(root/'protocol.json') or
            result['movie_sha256']!=sha256(root/'reconstruction/reconstruction.mp4')):
        raise ValueError('reference completion changed')
    source=base.read(root/'source.json')
    if base.inspect_frames(Path(source['directory']),source['video']['fps'])!=source:
        raise ValueError('reference source changed')
    semantic=sparc_semantics.validate_manifest(root/'semantic.json',source,sparc.profile())
    selected=base.read(root/'selection.json')
    if selected['cuts']:raise ValueError('this pilot requires no detected scene cuts')
    transport.validate_timeline(source['video'],semantic['keys'])
    return source,semantic


def audit_legacy(source, keys, condition, prepared_path):
    """PNG bytes may differ; RGB pixels, timeline and every source hash must match."""
    import numpy as np
    from PIL import Image
    from .wire import unpack
    from .bsu_sequence_refinement import decode_packet
    prepared=base.read(prepared_path);job=base.read(condition/'job.json')
    if ({k:prepared[k] for k in ('frames','width','height','fps')}!=source['video'] or
            job['source_frames']!=prepared['source_frames']):
        raise ValueError('legacy trajectory source/timeline differs')
    receipt=base.read(condition/'stages/prepare.json')
    if receipt['status']!='PASSED' or snapshot(condition,receipt['required'])!=receipt['artifacts']:
        raise ValueError('legacy trajectory sender artifacts changed')
    old_packet=condition/'base/reconstruction/event_pairs/run/transmitter/metadata.bin'
    packet=condition/'sender/trajectory.bin'
    base_header,_=unpack(old_packet.read_bytes())
    if [row['index'] for row in base_header['keyframes']]!=keys or base_header['video']!=source['video']:
        raise ValueError('legacy trajectory endpoints differ')
    decoded=decode_packet(packet.read_bytes(),old_packet.read_bytes())
    raw,_=unpack(packet.read_bytes())
    matched_bytes=0
    for i in range(source['video']['frames']):
        old=prepared_path.parent/'source_frames'/f'{i}.png'
        new=Path(source['directory'])/f'{i}.png'
        if sha256(old)!=prepared['source_frame_hashes'][str(i)]:raise ValueError('legacy source hash changed')
        if sha256(new)!=source['hashes'][new.name]:raise ValueError('current source hash changed')
        with Image.open(old) as a,Image.open(new) as b:
            if not np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB'))):
                raise ValueError('legacy trajectory source pixels differ')
        matched_bytes+=sha256(old)==sha256(new)
    evidence=[packet,old_packet,prepared_path,condition/'job.json',condition/'stages/prepare.json']
    return raw['windows'],dict(reused_windows=len(decoded),new_tracking_calls=0,
        rgb_identical_frames=source['video']['frames'],png_byte_identical_frames=matched_bytes,
        source_timeline_and_key_indices_equal=True,
        provenance={str(p):sha256(p) for p in evidence},
        note='Reused compressed source tracks; optional thinning uses their decoded knots, not newly tracked pixels.')


def thin(windows, point_count, knot_count):
    import numpy as np
    from . import sparse_trajectory as sparse
    result=[]
    for window in windows:
        a,b=window['start'],window['end'];old=sparse.decode(window['trajectory'],b-a+1)
        selected=[]
        for reverse in (0,1):
            group=np.flatnonzero(old['origins']==reverse)
            chosen=sparse.choose_tracks(old['xy'][:,group],old['valid'][:,group],point_count//2,bool(reverse))
            selected.extend(group[chosen].tolist())
        if len(selected)<3:continue
        # Keep existing time knots to avoid fabricating new tracking evidence.
        ids=np.unique(np.round(np.linspace(0,len(old['times'])-1,min(knot_count,len(old['times'])))).astype(int))
        times=old['times'][ids]
        value=sparse.encode(old['knot_xy'][ids][:,selected],old['knot_valid'][ids][:,selected],
                            old['origins'][selected],times,b-a+1)
        result.append(dict(start=a,end=b,trajectory=value))
    return result


def choose_motion(keys, windows, config):
    from .packets import accounting
    cap=int(accounting(len(keys))['complex_channel_uses']*config['side_information_ratio_cap'])
    trials=[]
    for points,knots in config['trajectory_levels']:
        selected=thin(windows,points,knots)
        packet=transport.make_motion(keys,selected)
        uses=accounting(len(packet))['complex_channel_uses']
        trials.append(dict(points=points,knots=knots,windows=len(selected),bytes=len(packet),channel_uses=uses))
        if selected and uses<=cap:
            return packet,dict(cap=cap,selected=trials[-1],trials=trials,quality_used=False)
    raise ValueError('no useful trajectory representation fits declared side-information budget')


def auxiliary_schedule(keys, count):
    """Predeclared longest-gap midpoint control; never use quality scores."""
    result=list(keys)
    for _ in range(count):
        a,b=max(zip(result,result[1:]),key=lambda pair:(pair[1]-pair[0],-pair[0]))
        if b-a<=1:break
        result.append((a+b)//2);result.sort()
    return result
