"""Experimental adapter hook for the unchanged native BSU decoder schedule."""
from pathlib import Path
import numpy as np
from PIL import Image
from .artifacts import sha256,write_json
from .bsu_motion_validation import read


def checkpoint_record(path):
    """Validate the exported receiver module before freezing an execution."""
    import torch
    from .trajectory_adapter import POLICY, TrajectoryAdapter
    path = Path(path).resolve()
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('policy') != POLICY:
        raise ValueError('unsupported trajectory adapter policy')
    adapter = TrajectoryAdapter()
    adapter.load_state_dict(value['state_dict'], strict=True)
    if not all(torch.isfinite(v).all() for v in adapter.state_dict().values()):
        raise ValueError('nonfinite trajectory adapter weights')
    return dict(checkpoint=str(path), checkpoint_sha256=sha256(path),
                policy=POLICY, steps=value.get('steps'), enabled=True)


def native_condition(tracks,keys,frames,height,width,continuation):
    from .trajectory_adapter import condition_maps
    condition=condition_maps(tracks,keys,list(range(frames)),height,width)
    if continuation:
        condition=np.concatenate([np.zeros((10,5,height,width),np.float32),condition],axis=1)
    return condition


def install(run):
    import torch
    from opensora.schedulers.rf import RFLOW
    from .trajectory_adapter import TrajectoryAdapter
    from .bsu_sequence_refinement import decode_packet
    config=read(run/'adapter_runtime.json');keys=read(run/'keyframes.json')['indices']
    trajectory=run/'received/trajectory.bin'
    tracks=decode_packet(trajectory.read_bytes(),(run/'received/metadata.bin').read_bytes()) if trajectory.exists() else {}
    checkpoint=Path(config['checkpoint'])
    if sha256(checkpoint)!=config['checkpoint_sha256']:raise ValueError('adapter checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
    original=RFLOW.sample;records=[]

    def sample(scheduler,model,text_encoder,z,*args,**kwargs):
        i=len(records);a,b=keys[i:i+2];trajectory=tracks.get((a,b))
        rgb=[np.asarray(Image.open(run/f'receiver/frames/sample/key_frames_received/{k}.png').convert('RGB')) for k in (a,b)]
        condition=native_condition(trajectory,rgb,b-a+1,z.shape[-2],z.shape[-1],i>0)
        if condition.shape[1]!=z.shape[2]:raise ValueError('native adapter temporal alignment mismatch')
        adapter=TrajectoryAdapter(hidden=model.hidden_size).to(device=z.device,dtype=z.dtype).eval()
        adapter.load_state_dict(state);adapter.requires_grad_(False)
        adapter.condition=torch.from_numpy(condition)[None].to(device=z.device,dtype=z.dtype) if config['enabled'] else None
        handle=adapter.attach(model)
        try:result=original(scheduler,model,text_encoder,z,*args,**kwargs)
        finally:handle.remove()
        records.append(dict(segment=i,start=a,end=b,enabled=config['enabled'],trajectory_present=trajectory is not None,
            condition_shape=list(condition.shape),prefix_latents=5 if i else 0,receiver_reads_source=False))
        write_json(run/'receiver/adapter_trace.json',records)
        return result

    RFLOW.sample=sample
    return records
