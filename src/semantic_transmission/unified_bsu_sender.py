"""Sender-only token analysis and actual-budget candidate screening."""
import gc
from pathlib import Path

import numpy as np

from . import unified_bsu_core as core
from .artifacts import sha256, write_json


def enhance_candidates(j, images, records):
    import torch
    from . import unified_bsu as u
    from .dense_generation_runtime import guide_latents
    from .unified_bsu_tokens import InnovationCodec, expanded_roi, importance
    config = j['profile']['innovation']
    c = torch.load(j['checkpoint']['path'], map_location='cpu', weights_only=True)
    if not c.get('innovation_state') or c.get('innovation_config') != config:
        raise ValueError('sender needs a matching jointly trained innovation checkpoint')
    codec = InnovationCodec(config['tokens'], config['width']).cuda().eval()
    codec.load_state_dict(c['innovation_state'], strict=True)
    modes, build = u.registry(); vae = build(u.recipe(j['runtime']['models']).vae, modes)
    vae = vae.cuda().to(torch.bfloat16).eval()
    root = Path(j['output']); out = root / 'sender'; p = j['profile']
    ws = core.windows(len(images), 17, p['window_overlap'])
    extra = []; audits = []
    with torch.no_grad():
        targets = [guide_latents(vae, u.padded_tensor(images[a:b])).float().cpu() for a, b in ws]
        for i, row in enumerate(records):
            if not row['fits']:
                continue
            u.event('송신 학습 토큰·실제 비트 검사', i, len(records), row['name'])
            base = u.load_rgb(out / (row['name']+'_base'))
            priors = [guide_latents(vae, u.padded_tensor(base[a:b])).float().cpu() for a, b in ws]
            masks = [importance(x, y) for x, y in zip(targets, priors)]
            def score(values):
                rows = []
                for target, prior, mask in zip(targets, values, masks):
                    mse = ((target-prior).square()*(1+3*expanded_roi(mask, prior))).mean()
                    temporal = torch.diff(target-prior, dim=2).abs().mean()
                    rows.append(float(mse+.1*temporal))
                return float(np.mean(rows))
            row['rgb_screening'] = row['distortion']
            row['distortion'] = dict(score=score(priors), objective='received_latent_roi_and_temporal')
            _, stream = core.decode_packet((out / f'{row["name"]}.bin').read_bytes())
            for step in config['steps']:
                chunks, restored = [], []
                for x, y in zip(targets, priors):
                    blob = codec.compress(x.cuda(), y.cuda(), step)
                    chunks.append(blob); restored.append(codec.decompress(blob, y.cuda())[0].cpu())
                data = core.encode_packet(j['context'], stream, row['coding'], j['decoder'], chunks)
                name = f'{row["name"]}_tokens{step}'
                path = out / f'{name}.bin'; path.write_bytes(data)
                uses = core.channel_uses(data)
                item = dict(name=name, coding=row['coding'], channel_uses=uses, packet_bytes=len(data),
                    packet_sha256=sha256(path), fits=uses <= j['budget'],
                    distortion=dict(score=score(restored), objective='received_latent_roi_and_temporal'),
                    innovation_step=step, breakdown=core.packet_breakdown(data))
                extra.append(item)
                audits.append(dict(name=name, bits=sum(len(x)*8 for x in chunks), actual_roundtrip=True,
                                   source_used_only_at_sender=True))
    del vae, codec; gc.collect(); torch.cuda.empty_cache()
    write_json(out / 'innovation_audit.json', dict(rows=audits, importance='coarse_motion_and_error_heuristic',
        semantic_hand_detector=False, final_generated_quality_used=False,
        joint_training_objective='flow_conditioning_plus_roi_distortion_temporal_and_rate'))
    return records+extra
