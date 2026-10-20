"""LTX stages for fresh BSU-Gen transmissions, with received-only conditioning."""
import os

from . import ltx_receiver as ltx
from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .wire import unpack

BACKEND = 'ltx25'
PAIR = 'endpoint_pair_v1'
MULTIKEY = 'bounded_multikey_v1'


def conditioning(cfg=None):
    """Register the tested grouping rules without importing image/GPU libraries."""
    mode = (cfg or {}).get('generation_conditioning', PAIR)
    if mode == PAIR:
        return dict(mode=mode)
    if mode != MULTIKEY:
        raise ValueError('unknown LTX generation conditioning')
    expected = dict(context_intervals=2, context_max_seconds=4.0,
                    scene_cut_mad=45.0, scene_cut_histogram=.55)
    params = ltx.read(ltx.REPO / 'configs/bsu_ltx_receiver_v2.json')
    if any(params.get(k) != v for k, v in expected.items()):
        raise ValueError('registered LTX multikey policy changed')
    return dict(mode=mode, **expected)


def decoder_fields(cfg=None):
    p = ltx.profile()
    policy = conditioning(cfg)
    # Reuse the existing fixed-size identifier: grouping adds no wire fields.
    identity = p if policy['mode'] == PAIR else dict(generator=p, conditioning=policy)
    return dict(generator_backend=BACKEND, generator_profile_sha256=fingerprint(identity),
                sampling_steps=[len(p['stage_1_sigmas'])-1, len(p['stage_2_sigmas'])-1])


def enabled(cfg):
    return cfg.get('generator_backend') == BACKEND


def condition_plan(keys, cfg=None):
    plan = dict(generator_backend=BACKEND, shared_boundary=ltx.profile()['shared_boundary'],
        intervals=[dict(start=a, end=b, frames=b-a+1, generated_frames=ltx.padded_frames(b-a+1))
                   for a, b in zip(keys, keys[1:])])
    policy = conditioning(cfg)
    if policy['mode'] == MULTIKEY:
        plan.update(conditioning=policy, grouping='deferred_until_received_keys_available')
    return plan


def jobs(cfg=None):
    from . import qwen35_pairs_comparison as pc
    for stage, module, products in pc.reconstruction_jobs():
        if stage == 'prepare-text':
            products = ['receiver_policy.json', 'receiver/ltx_request.json']
            if conditioning(cfg)['mode'] == MULTIKEY:
                products.append('receiver/ltx_conditions')
            yield 'prepare-generator', pc.MODULE, products
        elif stage == 'reconstruct':
            yield stage, pc.MODULE, ['receiver/ltx_generation', 'receiver/reconstruction']
        else:
            yield stage, module, products


def owned_products(stage, required):
    # LTX owns its interval receipts. Preserve completed intervals when the
    # outer worker fails; its own Stages verifies their hashes on resume.
    return [p for p in required if not (stage == 'reconstruct' and p == 'run/receiver/ltx_generation')]


def received_request(run):
    cfg = ltx.read(run / 'run_config.json')
    packet = run / 'received/metadata.bin'
    header, _ = unpack(packet.read_bytes())
    decoder = header['decoder']
    fields = decoder_fields(cfg)
    expected = dict(fields, seed=ltx.profile()['generation_seed'], steps=sum(fields['sampling_steps']),
                    policy='endpoint_exact', concatenation_policy='endpoint_exact')
    if decoder != expected:
        raise ValueError('received LTX decoder profile differs from registered default')
    keys = run / 'receiver/frames/sample/key_frames_received'
    if conditioning(cfg)['mode'] == MULTIKEY:
        from . import ltx_conditioning
        request = ltx_conditioning.prepare(packet, keys, run / 'receiver/ltx_conditions',
                                          variant='context', seed=expected['seed'])
    else:
        request = ltx.request(packet, keys)
    request['packet_decoder_overridden_by_registered_profile'] = False
    request['signature'] = fingerprint({k:v for k,v in request.items() if k != 'signature'})
    return request


def assets(cfg):
    current = ltx.runtime(cfg['ltx_model_dir'], cfg['ltx_vendor'])
    if current != cfg['generator_runtime']:
        raise ValueError('frozen LTX runtime changed')
    return current


def prepare(run):
    from .qwen35_pairs_video import freeze
    cfg = ltx.read(run / 'run_config.json')
    assets(cfg)
    request = received_request(run)
    freeze(run / 'receiver/ltx_request.json', request)
    freeze(run / 'receiver_policy.json', dict(generator_backend=BACKEND,
        profile=ltx.profile(), received_input_signature=request['signature'],
        source_pixels_read=False, sampling_steps=decoder_fields(cfg)['sampling_steps'],
        conditioning=conditioning(cfg), generation_windows=len(request['windows'])))


def reconstruct(run, factory=None):
    cfg = ltx.read(run / 'run_config.json')
    request = ltx.read(run / 'receiver/ltx_request.json')
    if request != received_request(run):
        raise ValueError('received LTX request changed after preparation')
    runtime = assets(cfg)
    raw = run / 'receiver/ltx_generation'
    ltx.reconstruct(request, runtime, raw, **({'factory':factory} if factory else {}))
    # Publish the common evaluation/refinement layout without reencoding.
    output = run / 'receiver/reconstruction'; output.mkdir(parents=True, exist_ok=True)
    pairs = [(raw / 'reconstruction.mp4', output / 'sample_0000.mp4')]
    frames = output / 'sample_0000_frames'; frames.mkdir(exist_ok=True)
    pairs.extend((p, frames / p.name) for p in sorted((raw / 'frames').glob('*.png')))
    for source, target in pairs:
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError(f'published LTX artifact changed: {target}')
        else:
            os.link(source, target)


def audit(run):
    from PIL import Image
    from .etri_60s_check import pts_audit
    from .video_io import probe
    cfg = ltx.read(run / 'run_config.json')
    request = received_request(run)
    ltx.verify_request(request)
    generation = ltx.read(run / 'receiver/ltx_generation/generation.json')
    inputs = ltx.read(run / 'receiver/decoder_inputs.json')
    keys = ltx.read(run / 'keyframes.json')['indices']
    video = run / 'receiver/reconstruction/sample_0000.mp4'
    info = probe(video)
    if (generation['status'] != 'COMPLETE_LTX_GENERATION' or generation['source_frames_read']
            or generation['received_input_signature'] != request['signature']
            or generation['profile'] != ltx.profile() or generation['video_sha256'] != sha256(video)
            or any(info[k] != cfg[k] for k in ('frames', 'fps', 'width', 'height'))
            or inputs['indices'] != keys or keys[0] != 0 or keys[-1] != cfg['frames']-1):
        raise ValueError('LTX reconstruction provenance or timeline mismatch')
    frames = sorted((video.parent / 'sample_0000_frames').glob('*.png'))
    if [p.name for p in frames] != [f'{i:05d}.png' for i in range(cfg['frames'])]:
        raise ValueError('LTX lossless frames missing or duplicated')
    for path in frames:
        with Image.open(path) as image:
            image.load()
            if image.mode != 'RGB' or image.size != (cfg['width'], cfg['height']):
                raise ValueError(f'invalid LTX frame: {path}')
        if sha256(path) != sha256(run / 'receiver/ltx_generation/frames' / path.name):
            raise ValueError('LTX publication differs from generated frames')
    channel = ltx.read(run / 'channel_accounting.json')
    if channel['status'] != 'PASSED' or not channel['metadata_exact_match']:
        raise ValueError('LTX received metadata channel failed')
    if (sha256(cfg['input']) != cfg['input_sha256']
            or ltx.read(run / 'input_audit.json')['source_sha256'] != cfg['input_sha256']):
        raise ValueError('evaluation source changed')
    write_json(run / 'output_audit.json', dict(status='PASSED', generator_backend=BACKEND,
        video=str(video), video_sha256=sha256(video), time_axis=pts_audit(video, cfg['frames'], cfg['fps']),
        source_indices=list(range(cfg['frames'])), all_lossless_frames_decoded=True,
        received_input_signature=request['signature'], generator_source_pixels_read=False,
        conditioning=conditioning(cfg), generation_windows=len(request['windows']),
        shared_boundary=ltx.profile()['shared_boundary'], hallucination_review='PENDING'))
