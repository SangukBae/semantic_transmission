"""Exact, artifact-backed budget boundary for the opt-in allocation predictor.

Each entry in digital_packets is an independently coded packet. Metadata fields
inside the same packet must NOT be rounded to LDPC blocks independently. The
caller must enumerate all auxiliary packets; unmodelled physical-link overhead
is outside this repository's channel-use budget.
"""
from itertools import product
import math
from pathlib import Path

from .artifacts import sha256
from .packets import accounting
from .wire import pack, unpack

SCHEMA = 'bsu_joint_allocation_inputs_v1'
CHANNEL = dict(k=6144, n=9216, bits_per_symbol=4)


def receipt(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path))


def verified_path(entry):
    path = Path(entry['path'])
    if not path.is_absolute() or sha256(path) != entry['sha256']:
        raise ValueError(f'artifact hash/path mismatch: {path}')
    return path


def validate_candidate(candidate, frames, minimum_spacing=1):
    keys, etas = candidate['keys'], candidate['etas']
    if (type(frames) is not int or frames < 2 or type(minimum_spacing) is not int
            or minimum_spacing < 1 or not isinstance(keys, list) or len(keys) < 2
            or any(type(k) is not int for k in keys) or keys[0] != 0 or keys[-1] != frames-1
            or any(b-a < minimum_spacing for a, b in zip(keys, keys[1:]))):
        raise ValueError('invalid key positions/endpoints/spacing')
    if len(etas) != len(keys) or any(type(e) not in (float, int) or not math.isfinite(e)
                                   or not .005 <= e <= 1. for e in etas):
        raise ValueError('invalid per-key eta; expected one value in [0.005,1] per key')


def candidate_grid(key_sets, eta_values, frames, max_candidates=256, minimum_spacing=1):
    """Explicit finite search space; the network learns which joint plan to use.

    Reject an oversized space rather than silently privileging early candidates.
    Production callers can instead provide a bounded proposal list of their own.
    """
    if not key_sets or not eta_values or len(set(eta_values)) != len(eta_values):
        raise ValueError('empty/duplicate candidate grid')
    if len({tuple(k) for k in key_sets}) != len(key_sets):
        raise ValueError('duplicate key sets')
    count = sum(len(eta_values)**len(keys) for keys in key_sets)
    if count > max_candidates:
        raise ValueError(f'candidate grid {count} exceeds limit {max_candidates}')
    result = []
    for keys in key_sets:
        for rates in product(eta_values, repeat=len(keys)):
            row = dict(id=f'candidate_{len(result):04d}', keys=list(keys), etas=list(rates))
            validate_candidate(row, frames, minimum_spacing)
            result.append(row)
    return result


def bundle(visual, metadata, auxiliary_packets=(), channel=None):
    paths = [Path(metadata).resolve(), *[Path(p).resolve() for p in auxiliary_packets]]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate digital packet')
    return dict(visual=receipt(visual), digital_packets=[receipt(p) for p in paths],
                channel=dict(CHANNEL if channel is None else channel), complete=True)


def wire_cost(value, candidate=None, frames=None):
    if value.get('complete') is not True or not value.get('digital_packets'):
        raise ValueError('complete transport bundle with metadata is required')
    channel = value['channel']
    if set(channel) != set(CHANNEL) or any(type(v) is not int for v in channel.values()):
        raise ValueError('invalid channel accounting profile')
    visual = verified_path(value['visual'])
    size = visual.stat().st_size
    if not size or size % 8:
        raise ValueError('invalid complex64 visual stream')
    paths = [verified_path(p) for p in value['digital_packets']]
    if len(paths) != len(set(paths)) or visual in paths:
        raise ValueError('duplicate transport artifact')
    packet = paths[0].read_bytes()
    if packet[:4] == b'BSMC':
        from .bsu_metadata_compression import decode
        packet = decode(packet)
    header, _ = unpack(packet)
    offset = 0
    for key in header['keyframes']:
        if (type(key['complex_count']) is not int or key['complex_count'] <= 0
                or key['complex_offset'] != offset):
            raise ValueError('invalid/noncontiguous visual descriptors')
        offset += key['complex_count']
    if offset != size // 8:
        raise ValueError('visual count differs from actual stream')
    if candidate is not None:
        validate_candidate(candidate, frames)
        if header['video']['frames'] != frames or [k['index'] for k in header['keyframes']] != candidate['keys']:
            raise ValueError('candidate keys differ from encoded packet')
        rates = [k.get('allocation_eta', header['visual_codec']['eta']) for k in header['keyframes']]
        if rates != candidate['etas']:
            raise ValueError('candidate rates differ from encoded packet')
    digital = [accounting(p.stat().st_size, **channel) for p in paths]
    uses = sum(p['complex_channel_uses'] for p in digital)
    return dict(visual_uses=size//8, digital_uses=uses, total_uses=size//8+uses,
                packets=digital, physical_link_overhead_included=False)


def select_verified(candidates, scores, budget, frames, minimum_spacing=1):
    """Recheck hashes and actual wire cost immediately before returning a plan."""
    if type(budget) is not int or budget <= 0 or len(scores) != len(candidates):
        raise ValueError('invalid budget or scores')
    if not candidates or any(not math.isfinite(s) for s in scores):
        raise ValueError('empty/nonfinite candidate scores')
    ledgers = []
    for c in candidates:
        validate_candidate(c, frames, minimum_spacing)
        ledgers.append(wire_cost(c['bundle'], c, frames))
    valid = [i for i, cost in enumerate(ledgers) if cost['total_uses'] <= budget]
    if not valid:
        raise ValueError('INFEASIBLE_REGISTERED_BUDGET: no feasible candidate')
    index = max(valid, key=lambda i: (scores[i], -i))
    return dict(candidate_id=candidates[index]['id'], index=index, budget=budget,
                ledger=ledgers[index], keys=candidates[index]['keys'], etas=candidates[index]['etas'],
                bundle=candidates[index]['bundle'], predicted_score=scores[index],
                measured_quality_used_for_selection=False)


def encode_candidate(model, row, candidate, source, output, *, seed=2025, device='cuda'):
    """Optional NTSCC+ worker API, including unequal eta within one key set.

    Returns a BASE bundle. If trajectories/other packets will be sent, append
    them with bundle(...) before selection. The receiver uses actual transmitted
    rate indices; allocation_eta is audited sender metadata, not a decode input.
    """
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from . import ntscc_plus_codec as codec
    from .bsu_budget_pilot import remap_segments
    from .bsu_ltx_pipeline import decoder_fields, MULTIKEY
    validate_candidate(candidate, row['video']['frames'])
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    header = dict(schema=SCHEMA, video=row['video'], keyframes=[],
        checkpoint_sha256=codec.CHECKPOINT_SHA256, visual_codec=dict(codec.SPEC),
        decoder=dict(decoder_fields(dict(generation_conditioning=MULTIKEY)), seed=2025,
                     steps=11, policy='endpoint_exact', concatenation_policy='endpoint_exact'),
        segments=remap_segments(row['keys'], row['segments'], candidate['keys']),
        complex_dtype='little_endian_complex64')
    payload = bytearray()
    cursor = 0
    previous_eta = model.eta
    try:
        devices = None if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), (output/'visual.c64').open('xb') as stream:
            for index, eta in zip(candidate['keys'], candidate['etas']):
                model.eta = eta
                # Same key/eta gets the same encoder noise across joint plans.
                torch.manual_seed(seed + index)
                tensor = pil_to_tensor(Image.open(source/f'{index:05d}.png').convert('RGB'))
                values, rates, power = codec.encode_symbols(model, tensor.float().unsqueeze(0).to(device)/255)
                packed = codec.pack_rates(rates.cpu().tolist())
                header['keyframes'].append(dict(index=index, allocation_eta=eta,
                    complex_offset=cursor, complex_count=values.numel(), average_power=power,
                    rate_offset=len(payload), rate_bytes=len(packed), rate_count=rates.numel()))
                payload.extend(packed)
                cursor += values.numel()
                stream.write(values.cpu().numpy().astype('<c8').tobytes())
        (output/'metadata.bin').write_bytes(pack(header, payload))
    finally:
        model.eta = previous_eta
    return bundle(output/'visual.c64', output/'metadata.bin')


def decode_received(received, output, *, model=None, device='cuda'):
    """Received-files-only decode for SCHEMA; no sender feature/source access.

    This opt-in worker API leaves the frozen pilot receiver and default BSU
    receiver unchanged. Its key PNGs and metadata can feed the LTX conditioner.
    """
    import numpy as np
    import torch
    from torchvision.utils import save_image
    from . import ntscc_plus_codec as codec
    received, output = Path(received), Path(output)
    packet = (received/'metadata.bin').read_bytes()
    if packet[:4] == b'BSMC':
        from .bsu_metadata_compression import decode
        packet = decode(packet)
    header, payload = unpack(packet)
    if (header.get('schema') != SCHEMA or header.get('checkpoint_sha256') != codec.CHECKPOINT_SHA256
            or header.get('visual_codec') != codec.SPEC):
        raise ValueError('unknown joint-allocation codec profile')
    candidate = dict(keys=[k['index'] for k in header['keyframes']],
                     etas=[k['allocation_eta'] for k in header['keyframes']])
    validate_candidate(candidate, header['video']['frames'])
    values = np.fromfile(received/'visual.c64', dtype='<c8')
    if (received/'visual.c64').stat().st_size % 8 or not np.isfinite(values).all():
        raise ValueError('invalid received complex stream')
    cursor = rate_cursor = 0
    parsed = []
    for key in header['keyframes']:
        if key['complex_offset'] != cursor or key['rate_offset'] != rate_cursor or key['complex_count'] <= 0:
            raise ValueError('noncontiguous received stream')
        end = rate_cursor+key['rate_bytes']
        indices = codec.unpack_rates(payload[rate_cursor:end], key['rate_count'])
        parsed.append((key, indices))
        cursor += key['complex_count']
        rate_cursor = end
    if cursor != len(values) or rate_cursor != len(payload):
        raise ValueError('unaccounted received symbols/rates')
    model = model if model is not None else codec.load_codec(Path(__file__).resolve().parents[2])[0]
    output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        for key, indices in parsed:
            start, count = key['complex_offset'], key['complex_count']
            image = codec.decode_symbols(model, torch.from_numpy(values[start:start+count].copy()).to(device),
                torch.tensor(indices, dtype=torch.long, device=device), key['average_power'],
                header['video']['height'], header['video']['width'])
            save_image(image, output/f'{key["index"]}.png')
    return dict(source_pixels_read=False, keys=candidate['keys'],
                received_metadata_sha256=sha256(received/'metadata.bin'))
