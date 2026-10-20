"""Official NTSCC+ weights with a source-independent, serializable JSCC boundary.

Checkerboard groups are the upstream batch of two (anchor, non-anchor), not
video keyframes. Hyperlatents only determine sender rate allocation; the receiver
needs the transmitted indices, power and symbols, never encoder context tensors.
"""
import math
import os
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from .artifacts import sha256

NAME = 'NTSCC+'
SCHEMA = 'bsu_ntscc_plus_inputs_v1'
CHECKPOINT = '.local/checkpoints/ntscc_plus_ckbd2_lmbd_0.18.pth.tar'
CHECKPOINT_SHA256 = '85181c01a7faab33f4c08fb6a0946032ad68bf581c8fe954f4484914dd2e36e8'
UPSTREAM_REVISION = '765c6f6c11cf3bbe249a26b624dbaf47a4f0676d'
RATES = (1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128,
         144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320)
SPEC = dict(name=NAME, checkpoint='ckbd2_lmbd_0.18', eta=0.2,
            rate_index_encoding='uint8', checkerboard_order='anchor_then_non_anchor',
            padding='center_zero_height128_width64', real_value_padding='one_zero_if_odd',
            rate_one_mask='zero_as_released',
            upstream_revision=UPSTREAM_REVISION)
VERSIONS = {'torch': '2.0.0+cu118', 'torchvision': '0.15.1+cu118',
            'natten': '0.14.6+torch200cu118', 'compressai': '1.2.4', 'timm': '0.6.13'}


def installation(repo):
    interpreter = repo / '.local/ntscc_plus_env/bin/python'
    code = ('import json, importlib.metadata as m; '
            f'print(json.dumps({{k:m.version(k) for k in {list(VERSIONS)!r}}}))')
    versions = json.loads(subprocess.check_output([str(interpreter), '-c', code], text=True,
                                                  env=runtime_environment(interpreter)))
    if versions != VERSIONS:
        raise ValueError(f'NTSCC+ runtime differs from validated versions: {versions}')
    library = Path(runtime_environment(interpreter)['LD_LIBRARY_PATH']) / 'libnvrtc.so'
    if not library.is_file():
        raise FileNotFoundError(f'run scripts/setup_ntscc_plus.sh: {library}')
    return dict(versions=versions, spec=SPEC, checkpoint_sha256=CHECKPOINT_SHA256)


def runtime_environment(interpreter, environment=None):
    env = dict(os.environ if environment is None else environment)
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    # Torch 2.0's CUDA wheel bundles a hashed NVRTC library; setup creates the
    # dlopen alias needed by cuDNN on Ada. Only this isolated worker uses it.
    env['LD_LIBRARY_PATH'] = str(Path(interpreter).absolute().parent.parent /
                                 'lib/python3.10/site-packages/torch/lib')
    return env


def padding(height, width):
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) < 1:
        raise ValueError('invalid NTSCC+ image dimensions')
    dh, dw = (-height) % 128, (-width) % 64
    return dw // 2, dw - dw // 2, dh // 2, dh - dh // 2


def pack_rates(indices):
    values = list(indices)
    if not values or any(type(v) is not int or not 0 <= v < len(RATES) for v in values):
        raise ValueError('invalid NTSCC+ rate indices')
    return bytes(values)


def unpack_rates(data, count):
    if count < 1 or len(data) != count:
        raise ValueError('NTSCC+ rate-index payload length mismatch')
    values = list(data)
    pack_rates(values)
    return values


def load_codec(repo):
    import importlib.metadata
    import torch
    if importlib.metadata.version('natten').split('+')[0] != '0.14.6':
        raise ValueError('NTSCC+ requires checkpoint-compatible NATTEN 0.14.6')
    path = repo / CHECKPOINT
    if sha256(path) != CHECKPOINT_SHA256:
        raise ValueError('NTSCC+ checkpoint hash mismatch')
    vendor = repo / '.local/vendor/NTSCC_plus'
    # Both upstream projects use top-level "net" and "channel" packages.
    if 'net' in sys.modules and str(vendor / 'net') not in list(sys.modules['net'].__path__):
        raise RuntimeError('load NTSCC and NTSCC+ in separate worker processes')
    sys.path.insert(0, str(vendor))
    from net.ntscc import NTSCC_plus
    config = SimpleNamespace(channel_type='awgn', SNR=10, device=torch.device('cuda'),
                             eta=SPEC['eta'], multiple_rate=list(RATES), logger=None)
    model = NTSCC_plus(config)
    state = torch.load(path, map_location='cpu')
    state = state.get('state_dict', state)
    generated = lambda k: 'attn_mask' in k or k == 'rate_adaption_enc.mask'
    result = model.load_state_dict({k: v for k, v in state.items() if not generated(k)}, strict=False)
    if result.unexpected_keys or any(not generated(k) for k in result.missing_keys):
        raise ValueError(f'NTSCC+ learned-weight mismatch: {result}')
    if model.rate_choice_tensor.tolist() != list(RATES):
        raise ValueError('NTSCC+ checkpoint rate table changed')
    model.transport_codec = NAME
    return model.cuda().eval(), path


def rate_mask(model, indices, height, width):
    import torch
    left, right, top, bottom = padding(height, width)
    h, w = (height + top + bottom) // 32, (width + left + right) // 16
    if indices.ndim != 1 or indices.numel() != 2 * h * w or bool(
            ((indices < 0) | (indices >= len(RATES))).any()):
        raise ValueError('invalid received NTSCC+ rate indices or dimensions')
    rates = model.rate_choice_tensor[indices].reshape(2, h, w)
    # The released encoder updates its mask in place: for rate 1, the second
    # comparison also clears the first entry. Preserve that behavior exactly.
    rates = torch.where(rates == 1, 0, rates)
    return torch.arange(max(RATES), device=indices.device)[None, :, None, None] < rates[:, None]


def encode_symbols(model, tensor):
    import torch
    from torch.nn.functional import pad
    if tensor.shape[0:2] != (1, 3) or not torch.isfinite(tensor).all():
        raise ValueError('NTSCC+ expects one finite RGB image')
    height, width = tensor.shape[-2:]
    # Preserve the official encoder's seeded stochastic hyperprior/likelihoods.
    symbols, indices = model.encode(pad(tensor, padding(height, width)))
    mask = rate_mask(model, indices, height, width)
    values = symbols.masked_select(mask)
    # Match the upstream reduction order, including masked zeros. Small FP32
    # power changes can cross TF32 rounding boundaries in the synthesis network.
    power = symbols.square().sum() / mask.sum()
    if not torch.isfinite(values).all() or not math.isfinite(float(power)) or float(power) <= 0:
        raise ValueError('invalid NTSCC+ channel input')
    normalized = values / (2 * power).sqrt()
    # Rate 1 can yield an odd total. Pay for the final complex symbol, discard
    # only its unused imaginary component at the receiver.
    if normalized.numel() % 2:
        normalized = pad(normalized, (0, 1))
    return torch.complex(normalized[::2], normalized[1::2]), indices, float(power)


def decode_symbols(model, values, indices, power, height, width):
    import torch
    mask = rate_mask(model, indices, height, width)
    count = int(mask.sum())
    if (values.numel() != (count + 1) // 2 or not torch.isfinite(values).all()
            or not math.isfinite(power) or power <= 0):
        raise ValueError('invalid NTSCC+ received symbols or normalization')
    real = torch.stack((values.real, values.imag), -1).flatten()[:count]
    received = torch.zeros(mask.shape, device=values.device, dtype=torch.float32)
    scale = torch.tensor(2 * power, dtype=torch.float32, device=values.device).sqrt()
    received[mask] = real * scale
    restored = model.decode(received, indices)
    left, _, top, _ = padding(height, width)
    restored = restored[:, :, top:top + height, left:left + width].clamp(0, 1)
    if not torch.isfinite(restored).all():
        raise ValueError('nonfinite NTSCC+ reconstruction')
    return restored
