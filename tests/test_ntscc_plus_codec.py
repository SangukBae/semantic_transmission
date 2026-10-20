from types import SimpleNamespace

import pytest
import torch

from semantic_transmission import ntscc_plus_codec as codec
from semantic_transmission.codec_transport import received_codec
from semantic_transmission.qwen35_pairs_video import stage_interpreter


def test_new_rate_alphabet_and_old_packet_compatibility():
    rates = list(range(26))
    assert codec.unpack_rates(codec.pack_rates(rates), 26) == rates
    assert received_codec({'schema': 'lgvsc_model_inputs_v1'})[0] == 'NTSCC'
    assert received_codec({'schema': codec.SCHEMA, 'visual_codec': codec.SPEC})[0] == 'NTSCC+'
    for values in ([], [26], [-1], [1.5], [True]):
        with pytest.raises(ValueError):
            codec.pack_rates(values)
    with pytest.raises(ValueError):
        codec.unpack_rates(b'\x01', 2)
    with pytest.raises(ValueError):
        received_codec({'schema': codec.SCHEMA, 'visual_codec': dict(codec.SPEC, eta=.4)})
    with pytest.raises(ValueError):
        received_codec({'schema': 'lgvsc_model_inputs_v1', 'visual_codec': codec.SPEC})


def test_checkerboard_transport_released_zero_mask_and_original_crop():
    model = SimpleNamespace(rate_choice_tensor=torch.tensor(codec.RATES))
    height, width = 127, 63
    indices = torch.ones(32, dtype=torch.long)
    indices[0] = 0  # Released rate-1 mask is zero; 31 * 4 = 124 real values.
    mask = codec.rate_mask(model, indices, height, width)
    assert mask.shape == (2, 320, 4, 4) and int(mask.sum()) == 124
    original = torch.arange(124, dtype=torch.float32) / 100
    values = torch.complex(original[::2], original[1::2])
    def decode(symbols, received_indices):
        torch.testing.assert_close(symbols[mask], original)
        assert not bool(symbols[~mask].any())
        assert torch.equal(indices, received_indices)
        return torch.ones(1, 3, 128, 64) * .5
    model.decode = decode
    restored = codec.decode_symbols(model, values, indices, .5, height, width)
    assert restored.shape == (1, 3, height, width)
    for bad in (values[:-1], values * float('nan')):
        with pytest.raises(ValueError):
            codec.decode_symbols(model, bad, indices, .5, height, width)
    with pytest.raises(ValueError):
        codec.decode_symbols(model, values, indices, -1., height, width)
    with pytest.raises(ValueError):
        codec.rate_mask(model, indices[:-1], height, width)
    assert codec.padding(320, 576) == (0, 0, 32, 32)


def test_codec_runtime_is_used_only_for_new_sender_and_receiver():
    protocol = dict(receiver=dict(visual_codec='NTSCC+'),
        interpreters=dict(python='generator', codec_python='plus', channel_python='sionna'))
    for stage in ('send', 'receive'):
        assert stage_interpreter(protocol, stage) == 'plus'
    assert stage_interpreter(protocol, 'channel') == 'sionna'
    assert stage_interpreter(protocol, 'reconstruct') == 'generator'
    protocol['receiver'].pop('visual_codec')
    assert stage_interpreter(protocol, 'send') == 'generator'
    protocol['receiver']['visual_codec'] = 'NTSCC+'
    del protocol['interpreters']['codec_python']
    with pytest.raises(KeyError):
        stage_interpreter(protocol, 'send')
