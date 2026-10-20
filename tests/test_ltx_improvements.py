import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from semantic_transmission import ltx_conditioning as cond, ltx_refinement as post, ltx_receiver as ltx
from semantic_transmission import bsu_ltx_improvements as runner
from semantic_transmission.artifacts import sha256
from semantic_transmission.bsu_sequence_refinement import POLICY
from semantic_transmission.sparse_trajectory import encode, decode
from semantic_transmission.webvid5 import fingerprint
from semantic_transmission.wire import pack


def moving_scene(frames=17, size=(576, 320)):
    rng = np.random.default_rng(4)
    # Persistent textured translation, not a reconstruction-quality-selected clip.
    base = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 1.)
    return np.stack([np.roll(base, t*2, axis=1) for t in range(frames)])


def translated_tracks(frames=17):
    points = np.array([(x, y) for y in (20, 80, 140) for x in (20, 90, 160, 230)], dtype=np.float32)
    xy = np.stack([points + [t, 0] for t in range(frames)]).astype(np.float32)
    xy = np.concatenate([xy, xy], axis=1)
    valid = np.ones(xy.shape[:2], bool)
    origins = np.array([0]*len(points)+[1]*len(points))
    times = np.array(sorted(set([0, frames//4, frames//2, 3*frames//4, frames-1])))
    return encode(xy[times], valid[times], origins, times, frames)


@pytest.fixture
def received(tmp_path):
    frames = moving_scene(33)
    keys = tmp_path/'keys'; keys.mkdir()
    for i in (0, 16, 32):
        Image.fromarray(frames[i]).save(keys/f'{i}.png')
    packet = tmp_path/'metadata.bin'
    header = dict(video=dict(width=576, height=320, fps=24, frames=33),
                  keyframes=[dict(index=i) for i in (0, 16, 32)],
                  segments=[dict(text='A textured surface moves right.', flow=2.),
                            dict(text='The surface continues moving.', flow=2.)])
    packet.write_bytes(pack(header))
    trajectory = tmp_path/'trajectory.bin'
    trajectory.write_bytes(pack(dict(policy=POLICY, video=header['video'],
        base_metadata_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
        windows=[dict(start=a, end=b, trajectory=translated_tracks()) for a, b in ((0, 16), (16, 32))])))
    return packet, keys, trajectory


@pytest.mark.parametrize('text,expected', [
    (' A person  turns left.\nThen stops. ', 'A person turns left. Then stops.'),
    ('A person moves. motion score: 3.2.', 'A person moves. motion score: 3.2.'),
    ('The sign says motion score: 3.2.', 'The sign says motion score: 3.2.'),
    ('The sign says motion score: 3.2. A person reads it.', 'The sign says motion score: 3.2. A person reads it.'),
    ('He does not turn left.', 'He does not turn left.'),
])
def test_prompt_preserves_facts_negation_and_original_numeric_text(text, expected):
    assert cond.prompt(text, 99., True) == expected


def test_prompt_variant_omits_only_receiver_added_motion_score():
    assert cond.prompt('A person turns right.', 3.2, False).endswith('motion score: 3.2.')
    assert cond.prompt('A person turns right.', 3.2, True) == 'A person turns right.'


def test_finite_and_nonempty_caption_required():
    for text, flow in (('', 1), ('A person.', float('nan'))):
        with pytest.raises(ValueError): cond.prompt(text, flow, True)


def test_context_uses_received_middle_key_and_preserves_all_times(tmp_path, received):
    packet, keys, tracks = received
    value = cond.prepare(packet, keys, tmp_path/'context', 'context', tracks)
    assert len(value['windows']) == 1
    w = value['windows'][0]
    assert (w['start'], w['end'], w['generated_frames']) == (0, 32, 33)
    assert [i['frame_idx'] for i in w['images']] == [0, 16, 32]
    assert 'From 0.000 to 0.667 seconds:' in w['prompt']
    assert 'From 0.667 to 1.333 seconds:' in w['prompt']
    assert value['receiver_improvements']['additional_transmitted_bytes'] == 0


def test_scene_cut_disables_merge_and_trajectory_guides(tmp_path, received):
    packet, keys, tracks = received
    for k, color in ((0, 0), (16, 255), (32, 0)):
        Image.new('RGB', (576, 320), (color,)*3).save(keys/f'{k}.png')
    value = cond.prepare(packet, keys, tmp_path/'cut', 'all', tracks)
    assert len(value['windows']) == 2
    assert all(c['suspected_cut'] for c in value['receiver_improvements']['cuts'])
    assert not any(i.get('strength') for w in value['windows'] for i in w['images'])


def test_trajectory_guides_are_actual_pipeline_conditions_with_bounded_strength(tmp_path, received):
    packet, keys, tracks = received
    value = cond.prepare(packet, keys, tmp_path/'trajectory', 'trajectory', tracks)
    guides = [i for w in value['windows'] for i in w['images'] if 'strength' in i]
    assert guides
    assert all(0 < i['strength'] <= cond.policy()['guide_max_strength'] for i in guides)
    assert all(i['path'] in value['inputs'] for i in guides)
    assert all(0 < i['frame_idx'] < 16 for i in guides)
    again = cond.prepare(packet, keys, tmp_path/'trajectory', 'trajectory', tracks)
    assert again == value


def test_invisible_wrong_or_missing_trajectory_is_not_invented(tmp_path, received):
    packet, keys, tracks = received
    value = cond.prepare(packet, keys, tmp_path/'missing', 'all')
    assert not any(i.get('strength') for w in value['windows'] for i in w['images'])
    tr = decode(translated_tracks())
    tr['valid'][:] = False
    pixels, audit = cond.guide([moving_scene()[0], moving_scene()[-1]], tr, 8, cond.policy())
    assert pixels is None and not audit['accepted']
    bad = tmp_path/'bad.bin'; bad.write_bytes(tracks.read_bytes())
    packet.write_bytes(pack(dict(video=dict(frames=2), keyframes=[dict(index=0), dict(index=1)])))
    with pytest.raises((ValueError, KeyError)):
        cond.prepare(packet, keys, tmp_path/'bad', 'all', bad)


def test_bad_photometric_correspondence_is_rejected():
    tr = decode(translated_tracks())
    images = [np.zeros((320, 576, 3), np.uint8), np.full((320, 576, 3), 255, np.uint8)]
    pixels, audit = cond.guide(images, tr, 8, cond.policy())
    assert pixels is None and not audit['accepted']


def test_hashes_reject_changed_received_and_derived_images(tmp_path, received):
    packet, keys, tracks = received
    value = cond.prepare(packet, keys, tmp_path/'hash', 'all', tracks)
    guide = next(i for w in value['windows'] for i in w['images'] if 'strength' in i)
    Path(guide['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='received condition changed'):
        cond.validate(value)


def test_resigned_invalid_timeline_is_rejected(tmp_path, received):
    value = cond.prepare(*received[:2], tmp_path/'bad_time', 'context')
    value['windows'][0]['images'][1]['frame_idx'] = 999
    value['signature'] = fingerprint({k: v for k, v in value.items() if k != 'signature'})
    with pytest.raises(ValueError, match='image condition'):
        cond.validate(value)


def test_global_policy_not_mutated_or_tuned_per_clip(tmp_path, received):
    before = sha256(cond.CONFIG)
    for variant in cond.VARIANTS:
        value = cond.prepare(*received[:2], tmp_path/variant, variant, received[2])
        assert value['receiver_improvements']['policy'] == cond.policy()
    assert sha256(cond.CONFIG) == before


def test_postprocessing_is_identity_when_received_keys_match():
    frames = moving_scene()
    out, audit = post.refine_segment(frames, frames[[0, -1]], 24)
    assert np.array_equal(out, frames)
    assert audit['reason'] == 'keys already exact'


def test_postprocessing_preserves_exact_keys_and_bounded_motion():
    frames = moving_scene()
    keys = frames[[0, -1]].copy()
    keys = np.clip(keys.astype(int)+15, 0, 255).astype(np.uint8)
    out, audit = post.refine_segment(frames, keys, 24)
    assert np.array_equal(out[[0, -1]], keys)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert audit['blend'] in cond.policy()['postprocess_blends']
    if audit['blend']:
        assert audit['attempts'][-1]['accepted']
        assert audit['attempts'][-1]['new_freeze_fraction'] <= .02


def test_motion_gate_rejects_freeze_and_hallucinated_motion():
    before = np.ones((8, 16, 16, 2), np.float32)
    assert not post.motion_gate(before, np.zeros_like(before), cond.policy())[0]
    assert not post.motion_gate(before, before*3, cond.policy())[0]
    assert post.motion_gate(before, before, cond.policy())[0]


def test_postprocessing_cut_does_not_blend_across_scene():
    frames = moving_scene(9)
    keys = [np.zeros_like(frames[0]), np.full_like(frames[0], 255)]
    out, audit = post.refine_segment(frames, keys, 24)
    assert audit['reason'] == 'suspected cut'
    assert np.array_equal(out[1:-1], frames[1:-1])


class FakeGenerator:
    calls = []
    fail = False
    def __init__(self, assets): pass
    def __call__(self, window, value):
        self.calls.append(window['start'])
        for i in range(window['generated_frames']):
            if self.fail and i == 2: raise RuntimeError('interrupted')
            yield np.full((320, 576, 3), (window['start']+i)%255, dtype=np.uint8)


def test_group_generation_full_timeline_resume_and_corruption(tmp_path, received):
    value = cond.prepare(*received[:2], tmp_path/'prepared', 'all', received[2])
    output = tmp_path/'generated'
    FakeGenerator.calls = []; FakeGenerator.fail = True
    with pytest.raises(RuntimeError): ltx.reconstruct(value, {}, output, FakeGenerator)
    FakeGenerator.fail = False
    ltx.reconstruct(value, {}, output, FakeGenerator)
    count = len(FakeGenerator.calls)
    ltx.reconstruct(value, {}, output, FakeGenerator)
    assert len(FakeGenerator.calls) == count
    assert len(runner.verified_raw(output, value)) == 33
    Image.new('RGB', (576, 320)).save(output/'frames/00010.png')
    with pytest.raises(ValueError, match='corrupted'):
        runner.verified_raw(output, value)


def test_generator_passes_per_image_strength_frame_position_and_seed(tmp_path, monkeypatch):
    import torch
    from typing import NamedTuple
    class Input(NamedTuple):
        path: str
        frame_idx: int
        strength: float
        crf: int
    stub = types.ModuleType('ltx_pipelines.utils.types')
    stub.ImageConditioningInput = Input
    monkeypatch.setitem(sys.modules, 'ltx_pipelines.utils.types', stub)
    captured = {}
    def pipeline(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(num_frames=9, video=[torch.zeros(9, 320, 576, 3)])
    generator = object.__new__(ltx.Generator)
    generator.torch, generator.pipeline = torch, pipeline
    window = dict(prompt='Moves.', seed=17, generated_frames=9,
        images=[dict(path='key.png', frame_idx=0), dict(path='guide.png', frame_idx=4, strength=.2, crf=0)])
    assert len(list(generator(window, dict(profile=ltx.profile(), video=dict(height=320, width=576, fps=24))))) == 9
    assert captured['seed'] == 17
    assert captured['images'][0].strength == 1.
    assert captured['images'][1] == Input('guide.png', 4, .2, 0)


def test_postprocess_raw_replay_requires_identical_generation_conditions(tmp_path, received):
    value = cond.prepare(*received[:2], tmp_path/'base', 'baseline')
    FakeGenerator.fail = False
    ltx.reconstruct(value, {}, tmp_path/'raw', FakeGenerator)
    changed = copy.deepcopy(value); changed['windows'][0]['prompt'] = 'Different.'
    with pytest.raises(ValueError, match='does not match'):
        runner.verified_raw(tmp_path/'raw', changed)


def test_no_source_or_label_argument_in_receiver_cli():
    import subprocess
    result = subprocess.run([sys.executable, '-m', runner.MODULE, '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--source' not in result.stdout and '--label' not in result.stdout


def test_diagnostic_prefix_preserves_paid_packet_and_three_keys(tmp_path, received):
    from semantic_transmission.ltx_improvement_validation import prefix
    value = cond.prepare(*received[:2], tmp_path/'prefix', 'all', received[2])
    result = prefix(value, 32)
    assert result['receiver_improvements']['source_inputs'] == value['receiver_improvements']['source_inputs']
    assert result['receiver_improvements']['original_key_indices'] == [0, 16, 32]
    assert result['receiver_improvements']['diagnostic_prefix']['full_video_frames'] == 33
    with pytest.raises(ValueError, match='complete generation window'):
        prefix(value, 12)


def test_replay_is_independent_of_unrelated_original_pixels(tmp_path, received):
    source = tmp_path/'original.png'
    Image.new('RGB', (576, 320), (255, 0, 0)).save(source)
    first = cond.prepare(*received[:2], tmp_path/'source_independence', 'all', received[2])
    Image.new('RGB', (576, 320), (0, 255, 0)).save(source)
    second = cond.prepare(*received[:2], tmp_path/'source_independence', 'all', received[2])
    assert first == second
    assert str(source) not in first['inputs']


def test_full_runner_reuses_verified_raw_and_can_resume(tmp_path, received):
    from semantic_transmission.ltx_improvement_validation import receiver_args
    base = cond.prepare(*received[:2], tmp_path/'base_request', 'baseline', received[2])
    FakeGenerator.fail = False
    ltx.reconstruct(base, {}, tmp_path/'raw', FakeGenerator)
    row = dict(packet=str(received[0]), keys=str(received[1]), trajectory=str(received[2]))
    args = receiver_args(row, tmp_path/'result', 'postprocess', 'run', Path(sys.executable), tmp_path/'raw')
    runner.run(args)
    final = args.output/'refined/reconstruction.mp4'
    digest = sha256(final)
    runner.run(args)
    assert sha256(final) == digest
    result = json.loads((args.output/'RESULT.json').read_text())
    assert result['generation_reused'] and not result['source_pixels_read']
    for key in (0, 16, 32):
        with Image.open(args.output/f'refined/frames/{key:05d}.png') as actual, Image.open(received[1]/f'{key}.png') as expected:
            assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_transported_raw_keeps_original_request_signature(tmp_path, received):
    value = cond.prepare(*received[:2], tmp_path/'transport', 'baseline')
    FakeGenerator.fail = False
    ltx.reconstruct(value, {}, tmp_path/'raw', FakeGenerator)
    protocol_path = tmp_path/'raw/protocol.json'
    protocol = json.loads(protocol_path.read_text()); protocol['request']['windows'][0]['prompt'] = 'Altered'
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='request signature changed'):
        runner.verified_raw(tmp_path/'raw', value)


def test_receiver_refuses_output_inside_received_key_directory(tmp_path, received):
    from semantic_transmission.ltx_improvement_validation import receiver_args
    row = dict(packet=str(received[0]), keys=str(received[1]), trajectory=None)
    args = receiver_args(row, received[1]/'generated', 'all', 'prepare', Path(sys.executable))
    with pytest.raises(ValueError, match='overlaps protected input'):
        runner.run(args)


@pytest.mark.parametrize('indices', [(0, 1), (0, 2, 3), (0, 16, 32), (0, 720, 1440)])
def test_short_and_long_received_timelines_keep_every_frame_and_key(tmp_path, indices):
    key_dir = tmp_path/'keys';key_dir.mkdir()
    for index in indices:Image.new('RGB', (576, 320), (50, 60, 70)).save(key_dir/f'{index}.png')
    packet = tmp_path/'packet.bin'
    packet.write_bytes(pack(dict(video=dict(width=576, height=320, fps=24, frames=indices[-1]+1),
        keyframes=[dict(index=i) for i in indices],
        segments=[dict(text='The scene remains still.', flow=0.) for _ in indices[1:]])))
    value = cond.prepare(packet, key_dir, tmp_path/'request', 'all')
    coverage = []
    for n, w in enumerate(value['windows']):
        stop = w['end']+1 if n == len(value['windows'])-1 else w['end']
        coverage.extend(range(w['start'], stop))
        assert (w['generated_frames']-1) % 8 == 0
    assert coverage == list(range(indices[-1]+1))
    assert value['receiver_improvements']['original_key_indices'] == list(indices)


@pytest.mark.parametrize('fps,noise', [(12, 5), (24, 20), (60, 40)])
def test_refinement_policy_is_shared_across_frame_rates_and_received_noise(fps, noise):
    frames = moving_scene(9)
    rng = np.random.default_rng(91)
    keys = np.clip(frames[[0, -1]].astype(float)+rng.normal(0, noise, frames[[0, -1]].shape), 0, 255).astype(np.uint8)
    original_policy = cond.policy()
    result, audit = post.refine_segment(frames, keys, fps)
    assert result.dtype == np.uint8 and result.shape == frames.shape
    assert np.array_equal(result[[0, -1]], keys)
    assert cond.policy() == original_policy
    assert audit['blend'] in original_policy['postprocess_blends']
