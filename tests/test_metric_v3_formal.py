import numpy as np
import pytest

from semantic_transmission.attribution_metric import support
from semantic_transmission.metric_v3_statistics import (classify_sta, classification_report, fit_classifier,
    paired, predict_classifier, quantile, rate, segment_score, truth_extra)
from semantic_transmission.sta_video_validation import encode_video, receive, representation, simulate


def video():
    return np.stack([np.full((192, 320, 3), 20 + t, np.uint8) for t in range(32)])


def test_packet_roundtrip_and_same_rgb_different_fault_stage():
    frames = video()
    intact, status, indices = receive(encode_video(frames))
    assert np.array_equal(intact, frames) and status == ['ok'] * 32 and indices == list(range(32))
    cases = {name: simulate(frames, name, .25) for name in ('tx_omission', 'packet_loss', 'bit_corruption', 'generation_freeze')}
    reconstructed = [x['reconstruction'] for x in cases.values()]
    assert all(np.array_equal(x, reconstructed[0]) for x in reconstructed)
    assert cases['tx_omission']['tx_status'].count('packet_lost') == 8
    assert cases['packet_loss']['tx_status'] == ['ok'] * 32
    assert cases['bit_corruption']['rx_status'].count('checksum_failed') == 8
    assert cases['generation_freeze']['rx_status'] == ['ok'] * 32
    # A source/reconstruction-only classifier cannot distinguish these four inputs.
    assert np.array_equal(cases['generation_freeze']['rx_frames'], frames)


def test_packet_decode_abstains_when_no_readable_image():
    with pytest.raises(ValueError, match='no readable'):
        receive([None] * 32)


def test_added_object_is_absent_from_transmitted_pixels():
    frames = video(); result = simulate(frames, 'unsupported_addition', .125)
    assert np.array_equal(result['tx_frames'], frames)
    assert np.array_equal(result['rx_frames'], frames)
    changed = np.any(result['reconstruction'] != frames, axis=-1)
    assert np.array_equal(changed, result['addition_masks'])
    assert np.count_nonzero(changed.any((1, 2))) == 4


def test_representation_has_one_temporal_tolerance_not_two():
    event = dict(kind='event', type='turn', time_s=1., direction=0, cell=[1, 1])
    obs = {'events': [event], 'tracks': []}
    rep = representation(obs, ['ok'] * 16, [], [])
    assert support(event, rep)
    assert support({**event, 'time_s': 1.25}, rep)
    assert not support({**event, 'time_s': 1.5}, rep)
    statuses = ['ok'] * 16; statuses[8] = 'checksum_failed'
    assert not support(event, representation(obs, statuses, [], []))


def test_abstentions_remain_misses_and_frozen_strict_threshold():
    rows = [{'source_id': 'a', 'scores': {'ere': v}} for v in (None, 0., .5, 1.)]
    result = rate(rows, 'ere', .5)
    assert result['coverage'] == .75
    assert result['detected'] == 1 and result['rate_abstentions_as_no_detection'] == .25
    assert quantile([0., 1., None]) == 1.
    assert segment_score([0., 2., 3., 0., 2.], 1.) == .4


def test_pairing_uses_sources_and_shared_measurability():
    pos = [{'source_id': s, 'scores': {'ere': 1., 'ssim': 0.}} for s in ('a', 'b')]
    neg = [{'source_id': s, 'scores': {'ere': 0., 'ssim': 1.}} for s in ('a', 'b')]
    report = paired(pos, neg, 'ere', 'ssim')
    assert report['paired_sources'] == 2
    assert report['mean_within_source_auc_difference'] == 0
    assert report['source_bootstrap_95ci'] == [0., 0.]
    assert not report['positive_lower_bound']


def test_sta_classifier_cannot_use_fault_label_at_prediction():
    rows = [{'source_id': str(i), 'label': name, 'scores': {'ere': i}} for i, name in enumerate(('edr', 'clr', 'gfr', 'ghr'))]
    fit = fit_classifier(rows, ['ere'])
    assert predict_classifier({'scores': {'ere': 2}}, fit) == 'gfr'
    threshold = {name: 0. for name in ('edr', 'clr', 'gfr', 'ghr')}
    assert classify_sta(dict(edr=0, clr=1, gfr=None, ghr=None), threshold) == 'clr'
    assert classify_sta(dict(edr=0, clr=0, gfr=0, ghr=0), threshold) == 'normal'


def test_macro_accuracy_does_not_overweight_extra_channel_variant():
    rows = [{'source_id': 'x', 'label': label, 'kind': label, 'severity': .5}
            for label in ('edr', 'clr', 'clr', 'gfr', 'ghr')]
    result = classification_report(rows, ['normal', 'clr', 'clr', 'normal', 'normal'])
    assert result['macro_accuracy'] == .25


def test_ser_truth_requires_identity_and_event_agreement():
    a = [dict(type='turn', time_s=1., object_id=1, direction=0)]
    assert truth_extra(a, a) == 0
    assert truth_extra(a, [{**a[0], 'object_id': 2}]) == 1
