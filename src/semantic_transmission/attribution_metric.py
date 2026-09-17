"""STA bookkeeping on observed units and explicit TX/RX representations.

This is descriptive attribution under a support rule, not causal identification.
An oracle scene-state representation is allowed only in labelled simulator tests.
"""
import hashlib
import math
import re

DIRECTIONS = ("east", "southeast", "south", "southwest", "west", "northwest", "north", "northeast")


def _tokens(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def decode_keyframe(payload, time_s, content_reader, expected_sha256=None):
    """Decode received bytes, then read content without a reference/truth input.

The reader receives only decoded RGB. It may abstain from event recognition in
an isolated frame. A temporal reader must obtain its context from decoded RX
frames, never from the original event labels.
"""
    record = {'time_s': float(time_s), 'decoded_events': [], 'decoded_object_tokens': []}
    if payload is None:
        return {**record, 'decode_status': 'packet_lost'}
    digest = hashlib.sha256(payload).hexdigest()
    record['payload_sha256'] = digest
    if expected_sha256 is not None and digest != expected_sha256:
        return {**record, 'decode_status': 'checksum_failed'}
    import cv2
    import numpy as np
    try:
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        frame = None
    if frame is None:
        return {**record, 'decode_status': 'decode_failed'}
    try:
        content = content_reader(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not isinstance(content, dict) or content.get('observable') is False:
            return {**record, 'decode_status': 'extraction_failed'}
        events = content.get('events', [])
        tokens = content.get('object_tokens', [])
        if not isinstance(events, list) or not isinstance(tokens, list):
            raise ValueError('content reader must return lists')
    except (RuntimeError, ValueError, TypeError, KeyError) as error:
        return {**record, 'decode_status': 'extraction_failed', 'error_type': type(error).__name__}
    return {**record, 'decode_status': 'ok', 'decoded_events': events,
            'decoded_object_tokens': tokens, 'content_source': 'decoded_pixels'}


def _readable(item):
    return item.get('decode_status') == 'ok' and item.get('decodable', True) is not False and not any(
        item.get(k, False) for k in ('packet_lost', 'checksum_failed', 'content_corrupted', 'extraction_failed'))


def _event_agrees(unit, state):
    if state.get('type') != unit['type']:
        return False
    try:
        direction = int(state['direction']); cell = state['cell']
        if not 0 <= direction <= 7 or len(cell) != 2 or any(not 0 <= c <= 2 for c in cell):
            return False
        diff = abs(direction - unit['direction'])
        return min(diff, 8 - diff) <= 1 and max(abs(x - y) for x, y in zip(cell, unit['cell'])) <= 1
    except (KeyError, TypeError, ValueError):
        return False


def _readable_caption(caption):
    # A literal text field is already decoded text; explicit transport/extraction
    # failure wins even if stale text remains beside it.
    return isinstance(caption.get('text'), str) and _readable({'decode_status': 'ok', **caption})


def support(unit, representation):
    """No intervention label is accepted; support comes from representation content."""
    if unit["kind"] == "event":
        time_s = unit["time_s"]
        if any(_readable(k) and abs(float(k['time_s']) - time_s) <= .25 + 1e-9
               and any(_event_agrees(unit, state) for state in k.get('decoded_events', []))
               for k in representation.get("keyframes", [])):
            return True
        required = {unit["type"], DIRECTIONS[unit["direction"]]}
        return any(_readable_caption(c) and c["start_s"] <= time_s <= c["end_s"] and required <= _tokens(c["text"])
                   for c in representation.get("captions", []))
    if unit["kind"] == "object":
        # Identity/word tokens are supplied by an extractor or simulator adapter;
        # there is deliberately no hidden lookup into renderer truth here.
        token = unit["object_token"]
        return any(_readable(k) and token in k.get("decoded_object_tokens", []) for k in representation.get("keyframes", [])) or any(
            _readable_caption(c) and token in _tokens(c["text"]) for c in representation.get("captions", []))
    raise ValueError("semantic unit must be event or object")


def attribution(reference, reconstructed, tx, rx, pairs):
    if any(i < 0 or i >= len(reference) or j < 0 or j >= len(reconstructed) for i, j in pairs):
        raise ValueError("invalid unit pair")
    if len({i for i, _ in pairs}) != len(pairs) or len({j for _, j in pairs}) != len(pairs):
        raise ValueError("one-to-one correspondence required")
    reproduced, existing = {i for i, _ in pairs}, {j for _, j in pairs}
    st = [support(u, tx) for u in reference]
    sr = [support(u, rx) for u in reference]
    rb = [support(u, rx) for u in reconstructed]
    counts = {"edr": (sum(not x for x in st), len(reference)),
              "clr": (sum(t and not r for t, r in zip(st, sr)), sum(st)),
              "gfr": (sum(r and i not in reproduced for i, r in enumerate(sr)), sum(sr)),
              "ghr": (sum(not r and j not in existing for j, r in enumerate(rb)), len(reconstructed))}
    result = {key: num / den if den else None for key, (num, den) in counts.items()}
    if not all(v is None or math.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError("invalid attribution rate")
    return {**result, "numerators_denominators": counts,
            "reference_support_tx": st, "reference_support_rx": sr, "reconstruction_support_rx": rb,
            "rx_supported_without_tx_support": sum(r and not t for t, r in zip(st, sr)),
            "unreadable_keyframes": {side: [k.get('decode_status', 'not_decoded') for k in rep.get('keyframes', []) if not _readable(k)]
                                     for side, rep in (('tx', tx), ('rx', rx))},
            "scope": "support-rule attribution; not proof of decoder causality"}


def rule_checks():
    """Six declared analytical cases, not a video attribution-accuracy estimate."""
    units = [{'kind': 'event', 'type': 'start', 'time_s': .5, 'direction': 0, 'cell': [1, 1]},
             {'kind': 'event', 'type': 'turn', 'time_s': 1.5, 'direction': 4, 'cell': [1, 1]}]
    rep = {'keyframes': [{'time_s': u['time_s'], 'decode_status': 'ok', 'decoded_events': [u],
                         'content_source': 'analytical_fixture'} for u in units]}
    damaged = {'keyframes': [{**k, 'decode_status': 'checksum_failed', 'content_corrupted': True} for k in rep['keyframes']]}
    extra = {'kind': 'object', 'object_token': 'triangle'}
    cases = [('untouched', units, rep, rep, [(0, 0), (1, 1)], [0., 0., 0., 0.]),
             ('tx_omission', [], {}, {}, [], [1., None, None, None]),
             ('packet_drop', [], rep, {}, [], [0., 1., None, None]),
             ('generation_omission', [], rep, rep, [], [0., 0., 1., None]),
             ('unsupported_addition', units + [extra], rep, rep, [(0, 0), (1, 1)], [0., 0., 0., 1 / 3]),
             ('corrupt_keyframe_retained', [], rep, damaged, [], [0., 1., None, None])]
    results = []
    for name, rec, tx, rx, pairs, expected in cases:
        result = attribution(units, rec, tx, rx, pairs)
        values = [result[k] for k in ('edr', 'clr', 'gfr', 'ghr')]
        results.append({'name': name, 'computed': result, 'expected': expected, 'passed': values == expected})
    return {'status': 'PASSED' if all(r['passed'] for r in results) else 'NOT_PASSED',
            'cases': len(results), 'passed': sum(r['passed'] for r in results), 'results': results,
            'scope': 'analytical support-rule checks, not decoded-video or heldout attribution accuracy'}
