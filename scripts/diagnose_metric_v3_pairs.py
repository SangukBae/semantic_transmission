#!/usr/bin/env python3
"""ERE and observed TX/RX STA diagnostics on existing reconstruction pairs.

Sparse keyframe hold interpolation is explicitly diagnostic. Missing SGD boundary
records yield unavailable STA, never substituted ground-truth representations.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.attribution_metric import attribution
from semantic_transmission.metric_v3_features import VisualObserver
from semantic_transmission.metric_v3_formal import read, save, verify
from semantic_transmission.pair_inputs import load_pair
from semantic_transmission.sta_video_validation import score_video_attribution


def sparse_video(paths, indices, native_fps, count):
    keyframes = []
    for path in paths:
        value = cv2.imread(str(path))
        if value is None:
            raise ValueError('unreadable saved keyframe: ' + str(path))
        keyframes.append(cv2.resize(cv2.cvtColor(value, cv2.COLOR_BGR2RGB), (320, 192), interpolation=cv2.INTER_AREA))
    times = [i / native_fps for i in indices]
    frames = [keyframes[max(0, bisect.bisect_right(times, t / 8.) - 1)] for t in range(count)]
    status = ['not_transmitted'] * count
    offsets = []
    for time_s in times:
        # The held pixel at a sample before this time still belongs to the
        # previous keyframe. Mark only the first sample that can read this one.
        t = int(np.ceil(time_s * 8 - 1e-9))
        if t < count:
            status[t] = 'ok'; offsets.append(abs(t / 8. - time_s))
    return np.stack(frames), status, max(offsets, default=0.)


def captions(rows, indices, fps):
    if len(rows) != len(indices) - 1:
        raise ValueError('caption/segment boundary mismatch')
    return [{'start_s': indices[i] / fps, 'end_s': indices[i + 1] / fps,
             'text': row['text'], 'decode_status': 'ok'} for i, row in enumerate(rows)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pairs', type=Path, default=Path('outputs/mte_otf_20260914_v1/real_pairs.json'))
    args = ap.parse_args(); root = args.output; p, base = verify(root)
    destination = root / 'real_pair_diagnostics.json'
    if destination.exists():
        raise FileExistsError(destination)
    import torch
    torch.set_num_threads(4); torch.manual_seed(20260914); cv2.setNumThreads(1)
    observer = VisualObserver(base, root, p['signature'])
    records = read(args.pairs)['pairs']; results = []
    save(root / 'real_pair_diagnostic_declaration.json', {'created_unix': time.time(),
        'script_sha256': sha256(Path(__file__)), 'manifest_sha256': sha256(args.pairs),
        'protocol_sha256': sha256(root / 'protocol.json'), 'semantic_ground_truth': None,
        'scope': 'existing reconstructions only; physical timestamps, no rate-matched model ranking',
        'sta_scope': 'saved TX/RX keyframes and literal captions, sparse keyframe zero-order hold is an unvalidated content-reading approximation'})
    for row in records:
        started = time.time(); a, b, alignment = load_pair(row, width=320, height=192, sample_fps=8.)
        scores = observer.compare(a, b)
        result = {**row, 'alignment': alignment, 'scores': scores, 'semantic_ground_truth': None,
                  'sta': None, 'sta_unavailable_reason': None}
        receiver = Path(row['reconstruction']).parent.parent
        run = receiver.parent
        descriptor = receiver / 'decoder_inputs.json'
        if receiver.name != 'receiver' or not descriptor.exists():
            result['sta_unavailable_reason'] = 'No compatible saved TX/RX boundary representation in this pair; source/reconstruction alone cannot identify the stage.'
        else:
            cfg = read(descriptor); fps = cfg['video']['fps']; rx_indices = cfg['indices']
            tx_indices = read(run / 'keyframes.json')['indices']
            tx_paths = [run / 'data/frames/sample' / (str(i) + '.png') for i in tx_indices]
            rx_paths = [receiver / 'frames/sample/key_frames_received' / (str(i) + '.png') for i in rx_indices]
            try:
                tx, ts, to = sparse_video(tx_paths, tx_indices, fps, len(a))
                rx, rs, ro = sparse_video(rx_paths, rx_indices, fps, len(a))
                value = score_video_attribution(observer, a, b, tx, rx, ts, rs)
                value['tx_representation']['captions'] = captions(read(run / 'metadata_tx.json'), tx_indices, fps)
                with (receiver / 'metadata.csv').open(newline='') as f:
                    rx_captions = list(csv.DictReader(f))
                value['rx_representation']['captions'] = captions(rx_captions, rx_indices, fps)
                pairs = scores['event_pairs'] + [(scores['reference_event_count'] + i, scores['reconstruction_event_count'] + j)
                                                  for i, j in scores.get('track_pairs', [])]
                rates = attribution(value['units_reference'], value['units_reconstruction'],
                                    value['tx_representation'], value['rx_representation'], pairs)
                result['sta'] = {**rates, 'units_reference': value['units_reference'], 'units_reconstruction': value['units_reconstruction'],
                    'tx_representation': value['tx_representation'], 'rx_representation': value['rx_representation'],
                    'tx_keyframe_count': len(tx_paths), 'rx_keyframe_count': len(rx_paths),
                    'tx_observed_keyframe_slots': ts.count('ok'), 'rx_observed_keyframe_slots': rs.count('ok'),
                    'max_keyframe_sample_time_offset_s': max(to, ro),
                    'keyframe_sampling_policy': 'first 8 Hz sample at or after the actual keyframe; never label prior held pixels as that keyframe',
                    'scope': 'diagnostic-only sparse hold approximation and literal captions; no semantic/cause truth',
                    'limitations': ['Few keyframes cannot reliably reveal starts/turns.',
                                    'Automatic object tokens are track identities; natural caption object nouns are not certified token mappings.',
                                    'Transmitted optical-flow side information is not interpreted by this adapter.',
                                    'Keyframes later than the last shared sample are outside the diagnostic window.']}
                files = [descriptor, run / 'metadata_tx.json', receiver / 'metadata.csv', run / 'keyframes.json', *tx_paths, *rx_paths]
                result['representation_sha256'] = {str(f): sha256(f) for f in files}
            except (ValueError, OSError, KeyError) as error:
                result['sta_unavailable_reason'] = str(error)
        result['elapsed_s'] = time.time() - started
        results.append(result)
        save(root / 'real_pair_diagnostics.partial.json', {'results': results})
        print(row['source_id'], row['model'], 'ERE', scores['ere'], 'STA_available', result['sta'] is not None, flush=True)
    save(destination, {'status': 'COMPLETED', 'results': results, 'pairs': len(results),
        'sta_available_pairs': sum(r['sta'] is not None for r in results), 'semantic_ground_truth': None,
        'script_sha256': sha256(Path(__file__)), 'scope': 'unlabelled existing reconstruction diagnostics; no model ranking or accuracy claim'})


if __name__ == '__main__':
    main()
