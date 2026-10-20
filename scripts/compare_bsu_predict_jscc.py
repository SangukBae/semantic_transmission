#!/usr/bin/env python3
"""Audit and score matched DCVC and anchor-JSCC final MP4s; no codec reruns."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.bsu_predict import inspect_frames
from semantic_transmission.official_quality import OfficialMetrics
from semantic_transmission.research_quality import read_video
from semantic_transmission.webvid5 import fingerprint
from semantic_transmission.webvid_ablation import snapshot


def read(path):
    return json.loads(path.read_text())


def audit(root, expected_status):
    protocol = read(root / 'protocol.json')
    assert protocol['signature'] == fingerprint({k: v for k, v in protocol.items() if k != 'signature'})
    dependencies = {}; checked = 0
    for name in ('prepare', 'sender', 'channel', 'decode', 'restore'):
        receipt = read(root / f'stages/{name}.json')
        assert receipt['status'] == 'PASSED'
        assert receipt['identity'] == protocol['signature']
        assert receipt['dependencies'] == dependencies
        assert snapshot(root, receipt['required']) == receipt['artifacts']
        checked += len(receipt['artifacts'])
        dependencies[name] = fingerprint(receipt)
    source = read(root / 'source.json')
    assert inspect_frames(Path(source['directory']), source['video']['fps']) == source
    result = read(root / 'RESULT.json')
    assert result['status'] == expected_status
    assert result['protocol_sha256'] == sha256(root / 'protocol.json')
    movie = root / 'reconstruction/reconstruction.mp4'
    assert result['movie_sha256'] == sha256(movie)
    return source, result, checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dcvc', type=Path, required=True)
    p.add_argument('--jscc', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('comparison output exists; use a new directory')
    old_source, old_result, old_checked = audit(args.dcvc, 'COMPLETE_BSU_PREDICT')
    source, result, checked = audit(args.jscc, 'COMPLETE_BSU_PREDICT_JSCC')
    assert source['video'] == old_source['video'] and source['hashes'] == old_source['hashes']
    old_indices = read(args.dcvc / 'decoded.json')['indices']
    indices = read(args.jscc / 'decoded.json')['indices']
    assert indices == old_indices
    old_profile = read(args.dcvc / 'protocol.json')['profile']
    profile = read(args.jscc / 'protocol.json')['profile']
    for field in ('snr_db', 'channel_seed', 'temporal_stride'):
        assert old_profile[field] == profile[field]
    assert old_profile['metadata_transport']['restore_numerics'] == profile['restore_numerics']
    args.output.mkdir(parents=True)
    paths = [Path(source['directory']) / f'{i}.png' for i in range(source['video']['frames'])]
    reference = np.stack([np.asarray(Image.open(path).convert('RGB')) for path in paths])
    reports = {}
    for label, root, completed in [('dcvc', args.dcvc, old_result), ('jscc', args.jscc, result)]:
        movie = root / 'reconstruction/reconstruction.mp4'
        frames = read_video(movie)
        assert frames.shape == reference.shape
        summary, rows = OfficialMetrics().evaluate(reference, frames)
        reports[label] = dict(channel_uses=completed['channel_uses'], quality=summary,
            movie_sha256=sha256(movie), protocol_sha256=completed['protocol_sha256'])
        with (args.output / f'{label}_per_frame.csv').open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(label, json.dumps(reports[label]), flush=True)
    report = dict(status='COMPLETE_MATCHED_BSU_PREDICT_JSCC_COMPARISON', video=source['video'],
        source_hashes=source['hashes'], source_frame_hashes_identical=True,
        transmitted_anchor_indices=indices, anchor_indices_identical=True,
        snr_db=10, channel_seed=42, restore_numerics_identical=True,
        source_reference='normalized sender RGB PNGs', evaluated_output='decoded final MP4',
        quality_profile='lgvsc_official_metrics_v1', results=reports,
        verified_stage_artifact_hashes=old_checked + checked,
        channel_uses_ratio=reports['jscc']['channel_uses'] / reports['dcvc']['channel_uses'],
        equal_actual_rate=False, independent_semantic_evaluation_performed=False,
        limits=['one video, one SNR and seed', 'different codec operating points',
                'JSCC encodes independent anchors; DCVC uses temporal prediction',
                'reference PNGs reused from the completed DCVC normalization stage'])
    write_json(args.output / 'COMPARISON.json', report)


if __name__ == '__main__':
    main()
