"""Verify two completed BSU-Predict runs against their real saved artifacts."""
import argparse
import json
from pathlib import Path

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.bsu_predict_compression import split
from semantic_transmission.packets import accounting


def read(path):
    return json.loads(path.read_text())


def compare(root):
    control, compressed = root / 'control', root / 'compressed'
    records = [read(p / 'protocol.json') for p in (control, compressed)]
    before, after = [read(p / 'RESULT.json') for p in (control, compressed)]
    assert before['status'] == after['status'] == 'COMPLETE_BSU_PREDICT'
    assert records[0]['source'] == records[1]['source']
    for key in ('assets', 'vendor_code', 'runtimes', 'code', 'config_sha256'):
        assert records[0][key] == records[1][key], key
    profiles = [{k:v for k,v in p['profile'].items() if k != 'metadata_transport'} for p in records]
    assert profiles[0] == profiles[1]
    assert records[0]['profile']['metadata_transport']['mode'] == 'raw'
    assert records[1]['profile']['metadata_transport']['mode'] == 'auto'
    transport_profiles = [{k:v for k,v in p['profile']['metadata_transport'].items() if k != 'mode'} for p in records]
    assert transport_profiles[0] == transport_profiles[1]
    # Check each completed stage's artifact hashes, not only its status flag.
    checked = 0
    for directory in (control, compressed):
        for stage in directory.glob('stages/*.json'):
            receipt = read(stage); assert receipt['status'] == 'PASSED'
            for relative, expected in receipt['artifacts'].items():
                assert sha256(directory / relative) == expected, relative
                checked += 1
    packets = [p / 'sender/packet_original.bin' for p in (control, compressed)]
    original_equal = sha256(packets[0]) == sha256(packets[1])
    assert original_equal, 'independent encoders produced different packets; not a valid compression pair'
    for directory, packet in zip((control, compressed), packets):
        assert sha256(directory / 'received/packet.bin') == sha256(packet)
    snapshots = []
    for relative in ('prepared', 'decoded', 'reconstruction/frames'):
        maps = [{p.name:sha256(p) for p in (folder / relative).glob('*.png')}
                for folder in (control, compressed)]
        assert maps[0] and maps[0] == maps[1], relative
        snapshots.append(dict(boundary=relative, files=len(maps[0]), all_sha256_identical=True))
    expected_frames = read(control / 'source.json')['video']['frames']
    assert snapshots[-1]['files'] == expected_frames
    mp4_equal = sha256(control / 'reconstruction/reconstruction.mp4') == sha256(compressed / 'reconstruction/reconstruction.mp4')
    assert mp4_equal, 'delivered MP4 differs'
    channels = [read(p / 'received/channel.json') for p in (control, compressed)]
    for directory, channel in zip((control, compressed), channels):
        assert channel['bit_errors'] == 0 and channel['original_packet_restored_exactly']
        budget = accounting((directory / 'sender/packet.bin').stat().st_size)
        assert all(channel[k] == v for k, v in budget.items())
    a, b = channels
    prefix, stream = split(packets[0].read_bytes())
    result = dict(status='COMPLETE_BSU_PREDICT_COMPRESSION_PAIR', model='BSU-Predict',
                  metadata_transport=transport_profiles[0],
                  source=records[0]['source'], video=read(control / 'source.json')['video'],
                  original_encoded_packets_identical=original_equal, decoded_and_restored_frames=snapshots,
                  final_mp4_identical=mp4_equal, checked_artifact_hashes=checked,
                  original_metadata_prefix_bytes=len(prefix), dcvc_stream_bytes=len(stream),
                  packet_bytes_before=a['packet_bytes'], packet_bytes_after=b['packet_bytes'],
                  bytes_saved=a['packet_bytes']-b['packet_bytes'],
                  channel_uses_before=a['complex_channel_uses'], channel_uses_after=b['complex_channel_uses'],
                  channel_saving_percent=100*(1-b['complex_channel_uses']/a['complex_channel_uses']),
                  ldpc_blocks_before=a['blocks'], ldpc_blocks_after=b['blocks'],
                  bit_errors=[a['bit_errors'], b['bit_errors']],
                  compressed_video=str(compressed / 'reconstruction/reconstruction.mp4'),
                  compressed_video_sha256=sha256(compressed / 'reconstruction/reconstruction.mp4'),
                  results=[dict(path=str(p / 'RESULT.json'), sha256=sha256(p / 'RESULT.json')) for p in (control, compressed)],
                  quality_metrics_recomputed=False, full_cohort_validated=False)
    write_json(root / 'COMPARISON.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.root.resolve()), indent=2))
