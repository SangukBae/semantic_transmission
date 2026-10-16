"""Replay digital BSU traffic with lossless coding; preserve all visual symbols.

This is an input-equivalence/channel experiment, not a new video-generation run.
Historical video quality is inherited only when every model input is identical.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from .artifacts import sha256, write_json
from .lossless_transport import choose, cost, restore, transmit
from .packets import accounting
from .wire import unpack

REPO = Path(__file__).resolve().parents[2]
ARMS = {'legacy': ('raw', False), 'zlib_separate': ('zlib', False),
        'xz_separate': ('xz', False), 'xz_bundle': ('xz', True)}
SEEDS = (42, 1042, 2042)


def read(path):
    return json.loads(path.read_text())


def event(stage, done, total):
    print('COMPARE_PROGRESS ' + json.dumps(dict(stage=stage, done=done, total=total,
          overall=f'{done}/{total} ({100 * done / total:.1f}%) 패킷 조건 완료')), flush=True)


def audit_case(case, baseline_row):
    run = case / 'base/reconstruction/event_pairs/run'
    report = read(case / 'RESULT.json')
    channel = read(run / 'channel_accounting.json')
    sender = read(run / 'sender_accounting.json')
    trajectory_channel = read(case / 'received/channel.json')
    if (report['mode'] != 'combined_context' or channel['status'] != 'PASSED'
            or channel['snr_db'] != 10 or channel['bits_per_symbol'] != 4
            or not channel['metadata_exact_match'] or channel['bit_errors']):
        raise ValueError('baseline channel/receiver conditions do not match')
    protected = {}

    def check(path, expected=None):
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('changed baseline artifact: ' + str(path))
        protected[str(path.resolve())] = digest
        return digest

    for name, record in sender['transmitter_files'].items():
        check(run / 'transmitter' / name, record['sha256'])
    for name, record in channel['received_files'].items():
        check(run / 'received' / name, record['sha256'])
    metadata = (run / 'transmitter/metadata.bin').read_bytes()
    if metadata != (run / 'received/metadata.bin').read_bytes():
        raise ValueError('historical metadata was not exact')
    header, rates = unpack(metadata)
    visual = (run / 'transmitter/visual.c64').stat().st_size // 8
    if visual != channel['visual_complex_channel_uses'] or visual != sum(k['complex_count'] for k in header['keyframes']):
        raise ValueError('visual symbol accounting mismatch')
    if (run / 'received/visual.c64').stat().st_size != visual * 8:
        raise ValueError('received visual stream length mismatch')
    trajectory = None
    if trajectory_channel['packet_present']:
        trajectory = (case / 'sender/trajectory.bin').read_bytes()
        if trajectory != (case / 'received/trajectory.bin').read_bytes() or trajectory_channel['bit_errors']:
            raise ValueError('historical trajectory was not exact')
        th, _ = unpack(trajectory)
        if th['base_metadata_sha256'] != hashlib.sha256(metadata).hexdigest():
            raise ValueError('trajectory/base binding failed')
        for name in ('sender', 'received'):
            check(case / name / 'trajectory.bin')
    expected_digital = channel['digital_complex_channel_uses'] + trajectory_channel['complex_channel_uses']
    if cost([metadata] + ([trajectory] if trajectory else [])) != expected_digital:
        raise ValueError('legacy padding/parity accounting mismatch')
    if visual + expected_digital != baseline_row['total_channel_uses']:
        raise ValueError('baseline total rate mismatch')
    check(case / 'refined/reconstruction.mp4', report['refined_quality']['video_sha256'])
    for path in [case / 'RESULT.json', run / 'channel_accounting.json', run / 'sender_accounting.json',
                 case / 'received/channel.json', run / 'receiver/metadata.csv',
                 run / 'receiver/decoder_inputs.json', run / 'receiver/decoder_config.py']:
        check(path)
    keys = run / 'receiver/frames/sample/key_frames_received'
    for key in header['keyframes']:
        check(keys / f"{key['index']}.png")
    return metadata, trajectory, dict(id=case.name, keys=len(header['keyframes']),
        frames=header['video']['frames'], visual_uses=visual, legacy_digital_uses=expected_digital,
        legacy_total_uses=visual + expected_digital, protected=protected)


def run(baseline, output, channel_enabled):
    if output.exists():
        raise ValueError('output already exists; use a new output directory')
    baseline_result = read(baseline / 'RESULT.json')
    if baseline_result['status'] != 'COMPLETE_RECONSTRUCTION_QUALITY_ACTION_EVALUATION':
        raise ValueError('baseline is not a completed quality/action evaluation')
    source_hash = sha256(baseline / 'RESULT.json')
    cases = []
    for row in baseline_result['videos']:
        cases.append(audit_case(baseline / 'benchmark/snr_10' / row['id'], row))
    output.mkdir(parents=True)
    code = {str(p.relative_to(REPO)): sha256(p) for p in [
        REPO / 'src/semantic_transmission/lossless_transport.py',
        REPO / 'src/semantic_transmission/bsu_lossless_validation.py',
        REPO / 'src/semantic_transmission/wire.py',
        REPO / 'src/semantic_transmission/packets.py',
        REPO / 'src/semantic_transmission/metadata_channel.py']}
    write_json(output / 'protocol.json', dict(baseline=str(baseline), source_result_sha256=source_hash,
        code=code, arms=ARMS, seeds=SEEDS, snr_db=10, ldpc_k=6144, ldpc_n=9216, bits_per_symbol=4,
        physical_link_overhead_included=False, visual_symbols='all original TX/RX symbols retained byte-exact',
        decision='sender byte cost only, no reconstruction/label-based candidate selection',
        output_claim='lossless receiver-input equivalence, not regenerated video metrics',
        cases=[c[2] for c in cases]))
    started = time.monotonic()
    rows, trials, done = [], [], 0
    total = len(cases) * len(ARMS) * (len(SEEDS) if channel_enabled else 1)
    for metadata, trajectory, case in cases:
        expected = {'metadata': metadata}
        if trajectory is not None:
            expected['trajectory'] = trajectory
        for arm, (method, bundle) in ARMS.items():
            begin = time.monotonic()
            mode, messages = choose(metadata, trajectory, method=method, bundle=bundle)
            encode_seconds = time.monotonic() - begin
            begin = time.monotonic()
            if restore(messages) != expected:
                raise ValueError('local input equivalence failed')
            decode_seconds = time.monotonic() - begin
            target = output / 'packets' / case['id'] / arm
            target.mkdir(parents=True)
            for i, packet in enumerate(messages):
                (target / f'{i}.bin').write_bytes(packet)
            digital = cost(messages)
            rows.append(dict(id=case['id'], arm=arm, mode=mode, keys=case['keys'],
                visual_uses=case['visual_uses'], digital_uses=digital,
                total_uses=case['visual_uses'] + digital, legacy_total_uses=case['legacy_total_uses'],
                saving_percent=100 * (case['legacy_digital_uses'] - digital) / case['legacy_total_uses'],
                encode_seconds=encode_seconds, decode_seconds=decode_seconds))
            for seed in SEEDS if channel_enabled else (None,):
                if channel_enabled:
                    restored, result = transmit(metadata, trajectory, 10, seed, method=method, bundle=bundle)
                    if restored != expected or result['complex_channel_uses'] != digital:
                        raise ValueError('channel result disagrees with sender cost/equivalence')
                    result.update(id=case['id'], arm=arm, seed=seed)
                    # Receiver artifacts contain the original packets, retaining
                    # existing hashes and base/trajectory bindings unchanged.
                    destination = target / f'received_{seed}'
                    destination.mkdir()
                    for name, packet in restored.items():
                        (destination / f'{name}.bin').write_bytes(packet)
                    write_json(destination / 'channel.json', result)
                    trials.append(result)
                done += 1
                event('lossless_channel' if channel_enabled else 'lossless_roundtrip', done, total)
    protected = {p: h for _, _, c in cases for p, h in c['protected'].items()}
    if sha256(baseline / 'RESULT.json') != source_hash:
        raise ValueError('baseline result changed during validation')
    for path, digest in protected.items():
        if sha256(Path(path)) != digest:
            raise ValueError('protected baseline changed: ' + path)
    summaries = {}
    original_total = sum(c[2]['legacy_total_uses'] for c in cases)
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        total_uses = sum(r['total_uses'] for r in selected)
        summaries[arm] = dict(keys=sum(r['keys'] for r in selected), total_uses=total_uses,
            digital_uses=sum(r['digital_uses'] for r in selected),
            saved_uses=original_total - total_uses, saving_percent=100 * (original_total - total_uses) / original_total,
            every_video_nonincreasing=all(r['total_uses'] <= r['legacy_total_uses'] for r in selected),
            min_video_saving_percent=min(r['saving_percent'] for r in selected),
            max_video_saving_percent=max(r['saving_percent'] for r in selected),
            encode_seconds=sum(r['encode_seconds'] for r in selected),
            decode_seconds=sum(r['decode_seconds'] for r in selected))
    with (output / 'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='COMPLETE_LOSSLESS_CHANNEL_EQUIVALENCE' if channel_enabled else 'COMPLETE_LOCAL_ROUNDTRIP_ONLY',
        baseline_result_sha256=source_hash, videos=len(cases), frames=sum(c[2]['frames'] for c in cases),
        summary=summaries, channel_trials=len(trials), digital_packet_transmissions=sum(len(t['reports']) for t in trials),
        bit_errors=sum(r['bit_errors'] for t in trials for r in t['reports']),
        all_received_model_packets_byte_exact=True, protected_artifacts_unchanged=len(protected),
        downstream_generation_rerun=False, downstream_quality='inherited by exact model-input equivalence, not new measurements',
        baseline_quality=baseline_result['quality']['refined']['delivered_mp4']['video_mean'],
        baseline_action=baseline_result['action']['bsu_context'],
        limits=['10 dB AWGN, three digital channel seeds; no all-SNR packet failure guarantee',
                'bundle waits for both packets and couples their decoding failure',
                'continuous visual JSCC symbol count is unchanged; no RF savings from complex64 file compression',
                'shared receiver transport decoder required; production defaults unchanged',
                'identical inputs imply same deterministic output/distribution; GPU bitwise generation not rerun'],
        elapsed_seconds=time.monotonic() - started)
    write_json(output / 'RESULT.json', result)
    files = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'completion_receipt.json', dict(files=files, code=code))
    print(json.dumps({'status': result['status'], 'summary': summaries}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=REPO / 'outputs/bsu_kinetics_context')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channel', action='store_true', help='run actual CPU Sionna LDPC/16QAM/AWGN')
    args = parser.parse_args()
    run(args.baseline.resolve(), args.output.resolve(), args.channel)


if __name__ == '__main__':
    main()
