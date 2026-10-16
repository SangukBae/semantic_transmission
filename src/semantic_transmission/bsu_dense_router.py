"""Route using actual sender cost, without inspecting quality or action labels."""
from .wire import pack, unpack
from .packets import accounting
from .lossless_transport import choose as lossless, restore

POLICY = 'bsu_cost_routed_dense_v1'


def choose(metadata, trajectory, visual_count, neural, minimum_keys=0):
    from .bsu_dense_neural import unpack_packet
    if type(visual_count) is not int or visual_count < 1:
        raise ValueError('positive visual symbol count required')
    if type(minimum_keys) is not int or minimum_keys < 0:
        raise ValueError('invalid source key-count floor')
    context, _ = unpack_packet(neural)
    original, _ = unpack(metadata)
    expected = dict(video=original['video'], indices=[k['index'] for k in original['keyframes']],
                    segments=original['segments'], decoder=original['decoder'])
    if context != expected:
        raise ValueError('original BSU and neural stream have different semantic/video inputs')
    _, messages = lossless(metadata, trajectory, method='xz', bundle=True)
    old = pack(dict(schema=POLICY, mode='bsu', lengths=[len(x) for x in messages],
                    visual_complex_count=visual_count), b''.join(messages))
    new = pack(dict(schema=POLICY, mode='neural', lengths=[len(neural)], visual_complex_count=0), neural)
    costs = dict(bsu=accounting(len(old))['complex_channel_uses']+visual_count,
                 neural=accounting(len(new))['complex_channel_uses'])
    eligible = len(original['keyframes']) >= minimum_keys
    mode = 'neural' if eligible and costs['neural'] < costs['bsu'] else 'bsu'
    return (new if mode == 'neural' else old), dict(mode=mode, channel_uses=costs[mode],
        candidate_costs=costs, visual_channel_uses=0 if mode == 'neural' else visual_count,
        minimum_original_keys=minimum_keys, neural_eligible=eligible,
        selection_reads_source_quality=False)


def receive(data):
    from .bsu_dense_neural import unpack_packet
    h, p = unpack(data)
    if h.get('schema') != POLICY or h.get('mode') not in ('bsu', 'neural'):
        raise ValueError('unknown BSU route')
    lengths = h['lengths']
    if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != len(p):
        raise ValueError('invalid routed packet lengths')
    messages, start = [], 0
    for n in lengths:
        messages.append(p[start:start+n]); start += n
    if h['mode'] == 'neural':
        if len(messages) != 1 or h['visual_complex_count'] != 0:
            raise ValueError('invalid learned route')
        unpack_packet(messages[0])
        return 'neural', dict(neural=messages[0])
    if type(h['visual_complex_count']) is not int or h['visual_complex_count'] <= 0:
        raise ValueError('missing analog symbol count')
    return 'bsu', restore(messages)


def validate(root, neural_root, minimum_keys=0):
    import copy
    import statistics
    from pathlib import Path
    from .artifacts import sha256, write_json
    from .bsu_dense_validation import REPO, read
    from .bsu_lossless_validation import audit_case
    from .metadata_channel import transmit
    from .key_reduction_selection import guard
    from .official_quality import KEYS
    from .qwen35_pairs_video import freeze
    from .webvid5 import fingerprint
    result = read(neural_root/'RESULT.json'); protocol = read(neural_root/'protocol.json')
    if result['status'] not in ('COMPLETE_NEURAL_DENSE_RATE_QUALITY_ACTION',
                              'COMPLETE_TEMPORAL_NEURAL_RATE_QUALITY_ACTION') or protocol['scale'] != 1:
        raise ValueError('complete native neural evaluation required')
    action = read(neural_root/'action/RESULT.json')
    baseline = read(REPO/'outputs/bsu_kinetics_context/RESULT.json')
    baseline_rows = {r['id']: r for r in baseline['videos']}
    if {c['id'] for c in protocol['cases']} != set(baseline_rows) or len(protocol['cases']) != len(baseline_rows):
        raise ValueError('rate comparison requires the complete matched BSU/LGVSC cohort')
    root.mkdir(parents=True, exist_ok=True)
    p = dict(policy=POLICY, neural=str(neural_root), neural_result_sha256=sha256(neural_root/'RESULT.json'),
        code_sha256=sha256(Path(__file__)), choice='minimum actual TX symbols including wrapper; no quality/labels',
        minimum_original_keys=minimum_keys,
        digital_seeds=[42,1042,2042], analog_branch='original verified AWGN symbols and received keys reused',
        output_reuse='inner decoder bytes identical; source/output/action hashes verified; no video regeneration')
    p['signature'] = fingerprint(p); freeze(root/'protocol.json', p)
    records = []
    for case in protocol['cases']:
        cid = case['id']; old = Path(case['original_run'])
        metadata, trajectory, audit = audit_case(old.parents[3], baseline_rows[cid])
        coded = (neural_root/cid/'sender/packet.bin').read_bytes()
        if coded != (neural_root/cid/'received/packet.bin').read_bytes():
            raise ValueError('neural inputs not exact')
        data, choice = choose(metadata, trajectory, audit['visual_uses'], coded, minimum_keys)
        dest = root/cid; dest.mkdir(exist_ok=True); (dest/'packet.bin').write_bytes(data)
        reports = []
        for seed in p['digital_seeds']:
            rx, report = transmit(data, 10, seed, bits_per_symbol=4)
            if rx != data or report['bit_errors']:
                raise ValueError('routed packet channel failed')
            mode, restored = receive(rx)
            expected = {'neural': coded} if mode == 'neural' else {
                'metadata': metadata, **({'trajectory': trajectory} if trajectory is not None else {})}
            if restored != expected:
                raise ValueError('inner decoder input changed')
            if report['complex_channel_uses']+choice['visual_channel_uses'] != choice['channel_uses']:
                raise ValueError('routed cost differs from actual channel')
            reports.append(report)
            if seed == 42:
                (dest/'received.bin').write_bytes(rx)
        candidate = next(r for r in result['rows'] if r['id'] == cid and r['mode'] == 'native')
        selected = candidate['quality'] if choice['mode'] == 'neural' else candidate['baseline']
        q = copy.deepcopy(selected); q['channel_uses'] = choice['channel_uses']
        if (sha256(q['movie']) != q['movie_sha256'] or
                sha256(case['source']['normalized_video']) != q['source_movie_sha256']):
            raise ValueError('reused output/source changed')
        ar = next(r for r in action['records'] if r['input']['id'] == cid and
                  r['input']['variant'] == ('native' if choice['mode'] == 'neural' else 'baseline'))
        if ar['input']['sha256'] != q['movie_sha256']:
            raise ValueError('classifier output differs')
        row = dict(id=cid, choice=choice, quality=q, baseline=candidate['baseline'], action=ar,
            strict_guard=guard(candidate['baseline'], q), channel=reports, video_regenerated=False,
            exact_inner_decoder_inputs_verified=True, original_artifacts=audit['protected'])
        write_json(dest/'RESULT.json', row); records.append(row)
    uses = sum(r['quality']['channel_uses'] for r in records)
    original = sum(r['baseline']['channel_uses'] for r in records)
    summary = dict(channel_uses=uses, bsu_saving_percent=100*(1-uses/original), ratio_to_lgvsc=uses/1407617,
        routes={m: sum(r['choice']['mode'] == m for r in records) for m in ('bsu','neural')},
        mean_quality={k: statistics.mean(r['quality']['quality'][k] for r in records) for k in KEYS},
        mean_temporal={k: statistics.mean(r['quality']['temporal'][k] for r in records) for k in
                       ['flow_error_mean','cadence_error_mean','false_freeze_rate']},
        action=dict(top1_correct=sum(r['action']['top1_correct'] for r in records),
                    top5_correct=sum(r['action']['top5_correct'] for r in records), videos=len(records)),
        strict_all_metric_passes=sum(r['strict_guard']['passed'] for r in records))
    write_json(root/'RESULT.json', dict(status='COMPLETE_COST_ROUTED_CHANNEL_INPUT_EQUIVALENCE_AND_REUSED_QUALITY',
        summary=summary, rows=records, per_clip_quality_adaptive_selection=False, generalization_proven=False))


def main():
    import argparse
    from pathlib import Path
    from .bsu_dense_validation import REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural', type=Path, default=REPO/'outputs/bsu_dense_neural_qp32_20261015_v1')
    parser.add_argument('--root', type=Path, default=REPO/'outputs/bsu_dense_routed_qp32_20261016_v1')
    parser.add_argument('--min-original-keys', type=int, default=0)
    args = parser.parse_args(); validate(args.root.resolve(), args.neural.resolve(), args.min_original_keys)


if __name__ == '__main__':
    main()
