"""Frozen source-reviewed 2x2 diagnostic; never trains or feeds facts to Qwen."""
import argparse
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
import time

from .qwen35_training_data import ROOT, INVENTORY, read_json, write_json, sha256, exclusion_keys

DEFAULT_RUN = ROOT / 'outputs/qwen35_oracle_diagnostic_v1'
REFERENCE = ROOT / 'configs/qwen35_oracle_diagnostic_v1.json'
CHECKPOINT = ROOT / 'outputs/qwen35_fact_joint_v3/control'


def witnessed(groups, keys):
    """All groups need distinct, ordered witnesses; empty evidence never passes."""
    if not groups:
        return False
    previous = -1
    for group in groups:
        candidates = sorted(i for i in set(group).intersection(keys) if i > previous)
        if not candidates:
            return False
        previous = candidates[0]
    return True


def coverage(row, keys):
    hits = {f['id']: witnessed(f['evidence_groups'], keys) for f in row['facts']}
    return {'facts': hits, 'fact_hits': sum(hits.values()), 'fact_count': len(hits),
            'event_hits': sum(hits[f['id']] for f in row['facts'] if f['kind'] == 'event'),
            'event_count': sum(f['kind'] == 'event' for f in row['facts'])}


def oracle(row, endpoints=True):
    candidates = [(0, *c, 7) for c in combinations(range(1, 7), 2)] if endpoints else list(combinations(range(8), 4))
    def score(keys):
        cov = coverage(row, keys)
        gaps = [b-a for a,b in zip(keys, keys[1:])]
        return (-cov['event_hits'], -cov['fact_hits'], max(gaps)-min(gaps), keys)
    keys = list(min(candidates, key=score))
    return {'keys': keys, 'coverage': coverage(row, keys), 'subsets_checked': len(candidates)}


def reject_overlap(source, prior, protected_ids, protected_hashes):
    if source['source_id'] in protected_ids or source['source_sha256'] in protected_hashes:
        raise ValueError('Protected evaluation source')
    for old in prior:
        if old['split'] in ('train', 'validation', 'test') and (
            old['source_id'] == source['source_id'] or old['source_sha256'] == source['source_sha256']):
            raise ValueError('Prior pilot source overlap')
    if source['split'] != 'reserve_unreviewed':
        raise ValueError('Expected reserve source')


def prepare(run):
    from .qwen35_training import CONFIG, checkpoint_path
    if (run/'reference_freeze.json').exists():
        verify(run)
        return
    refs = read_json(REFERENCE)
    prior_path = ROOT/'outputs/qwen35_multitask_data_v1/source_split.json'
    prior = read_json(prior_path)['sources']
    sources = {s['id']: s for s in prior}
    protected_ids, protected_hashes, exclusion_files = exclusion_keys()
    rows = []
    checked = {}
    for n, ref in enumerate(refs['records']):
        source = sources[ref['video_id']]
        reject_overlap(source, prior, protected_ids, protected_hashes)
        folder = INVENTORY/source['dataset'].lower()/source['id']
        metadata = read_json(folder/'source_prepared.json')
        if source['id'] not in checked:
            if sha256(source['source_path']) != source['source_sha256']:
                raise ValueError('Original source hash changed')
            if sha256(folder/'normalized.mp4') != metadata['normalized_sha256']:
                raise ValueError('Normalized source hash changed')
            checked[source['id']] = dict(source, normalized_path=str(folder/'normalized.mp4'),
                normalized_sha256=metadata['normalized_sha256'])
        frames = []
        if ref['indices'] != sorted(set(ref['indices'])) or len(ref['indices']) != 8:
            raise ValueError('Eight chronological distinct candidates required')
        for index in ref['indices']:
            path = folder/'source_frames'/f'{index}.png'
            digest = sha256(path)
            if digest != metadata['source_frame_hashes'][str(index)]:
                raise ValueError('Source frame changed')
            frames.append({'path': str(path), 'index': index, 'sha256': digest})
        row = dict(ref, frames=frames, fps=24, window=n, split='diagnostic',
                   source_sha256=source['source_sha256'], source_id=source['source_id'])
        if len({f['id'] for f in row['facts']}) != len(row['facts']):
            raise ValueError('Duplicate fact id')
        for fact in row['facts']:
            if fact['kind'] not in ('event','state') or any(not 0 <= i < 8 for g in fact['evidence_groups'] for i in g):
                raise ValueError('Invalid fact evidence')
        row['oracle'] = oracle(row)
        row['unrestricted_oracle'] = oracle(row, endpoints=False)
        rows.append(row)
    checkpoint = checkpoint_path(CHECKPOINT)
    config = read_json(CONFIG)
    protocol = {'reference_sha256': sha256(REFERENCE), 'reference': refs,
        'checkpoint': str(checkpoint), 'checkpoint_manifest_sha256': sha256(checkpoint/'complete.json'),
        'config': config, 'prior_source_split_sha256': sha256(prior_path), 'exclusion_files': exclusion_files,
        'sources': list(checked.values()), 'rows': rows,
        'cells': ['learned_base','learned_joint','oracle_base','oracle_joint'],
        'generation': 'One frozen vision encoding per row; same NF4 Qwen, BF16, FP32 nonvisual norms; adapter toggled; deterministic decoding; no labels in prompt.',
        'input_matching': 'Same selected keys give identical visual and text input. Different keysets have identical image count/grid and max token budgets; literal frame/time labels can differ in text token count.',
        'precision': 'NF4; BF16; nonvisual norms FP32', 'training_performed': False,
        'review_status': 'Provisional non-independent source-grounded AI review only',
        'implementation': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
            ROOT/'src/semantic_transmission/qwen35_ablation.py', ROOT/'src/semantic_transmission/qwen35_training.py',
            ROOT/'src/semantic_transmission/qwen35_multitask.py', ROOT/'scripts/run_qwen35_oracle_diagnostic.sh']}}
    write_json(run/'protocol.json', protocol)
    write_json(run/'reference_freeze.json', {'protocol_sha256': sha256(run/'protocol.json'),
        'frozen_before_generation': True, 'windows': len(rows), 'source_count': len(checked)})
    verify(run)
    print('REFERENCE_FROZEN', len(rows), 'windows', len(checked), 'sources', flush=True)


def verify(run):
    freeze = read_json(run/'reference_freeze.json')
    if freeze['protocol_sha256'] != sha256(run/'protocol.json'):
        raise ValueError('Protocol changed after reference freeze')
    p = read_json(run/'protocol.json')
    if p['reference_sha256'] != sha256(REFERENCE):
        raise ValueError('Reference changed after freeze')
    if sha256(Path(p['checkpoint'])/'complete.json') != p['checkpoint_manifest_sha256']:
        raise ValueError('Checkpoint manifest changed')
    for name, digest in read_json(Path(p['checkpoint'])/'complete.json')['files'].items():
        if sha256(Path(p['checkpoint'])/name) != digest:
            raise ValueError(f'Checkpoint changed: {name}')
    for name, digest in p['implementation'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Frozen implementation changed: {name}')
    for source in p['sources']:
        if sha256(source['source_path']) != source['source_sha256'] or sha256(source['normalized_path']) != source['normalized_sha256']:
            raise ValueError('Source video changed')
    for row in p['rows']:
        for frame in row['frames']:
            if sha256(frame['path']) != frame['sha256']:
                raise ValueError('Source image changed')
        if row['oracle'] != oracle(row):
            raise ValueError('Oracle selection does not reproduce')
    return p


def generate(run):
    import torch
    from .qwen35_ablation import encode, generate as caption, load_inference
    from .qwen35_training import release_gpu
    p = verify(run)
    config = p['config']
    pending = [row for row in p['rows'] if not (run/'records'/f'{row["id"]}.json').exists()]
    if pending:
        start = time.monotonic()
        model, processor, head, runtime = load_inference(config, Path(p['checkpoint']))
        load_seconds = time.monotonic()-start
        torch.cuda.reset_peak_memory_stats()
        for row in pending:
            begin = time.monotonic()
            features = encode(model, processor, row, config)
            torch.cuda.synchronize()
            encode_seconds = time.monotonic()-begin
            with torch.inference_mode():
                prediction = head(features['pooled'], config['temperature'])
            choices = {'learned': prediction['indices'].tolist(), 'oracle': row['oracle']['keys']}
            order = [(s,c) for s in choices for c in ('base','joint')]
            if row['window'] % 2:
                order.reverse()
            cells, cache = {}, {}
            for selector, captioner in order:
                keys = choices[selector]
                identity = (tuple(keys), captioner)
                reused = identity in cache
                if not reused:
                    cache[identity] = caption(model, processor, row, keys, features, config, adapter=captioner=='joint')
                name = f'{selector}_{captioner}'
                cells[name] = dict(cache[identity], keys=keys, coverage=coverage(row,keys), identical_input_reuse=reused,
                    source_hashes=[row['frames'][i]['sha256'] for i in keys])
                print(row['id'], name, cells[name]['status'], keys, flush=True)
            for selector in choices:
                if cells[f'{selector}_base']['input_tokens'] != cells[f'{selector}_joint']['input_tokens']:
                    raise ValueError('Same-key captioners have mismatched input tokens')
            write_json(run/'records'/f'{row["id"]}.json', {'id':row['id'], 'cells':cells,
                'selector_importance':prediction['importance'].tolist(), 'encode_seconds':encode_seconds,
                'seconds':time.monotonic()-begin, 'model_load_seconds':load_seconds,
                'max_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                'runtime':runtime, 'protocol_sha256':sha256(run/'protocol.json')})
            del features,prediction
        del model,processor,head
        release_gpu()
    files = {f'records/{r["id"]}.json':sha256(run/'records'/f'{r["id"]}.json') for r in p['rows']}
    for name in files:
        record = read_json(run/name)
        if record['protocol_sha256'] != sha256(run/'protocol.json'):
            raise ValueError('Mismatched generation record')
        if set(record['cells']) != set(p['cells']) or any(c['status'] != 'COMPLETE' for c in record['cells'].values()):
            raise ValueError('Incomplete generation')
    write_json(run/'generation_freeze.json', {'protocol_sha256':sha256(run/'protocol.json'), 'files':files})
    write_json(run/'GENERATION_RESULT.json', {'status':'COMPLETE_REVIEW_PENDING','windows':len(p['rows']),
        'condition_windows':len(files)*4,'training_performed':False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','generate','verify'])
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.run)
    elif args.command == 'verify':
        print('VERIFIED', len(verify(args.run)['rows']), 'windows')
    else:
        import fcntl
        with ExitStack() as stack:
            for path in ['/tmp/qwen35_multitask_gpu.lock',str(ROOT/'.local/etri_60s_check.lock')]:
                lock=stack.enter_context(open(path,'a'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            generate(args.run)


if __name__ == '__main__':
    main()
