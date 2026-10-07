"""Frozen, matched development caption evaluation; no optimizer or backward."""
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path
import time

from .qwen35_training_data import ROOT, read_json, write_json, sha256
from .qwen35_training import checkpoint_path, release_gpu
from .qwen35_evidence_repair_data import RUN as DATA, verify as verify_data
from .qwen35_astra_transfer import load_model
from .qwen35_ablation import encode, generate

RUN = DATA / 'evaluation'
ARMS = ['initial', 'caption', 'evidence', 'judgment', 'preference']


def cell_order(ordinal):
    offset = ordinal % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def input_row(row):
    return {k: row[k] for k in ('id', 'source_id', 'fps', 'frames', 'keys')}


def check_checkpoint(path):
    for name, digest in read_json(path / 'complete.json')['files'].items():
        if sha256(path / name) != digest:
            raise ValueError('Changed completed checkpoint')


def prepare():
    p = verify_data()
    reference = read_json(RUN / 'reference_freeze.json')
    if reference['reference_sha256'] != sha256(RUN / 'reference.json'):
        raise ValueError('Changed source reference')
    rows = [input_row(r) for r in read_json(DATA / 'validation.json')['records']]
    if {r['id'] for r in rows} != {r['id'] for r in read_json(RUN / 'reference.json')['records']}:
        raise ValueError('Incomplete development references')
    checkpoints = {'initial': Path(p['parent_checkpoint'])}
    results = {}
    for arm in ARMS[1:]:
        result = read_json(DATA / arm / 'RESULT.json')
        if result['status'] != 'COMPLETE_TRAINING_SEMANTIC_EVALUATION_PENDING' or result['state']['updates'] != 60 or result['state']['next_epoch'] != 2:
            raise ValueError('Training not complete')
        ck = checkpoint_path(DATA / arm)
        if str(ck) != result['checkpoint']:
            raise ValueError('Completed checkpoint differs from declared result')
        checkpoints[arm] = ck
        results[arm] = sha256(DATA / arm / 'RESULT.json')
    protocol = dict(arms=ARMS, rows=rows, config=p['config'],
        checkpoints={a: dict(path=str(ck), complete_sha256=sha256(ck/'complete.json')) for a, ck in checkpoints.items()},
        training_result_hashes=results, data_freeze_sha256=sha256(DATA/'freeze.json'),
        reference_freeze_sha256=sha256(RUN/'reference_freeze.json'), implementation_sha256=sha256(Path(__file__)),
        dependencies={n: sha256(Path(__file__).with_name(n)) for n in ['qwen35_ablation.py', 'qwen35_astra_transfer.py', 'qwen35_training.py', 'qwen35_evidence_repair_data.py']},
        initial_definition='Previous Astra-caption rank32 Qwen before the current120-window additional training, not raw Qwen or Astra direct output.',
        checkpoint_policy='All four predeclared final epoch2 checkpoints; no selection based on these captions.',
        input_policy='Same four keys, source pixels, timestamps, resolution, prompt and160-token cap. Only selected visual tokens enter caption generation. No reference facts or captions in model input.',
        compute_policy='One backbone, five frozen adapters, rotated call order. Inference only, optimizer/backward=0.',
        scope='Four development originals/16 windows. Previously observed during development. No fresh final test; no selection, channel, transmission, or reconstruction evaluation.',
        review_policy='Source reference frozen before new generations; one Codex reviewer, not independent Astra/human. Identical captions within a window receive one judgment.',
        gate='Fact/action full hits must not fall and error windows must not rise against initial and caption; require at least one strict gain. Development evidence only.')
    path = RUN/'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Changed frozen evaluation protocol')
    write_json(path, protocol)
    return protocol


def verify():
    p = read_json(RUN/'protocol.json')
    if p['implementation_sha256'] != sha256(Path(__file__)) or p['data_freeze_sha256'] != sha256(DATA/'freeze.json'):
        raise ValueError('Changed evaluation implementation/data')
    for name, digest in p['dependencies'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Changed inference dependency')
    if sha256(RUN/'reference_freeze.json') != p['reference_freeze_sha256']:
        raise ValueError('Changed reference freeze')
    f = read_json(RUN/'reference_freeze.json')
    if sha256(RUN/'reference.json') != f['reference_sha256']:
        raise ValueError('Changed source reference')
    for name, digest in f['sheets'].items():
        if sha256(name) != digest:
            raise ValueError('Changed reviewed source sheet')
    for ck in p['checkpoints'].values():
        path = Path(ck['path'])
        if sha256(path/'complete.json') != ck['complete_sha256']:
            raise ValueError('Changed checkpoint manifest')
        check_checkpoint(path)
    for row in p['rows']:
        for frame in row['frames']:
            if sha256(frame['path']) != frame['sha256']:
                raise ValueError('Changed source pixels')
    return p


def validate_record(row, record, protocol_hash):
    if record['id'] != row['id'] or record['protocol_sha256'] != protocol_hash or set(record['cells']) != set(ARMS):
        raise ValueError('Incomplete/wrong generation record')
    first = record['cells']['initial']
    for cell in record['cells'].values():
        if cell['keys'] != row['keys'] or cell['source_hashes'] != [row['frames'][k]['sha256'] for k in row['keys']]:
            raise ValueError('Changed selection/source')
        if cell['input_tokens'] != first['input_tokens'] or cell['grid'] != first['grid']:
            raise ValueError('Unmatched input budget')


def run():
    import torch
    prepare(); p = verify(); ph = sha256(RUN/'protocol.json')
    pending = []
    for row in p['rows']:
        path = RUN/'records'/f'{row["id"]}.json'
        if path.exists():
            if sha256(path) != read_json(path.with_suffix('.receipt.json'))['sha256']:
                raise ValueError('Changed saved output')
            validate_record(row, read_json(path), ph)
        else:
            pending.append(row)
    if pending:
        model, processor, runtime = load_model(p['config'], Path(p['checkpoints']['initial']['path'])/'adapter')
        start = time.monotonic()
        for arm in ARMS[1:]:
            model.load_adapter(str(Path(p['checkpoints'][arm]['path'])/'adapter'), adapter_name=arm, is_trainable=False)
        runtime['additional_adapter_load_seconds'] = time.monotonic()-start
        model.requires_grad_(False); model.gradient_checkpointing_disable(); model.eval()
        torch.cuda.reset_peak_memory_stats()
        for row in pending:
            start = time.monotonic(); features = encode(model, processor, row, p['config'])
            torch.cuda.synchronize(); encoding = time.monotonic()-start
            cells = {}; order = cell_order(p['rows'].index(row))
            for arm in order:
                model.set_adapter('default' if arm == 'initial' else arm)
                model.requires_grad_(False); model.eval()
                if any(v.requires_grad for v in model.parameters()):
                    raise ValueError('Evaluation must be frozen')
                cell = generate(model, processor, row, row['keys'], features, p['config'])
                if cell['input_tokens'] > p['config']['max_input_tokens']:
                    raise ValueError('Exceeded input budget')
                cells[arm] = dict(cell, keys=row['keys'], source_hashes=[row['frames'][k]['sha256'] for k in row['keys']])
                print('EVAL', row['id'], arm, cell['status'], flush=True)
            record = dict(id=row['id'], cells=cells, order=order, encode_seconds=encoding,
                runtime=runtime, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, protocol_sha256=ph)
            validate_record(row, record, ph)
            path = RUN/'records'/f'{row["id"]}.json'; write_json(path, record)
            write_json(path.with_suffix('.receipt.json'), dict(sha256=sha256(path), protocol_sha256=ph))
            del features
        del model, processor
        release_gpu()
    verify()
    files = {}
    for row in p['rows']:
        name = f'records/{row["id"]}.json'
        validate_record(row, read_json(RUN/name), ph); files[name] = sha256(RUN/name)
    write_json(RUN/'generation_freeze.json', dict(protocol_sha256=ph, files=files))
    write_json(RUN/'RESULT.json', dict(status='GENERATED_DEVELOPMENT_REVIEW_PENDING', windows=len(p['rows']),
        condition_windows=len(p['rows'])*len(ARMS), optimizer_steps=0, backward_calls=0, independent_final_test=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'generate'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(); print('PREPARED')
    elif args.command == 'verify':
        verify(); print('VERIFIED')
    else:
        with ExitStack() as stack:
            for name in ['/tmp/qwen35_multitask_gpu.lock', str(ROOT/'.local/etri_60s_check.lock')]:
                lock = stack.enter_context(open(name, 'w')); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run()
