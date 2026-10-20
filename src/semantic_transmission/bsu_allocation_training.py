"""Train/evaluate/select cached BSU allocations without rerunning LTX.

This is an opt-in research path. A completed training run is not evidence of
better video quality. Development-only smoke checkpoints cannot be deployed
without explicitly allowing development inference.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

import torch

from .artifacts import sha256, write_json
from .adapter_training_state import atomic_torch_save, commit_checkpoint, load_checkpoint
from .bsu_allocation import JointAllocationPredictor, ModelConfig, allocation_loss, choose, feasible_mask
from .bsu_allocation_data import METRICS, collate, import_pilot, load_dataset
from .bsu_allocation_transport import select_verified

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO/'configs/bsu_allocation.json'
VERSION = 'bsu_joint_allocation_predictor_v1'


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'execution.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_config(config, document):
    model = ModelConfig(**config['model'])
    if model.feature_dim != document['feature_dim'] or model.metrics != len(METRICS):
        raise ValueError('model/dataset feature or metric dimensions differ')
    objective, training = config['objective'], config['training']
    if objective['metric_names'] != list(METRICS):
        raise ValueError('metric order differs')
    for name in ('max_steps', 'batch_size', 'save_every', 'validate_every', 'cpu_threads'):
        if type(training[name]) is not int or training[name] <= 0:
            raise ValueError(f'invalid {name}')
    if type(training['warmup_steps']) is not int or training['warmup_steps'] < 0:
        raise ValueError('invalid warmup_steps')
    for name in ('learning_rate', 'gradient_clip'):
        if not math.isfinite(training[name]) or training[name] <= 0:
            raise ValueError(f'invalid {name}')
    if not math.isfinite(training['weight_decay']) or training['weight_decay'] < 0:
        raise ValueError('invalid weight decay')
    for name in ('regression_weight', 'ranking_weight', 'regret_weight', 'temperature'):
        if not math.isfinite(objective[name]) or objective[name] < 0:
            raise ValueError(f'invalid {name}')
    if objective['temperature'] <= 0 or sum(objective[n] for n in
            ('regression_weight', 'ranking_weight', 'regret_weight')) <= 0:
        raise ValueError('invalid loss objective')
    return model


def evaluate_records(model, records, objective, device):
    rows = []
    model.eval()
    with torch.inference_mode():
        for record in records:
            batch = collate([record], objective, device)
            output = model(**batch['inputs'])
            valid = feasible_mask(batch['costs'], batch['inputs']['budget'], batch['inputs']['candidate_mask'])
            index = int(choose(output['scores'], batch['costs'], batch['inputs']['budget'],
                               batch['inputs']['candidate_mask'])[0])
            utility = batch['utility'][0]
            best = float(utility.masked_fill(~valid[0], -torch.inf).max())
            value = float(utility[index])
            rows.append(dict(id=record['row']['id'], source_id=record['row']['source_id'],
                candidate_id=record['row']['candidates'][index]['id'],
                regret=max(0., best-value), utility_gain_over_baseline=value,
                best_measured_feasible_utility=best, best_candidate_selected=best-value <= 1e-6,
                actual_channel_uses=int(batch['costs'][0, index]), budget=record['row']['budget'],
                selected_metrics=record['row']['candidates'][index]['metrics']))
    if not rows:
        raise ValueError('empty evaluation split')
    # Equal ORIGINAL-source weight, even with unequal clips/seeds per source.
    groups = sorted({r['source_id'] for r in rows})
    def mean(name):
        return statistics.mean(statistics.mean(r[name] for r in rows if r['source_id'] == g) for g in groups)
    return dict(rows=rows, sources=len(groups), source_mean_regret=mean('regret'),
        source_mean_utility_gain=mean('utility_gain_over_baseline'),
        source_mean_best_selection_rate=mean('best_candidate_selected'),
        budget_violations=sum(r['actual_channel_uses'] > r['budget'] for r in rows),
        best_is_only_within_evaluated_candidates=True, generalization_proven=False)


def train(dataset, output, config, *, device='cpu', development=False, stop_after=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    document, records = load_dataset(dataset)
    spec = validate_config(config, document)
    if development:
        training = validation = [r for r in records if r['row']['split'] == 'development']
        if not training:
            raise ValueError('development smoke needs development records')
    else:
        training = [r for r in records if r['row']['split'] == 'train']
        validation = [r for r in records if r['row']['split'] == 'validation']
        if not training or not validation:
            raise ValueError('source-disjoint train AND validation splits are required; pilot is development-only')
    # Validate objective scales and the full batch before writing any state.
    collate(training[:1], config['objective'])
    settings = config['training']
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop_after must be a positive completed step')
    torch.set_num_threads(settings['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings['seed'])
    if str(device).startswith('cuda'):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    files = ['bsu_allocation', 'bsu_allocation_data', 'bsu_allocation_transport', 'bsu_allocation_training',
             'packets', 'wire', 'bsu_metadata_compression', 'adapter_training_state']
    contract = dict(version=VERSION, dataset_sha256=sha256(dataset), config=config,
        device=str(device), torch_version=str(torch.__version__), development_only=development,
        code={name: sha256(Path(__file__).with_name(name+'.py')) for name in files})
    signature = fingerprint(contract)
    model = JointAllocationPredictor(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min(1., (step+1)/max(1, settings['warmup_steps'])))
    sampler = torch.Generator().manual_seed(settings['seed']+1)
    groups = {}
    for index, record in enumerate(training):
        groups.setdefault(record['row']['source_id'], []).append(index)
    group_rows = list(groups.values())
    state = dict(step=0, history=[], validation=[], best_regret=None, active_seconds=0.)
    meta = dict(version=VERSION, model_config=model.specification(), feature_schema=document['feature_schema'],
        pipeline_id=document['pipeline_id'], objective=config['objective'], development_only=development,
        fit_source_ids=sorted({r['row']['source_id'] for r in training+validation}),
        fit_source_hashes=sorted({r['row']['source_sha256'] for r in training+validation}),
        signature=signature, quality_improvement_proven=False)
    stopping = []
    handlers = {}
    start = time.monotonic()
    with run_lock(output):
        protocol = output/'protocol.json'
        if protocol.exists() and read(protocol) != contract:
            raise ValueError('frozen training configuration/data/code changed; use a new output')
        if not protocol.exists():
            if any(p.name not in ('execution.lock', 'run.log') for p in output.iterdir()):
                raise ValueError('output contains unrelated artifacts')
            write_json(protocol, contract)
        old = load_checkpoint(output, signature)
        if old:
            model.load_state_dict(old['model'])
            optimizer.load_state_dict(old['optimizer'])
            scheduler.load_state_dict(old['scheduler'])
            state = old['training']
            sampler.set_state(old['sampling_rng'])
            torch.set_rng_state(old['cpu_rng'])
            if str(device).startswith('cuda'):
                torch.cuda.set_rng_state_all(old['cuda_rng'])
        elapsed_before = state['active_seconds']
        def status(status_name, stage, error=None):
            write_json(output/'status.json', dict(status=status_name, stage=stage, step=state['step'],
                total=settings['max_steps'], active_seconds=elapsed_before+time.monotonic()-start,
                pid=os.getpid(), development_only=development, error=error,
                log=str(output/'run.log'), checkpoint=str(output/'latest.json')))
        def save():
            state['active_seconds'] = elapsed_before+time.monotonic()-start
            value = dict(step=state['step'], model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), training=state, cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                sampling_rng=sampler.get_state(), metadata=meta)
            return commit_checkpoint(output, value, signature)
        def validate():
            status('RUNNING', 'validation')
            result = evaluate_records(model, validation, config['objective'], device)
            result.update(step=state['step'], development_only=development)
            state['validation'].append(result)
            if state['best_regret'] is None or result['source_mean_regret'] < state['best_regret']:
                state['best_regret'] = result['source_mean_regret']
                path = output/'best.pt'
                atomic_torch_save(dict(metadata=dict(meta, trained_steps=state['step']),
                    model=model.state_dict()), path)
                write_json(output/'best.json', dict(path=str(path), sha256=sha256(path),
                    step=state['step'], source_mean_regret=state['best_regret']))
            return result
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, lambda signum, frame: stopping.append(signum))
            if old is None:
                # Initial evaluation is diagnostic only; never export random
                # weights as the best trained deployment checkpoint.
                initial = evaluate_records(model, validation, config['objective'], device)
                initial.update(step=0, development_only=development)
                state['validation'].append(initial)
                save()
            limit = min(settings['max_steps'], stop_after or settings['max_steps'])
            while state['step'] < limit and not stopping:
                status('RUNNING', 'training')
                indices = []
                for _ in range(settings['batch_size']):
                    group = group_rows[int(torch.randint(len(group_rows), (), generator=sampler))]
                    indices.append(group[int(torch.randint(len(group), (), generator=sampler))])
                batch = collate([training[i] for i in indices], config['objective'], device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(**batch['inputs'])
                feasible = feasible_mask(batch['costs'], batch['inputs']['budget'], batch['inputs']['candidate_mask'])
                losses = allocation_loss(prediction, batch['targets'], batch['utility'], feasible,
                    batch['inputs']['candidate_mask'], **{k: config['objective'][k] for k in
                        ('regression_weight', 'ranking_weight', 'regret_weight', 'temperature')})
                if not torch.isfinite(losses['loss']):
                    raise ValueError('nonfinite training loss')
                losses['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                state['step'] += 1
                state['history'].append(dict(step=state['step'], **{k: float(v.detach()) for k,v in losses.items()},
                    gradient_norm=float(norm), sample_ids=[training[i]['row']['id'] for i in indices]))
                if state['step'] % settings['validate_every'] == 0:
                    validate()
                if state['step'] % settings['save_every'] == 0:
                    save()
                print(f'STEP {state["step"]}/{settings["max_steps"]} loss={float(losses["loss"].detach()):.6f}', flush=True)
            if state['step'] and state['validation'][-1]['step'] != state['step']:
                validate()
            save()
            complete = state['step'] >= settings['max_steps']
            result = dict(status='COMPLETE_ALLOCATION_TRAINING' if complete else 'INTERRUPTED_RESUMABLE',
                step=state['step'], development_only=development, parameters=sum(p.numel() for p in model.parameters()),
                validation=state['validation'][-1], best_checkpoint=str(output/'best.json') if state['step'] else None,
                quality_improvement_proven=False, default_adopted=False,
                training_sources=len(groups), validation_sources=len({r['row']['source_id'] for r in validation}))
            write_json(output/'RESULT.json', result)
            status(result['status'], 'complete' if complete else 'interrupted')
            return result
        except BaseException as error:
            # Preserve the last committed whole optimizer step, not a partially
            # executed update. Resume verifies its hash and every input again.
            status('FAILED', 'failed; resume from latest committed step', str(error))
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def load_predictor(checkpoint, document, *, device='cpu', allow_development=False):
    checkpoint = Path(checkpoint)
    receipt = read(checkpoint)
    if sha256(receipt['path']) != receipt['sha256']:
        raise ValueError('predictor checkpoint hash mismatch')
    value = torch.load(receipt['path'], map_location='cpu', weights_only=True)
    meta = value['metadata']
    if meta['version'] != VERSION or meta['trained_steps'] < 1:
        raise ValueError('a trained allocation checkpoint is required')
    if meta['development_only'] and not allow_development:
        raise ValueError('development-only checkpoint: explicitly allow development inference')
    if (meta['feature_schema'] != document['feature_schema'] or meta['pipeline_id'] != document['pipeline_id']
            or meta['model_config']['feature_dim'] != document['feature_dim']):
        raise ValueError('checkpoint feature/pipeline contract differs; retrain for the changed pipeline')
    model = JointAllocationPredictor(ModelConfig(**meta['model_config'])).to(device)
    model.load_state_dict(value['model'])
    return model.eval(), meta


def select(dataset, checkpoint, *, device='cpu', allow_development=False, record_id=None):
    document, records = load_dataset(dataset, require_labels=False)
    model, meta = load_predictor(checkpoint, document, device=device, allow_development=allow_development)
    selected = []
    if record_id is not None:
        records = [r for r in records if r['row']['id'] == record_id]
        if not records:
            raise ValueError('unknown record ID')
    with torch.inference_mode():
        for record in records:
            batch = collate([record], meta['objective'], device, labels=False)
            prediction = model(**batch['inputs'])
            scores = prediction['scores'][0].cpu().tolist()
            row = record['row']
            choice = select_verified(row['candidates'], scores, row['budget'], row['frames'],
                                     row.get('minimum_key_spacing', 1))
            choice.update(id=row['id'], development_only=meta['development_only'])
            selected.append(choice)
    return dict(selections=selected, default_adopted=False, quality_improvement_proven=False)


def status_lines(output):
    output = Path(output)
    status = read(output/'status.json') if (output/'status.json').exists() else {}
    step, total = status.get('step', 0), status.get('total')
    percent = f'{100*step/total:.1f}%' if total else 'unknown'
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True, timeout=3).strip().replace('\n', ' | ')
    except (OSError, subprocess.SubprocessError):
        gpu = 'unavailable'
    return [f'Status      {status.get("status", "NOT_STARTED")}',
        f'Stage       {status.get("stage", "unknown")}',
        f'Steps       {step}/{total or "unknown"} ({percent})',
        f'Elapsed     {status.get("active_seconds", 0):.1f}s active | ETA unknown',
        f'GPU         utilization / used VRAM / total VRAM: {gpu}',
        'Resume      same train command; hashes, optimizer, scheduler, RNG and sampler restored',
        f'Log         {output / "run.log"}', f'Result      {output / "RESULT.json"}',
        f'Failure     {status.get("error") or "none recorded"}']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import-pilot')
    p.add_argument('--pilot', type=Path, default=REPO/'outputs/bsu_key_budget_pilot_20261019')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('audit')
    p.add_argument('--dataset', type=Path, required=True)
    p = sub.add_parser('train')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--device', default='cpu')
    p.add_argument('--development', action='store_true')
    p.add_argument('--stop-after', type=int)
    for name in ('select', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', type=Path, required=True)
        p.add_argument('--checkpoint', type=Path, required=True, help='best.json checkpoint receipt')
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--device', default='cpu')
        p.add_argument('--allow-development', action='store_true')
        if name == 'select':
            p.add_argument('--record-id')
        else:
            p.add_argument('--split', choices=['test', 'validation', 'development'], default='test')
    for name in ('status', 'watch'):
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'import-pilot':
        print(import_pilot(args.pilot, args.output))
    elif args.command == 'audit':
        document, records = load_dataset(args.dataset)
        print(json.dumps(dict(status='VERIFIED_ALLOCATION_DATASET', records=len(records),
            candidates=sum(len(r['costs']) for r in records), feature_dim=document['feature_dim'],
            split_sources={s: len({r['row']['source_id'] for r in records if r['row']['split'] == s})
                for s in sorted({r['row']['split'] for r in records})}), indent=2))
    elif args.command == 'train':
        result = train(args.dataset, args.output, read(args.config), device=args.device,
                       development=args.development, stop_after=args.stop_after)
        print(json.dumps({k: v for k,v in result.items() if k != 'validation'}, indent=2))
    elif args.command in ('status', 'watch'):
        try:
            while True:
                if args.command == 'watch':
                    print('\033[2J\033[H', end='')
                print('\n'.join(status_lines(args.output)), flush=True)
                if args.command != 'watch':
                    break
                time.sleep(3)
        except KeyboardInterrupt:
            pass
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.command == 'select':
            result = select(args.dataset, args.checkpoint, device=args.device,
                allow_development=args.allow_development, record_id=args.record_id)
        else:
            document, records = load_dataset(args.dataset)
            model, meta = load_predictor(args.checkpoint, document, device=args.device,
                                        allow_development=args.allow_development)
            records = [r for r in records if r['row']['split'] == args.split]
            if args.split == 'test' and any(r['row']['source_id'] in meta['fit_source_ids']
                    or r['row']['source_sha256'] in meta['fit_source_hashes'] for r in records):
                raise ValueError('test source overlaps checkpoint training/validation')
            result = evaluate_records(model, records, meta['objective'], args.device)
            result.update(split=args.split, development_only=meta['development_only'])
        write_json(args.output, result)
        print(args.output)


if __name__ == '__main__':
    main()
