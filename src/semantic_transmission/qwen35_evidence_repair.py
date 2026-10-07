"""Opt-in LoRA training with complete evidence paths and reviewed contrasts.

prepare/verify/dry-run/audit-inputs never load Qwen weights or invoke backward.
Only the explicit `train ARM` command can create an optimizer or checkpoint.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import math
from pathlib import Path
import random
import time

from .qwen35_evidence_repair_data import (CONFIG, RUN, VERDICTS, config_at,
    informative_queries, ordered_paths, prepare, verify)
from .qwen35_training_data import ROOT, read_json, write_json, sha256

CHOICES = {'supported': 'S', 'contradicted': 'C', 'insufficient': 'U'}


def evidence_prompt(row, query, phase):
    mapping = ', '.join(f'{letter}=i{key}' for letter, key in zip('ABCD', row['keys']))
    return (f'Use the supplied chronological observations only. Labels: {mapping}. '
            f'Visible claim to locate: {query["text"]}\n'
            f'Required phase {phase+1} of {len(query["phases"])}: {query["phases"][phase]["text"]}\n'
            'Choose one observation showing this phase. Return only A, B, C, or D.')


def judgment_prompt(row, claim):
    return ('Judge the claim using ONLY the supplied observations. '
            'S=supported by visible evidence; C=contradicted by visible evidence; '
            'U=insufficient evidence. An unobserved event is not automatically false. '
            'One still image cannot establish an action or a temporal change. '
            f'Claim: {claim["text"]}\nReturn only S, C, or U.')


def ordered_evidence_loss(logits, paths):
    """Marginal likelihood of COMPLETE ordered paths, not their endpoints."""
    import torch
    if logits.ndim != 2 or logits.shape[1] != 4 or not paths:
        raise ValueError('Expected phase-by-four logits and at least one path')
    if any(len(p) != logits.shape[0] or any(type(i) is not int or i not in range(4) for i in p)
           or any(a >= b for a, b in zip(p, p[1:])) for p in paths):
        raise ValueError('Invalid complete chronological paths')
    logp = logits.float().log_softmax(-1)
    scores = torch.stack([sum(logp[k, i] for k, i in enumerate(p)) for p in sorted(set(paths))])
    return -torch.logsumexp(scores, 0) / logits.shape[0]


def preference_loss(positive_ce, negative_ce, margin):
    """Mean-token likelihood margin: both omission and false claim are reviewed."""
    import torch
    return torch.relu(positive_ce - negative_ce + margin)


def streamed_backward(forwards, objective, scale=1., devices=()):
    """Exact first-order gradient, with only one model graph live at a time.

    Small detached outputs determine the objective gradient. Each model call is
    then replayed with its original RNG state, including checkpoint/dropout RNG.
    This does not detach the resulting gradient from the trainable LoRA.
    """
    import torch
    values, rngs = [], []
    with torch.no_grad():
        for forward in forwards:
            rngs.append((torch.get_rng_state(), {d: torch.cuda.get_rng_state(d) for d in devices}))
            values.append(forward().detach().float())
    leaves = [x.clone().requires_grad_(True) for x in values]
    loss = objective(leaves)
    if loss.numel() != 1 or not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite/non-scalar auxiliary objective')
    grads = torch.autograd.grad(loss * scale, leaves)
    value = float(loss.detach())
    del loss, leaves
    for forward, grad, expected, rng in zip(forwards, grads, values, rngs):
        if not bool(grad.any()):
            continue
        with torch.random.fork_rng(devices=list(devices)):
            torch.set_rng_state(rng[0])
            for device, state in rng[1].items():
                torch.cuda.set_rng_state(state, device)
            actual = forward()
            if not torch.allclose(actual.detach().float(), expected, atol=1e-4, rtol=1e-4):
                raise ValueError('Auxiliary replay changed; do not apply mismatched gradient')
            torch.autograd.backward(actual, grad.to(actual.dtype))
            del actual
    return value


def selected_claims(row, epoch):
    # Equal loss weight for all3 verdicts; rotate any alternatives within class.
    result = []
    for verdict in VERDICTS:
        claims = [c for c in row.get('claims', []) if c['verdict'] == verdict]
        if claims:
            result.append(claims[epoch % len(claims)])
    return result


def cache_manifest(row):
    old = ROOT / config_at()['source_run'] / 'features'
    path = old / f'{row["id"]}.pt'
    meta = read_json(path.with_suffix('.json'))
    if sha256(path) != meta['sha256']:
        raise ValueError('Changed source visual cache')
    return path, meta


def load_features(row):
    import torch
    path, _ = cache_manifest(row)
    features = torch.load(path, map_location='cpu', weights_only=True)
    if features['frame_hashes'] != [f['sha256'] for f in row['frames']]:
        raise ValueError('Feature cache belongs to different source observations')
    return features


def choice_inputs(processor, row, config, prompt, observations):
    from .qwen35_training import engine, images_for, labels_for
    keys = [row['keys'][i] for i in observations]
    inputs = engine().base.preprocess(processor, images_for(row, keys), labels_for(row, keys), prompt, config['max_pixels'])
    if inputs.input_ids.shape[1] > config['max_input_tokens']:
        raise ValueError('Choice query exceeds input token budget')
    return inputs, keys


def choice_logits(model, processor, row, features, config, prompt, observations, choices):
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    inputs, keys = choice_inputs(processor, row, config, prompt, observations)
    if not torch.equal(inputs.image_grid_thw, features['grid'][keys]):
        raise ValueError('Choice query visual grid changed')
    ids = [processor.tokenizer.encode(x, add_special_tokens=False) for x in choices]
    if any(len(x) != 1 for x in ids):
        raise ValueError('Choice labels must be single tokens')
    inputs.pop('pixel_values')
    inputs = inputs.to('cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(**inputs, use_cache=False, logits_to_keep=1,
            mm_encoder_outputs={'image': BaseModelOutputWithPooling(pooler_output=tuple(features['tokens'][keys].to('cuda').unbind(0)))})
    return out.logits[0, -1, [i[0] for i in ids]]


def validate(model, processor, head, rows, config):
    import torch
    from .qwen35_training import caption_loss
    from .qwen35_caption_repair import fixed_prediction
    model.eval()
    ce, paths, verdicts, confusion = [], Counter(), Counter(), Counter()
    by_class = {v: Counter() for v in VERDICTS}; by_size = {}
    with torch.no_grad():
        for row in rows:
            features = load_features(row)
            ce.append(float(caption_loss(model, processor, head, fixed_prediction(row['keys']), features, row, config)))
            for q in informative_queries(row):
                logits = [choice_logits(model, processor, row, features, config, evidence_prompt(row, q, i), list(range(4)), 'ABCD') for i in range(len(q['phases']))]
                pred = tuple(int(x.argmax()) for x in logits)
                targets = ordered_paths(q)
                paths.update(queries=1, complete_hits=int(pred in targets),
                    middle_required_queries=int(all(any(i in (1, 2) for i in p) for p in targets)))
            for claim in row.get('claims', []):
                logits = choice_logits(model, processor, row, features, config, judgment_prompt(row, claim), claim['observations'], 'SCU')
                predicted = VERDICTS[int(logits.argmax())]
                verdicts.update(queries=1, hits=int(predicted == claim['verdict']))
                by_class[claim['verdict']].update(queries=1, hits=int(predicted == claim['verdict']))
                size = str(len(claim['observations'])); by_size.setdefault(size, Counter())
                by_size[size].update(queries=1, hits=int(predicted == claim['verdict']))
                confusion[f'{claim["verdict"]}->{predicted}'] += 1
            del features
    model.train()
    model.base_model.model.model.visual.eval()
    return dict(caption_ce=sum(ce)/len(ce), evidence=paths, judgment=verdicts,
        judgment_confusion=confusion, judgment_by_class=by_class, judgment_by_input_size=by_size,
        scope='Teacher-forced development diagnostics, NOT free-caption semantic quality')


def dry_run():
    p = prepare()
    settings = p['settings']
    rows = {s: read_json(RUN / f'{s}.json')['records'] for s in ('train', 'validation')}
    manifests = {}
    for rr in rows.values():
        for row in rr:
            path, meta = cache_manifest(row)
            manifests[row['id']] = dict(path=str(path), sha256=meta['sha256'])
    queries = Counter()
    for epoch in range(settings['epochs']):
        for index, row in enumerate(rows['train']):
            available = informative_queries(row)
            for q in available:
                queries.update(evidence_queries=1, evidence_phases=len(q['phases']))
            queries.update({c['verdict']: 1 for c in selected_claims(row, epoch)})
            if row.get('preference_pairs'):
                queries['preference_pairs'] += 1
    report = dict(status='PASS_STATIC_NO_TRAINING', optimizer_steps=0, model_loaded=False,
        real_model_forward_calls=0, real_model_backward_calls=0,
        checkpoint_saved=False, arms=settings['arms'], planned_auxiliary_schedule=queries,
        updates_per_arm=settings['epochs'] * math.ceil(len(rows['train']) / settings['gradient_accumulation']),
        cache_manifest=manifests, freeze_sha256=sha256(RUN / 'freeze.json'),
        implementation_sha256=sha256(Path(__file__)), dependencies=training_dependencies(),
        gpu_training_validated=False, free_caption_quality_evaluated=False)
    write_json(RUN / 'dry_run.json', report)
    return report


def audit_inputs():
    """Actual CPU processor/tokenizer checks, no model weights or CUDA calls."""
    import torch
    from .qwen35_training import engine, images_for, labels_for
    p = verify()
    config = p['config']
    processor = engine().base.load_processor()
    rows = [r for s in ('train', 'validation') for r in read_json(RUN / f'{s}.json')['records']]
    max_tokens, checked, frame_counts = 0, 0, Counter()
    for label in 'ABCDSCU':
        if len(processor.tokenizer.encode(label, add_special_tokens=False)) != 1:
            raise ValueError('Non-single-token choice')
    for row in rows:
        features = load_features(row)
        targets = [row['caption']] + [pair['rejected'] for pair in row.get('preference_pairs', [])]
        images = images_for(row, row['keys'])
        messages = engine().base.messages_for(images, labels_for(row, row['keys']), config['caption_prompt'])
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        kwargs = dict(images=images, return_tensors='pt', images_kwargs={'size': {'shortest_edge':4096, 'longest_edge':config['max_pixels']}})
        prefix_ids = processor(text=[prefix], **kwargs).input_ids
        for target in targets:
            inputs = processor(text=[prefix + target + '<|im_end|>'], **kwargs)
            if not torch.equal(inputs.input_ids[:, :prefix_ids.shape[1]], prefix_ids):
                raise ValueError('Changed caption prefix boundary')
            if not torch.equal(inputs.image_grid_thw, features['grid'][row['keys']]):
                raise ValueError('Caption grid differs from fixed feature cache')
            max_tokens = max(max_tokens, inputs.input_ids.shape[1]); checked += 1
        requests = [(evidence_prompt(row, q, i), list(range(4))) for q in informative_queries(row) for i in range(len(q['phases']))]
        requests += [(judgment_prompt(row, c), c['observations']) for c in row.get('claims', [])]
        for prompt, observations in requests:
            inputs, keys = choice_inputs(processor, row, config, prompt, observations)
            if not torch.equal(inputs.image_grid_thw, features['grid'][keys]):
                raise ValueError('Query cache grid mismatch')
            max_tokens = max(max_tokens, inputs.input_ids.shape[1]); checked += 1; frame_counts[str(len(keys))] += 1
        del features
    if max_tokens > config['max_input_tokens']:
        raise ValueError('Supervised example exceeds input budget')
    report = dict(status='PASS_CPU_PROCESSOR_ONLY', examples_checked=checked, max_input_tokens=max_tokens,
        input_limit=config['max_input_tokens'], auxiliary_frame_counts=frame_counts,
        model_loaded=False, optimizer_steps=0, real_model_forward_calls=0, real_model_backward_calls=0,
        implementation_sha256=sha256(Path(__file__)), dependencies=training_dependencies(),
        freeze_sha256=sha256(RUN / 'freeze.json'))
    write_json(RUN / 'input_audit.json', report)
    return report


def training_dependencies():
    names = ['qwen35_evidence_repair_data.py', 'qwen35_training.py', 'qwen35_astra_transfer.py',
             'qwen35_caption_repair.py', 'qwen35_multitask.py', 'qwen35_backend.py']
    return {n: sha256(Path(__file__).with_name(n)) for n in names}


def train(arm):
    import torch
    from .qwen35_astra_transfer import load_model
    from .qwen35_training import (caption_loss, checkpoint_path, save_checkpoint, seed_all, release_gpu)
    from .qwen35_caption_repair import fixed_prediction
    from .qwen35_multitask import TemporalSelector
    p = verify(); settings = p['settings']; config = p['config']
    if arm not in settings['arms']:
        raise ValueError('Unknown comparison arm')
    for name in ('dry_run.json', 'input_audit.json'):
        report = read_json(RUN / name)
        if report['freeze_sha256'] != sha256(RUN / 'freeze.json') or report['optimizer_steps'] != 0:
            raise ValueError('Matching no-training preparation required')
        if report['implementation_sha256'] != sha256(Path(__file__)) or report['dependencies'] != training_dependencies():
            raise ValueError('Preparation code changed; rerun dry-run and audit-inputs without training')
    stage = RUN / arm
    identity = dict(arm=arm, freeze_sha256=sha256(RUN / 'freeze.json'),
        implementation_sha256=sha256(Path(__file__)), dependencies=training_dependencies(),
        parent_complete_sha256=p['parent_complete_sha256'], objectives=settings['arms'][arm],
        settings=settings, selector_frozen=True, test_access=False)
    if (stage / 'protocol.json').exists() and read_json(stage / 'protocol.json') != identity:
        raise ValueError('Training code/settings changed; use a new run')
    if (stage / 'RESULT.json').exists():
        checkpoint_path(stage); print('ALREADY_COMPLETE', arm); return
    write_json(stage / 'protocol.json', identity)
    rows = read_json(RUN / 'train.json')['records']; val = read_json(RUN / 'validation.json')['records']
    parent = Path(p['parent_checkpoint'])
    previous = checkpoint_path(stage, 'latest') if (stage / 'latest.json').exists() else None
    model, processor, runtime = load_model(config, (previous or parent) / 'adapter')
    seed_all(settings['seed'])
    head = TemporalSelector(hidden=config['head_hidden']).cuda().eval()
    original = torch.load(parent / 'head.pt', map_location='cuda', weights_only=True)
    head.load_state_dict(original); head.requires_grad_(False)
    params = [v for v in model.parameters() if v.requires_grad]
    if not params or any(v.requires_grad for v in model.base_model.model.model.visual.parameters()):
        raise ValueError('Expected language LoRA only')
    optimizer = torch.optim.AdamW(params, lr=settings['lr'])
    state = dict(updates=0, next_epoch=0, next_offset=0, history=[], training_seconds=0.,
        runtime=runtime, protocol_sha256=sha256(stage / 'protocol.json'))
    if previous:
        state = read_json(previous / 'state.json')
        optimizer.load_state_dict(torch.load(previous / 'optimizer.pt', map_location='cuda', weights_only=True))
        rng = torch.load(previous / 'rng.pt', map_location='cpu', weights_only=True)
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda']); random.setstate(rng['python'])
    else:
        state['initial_validation'] = validate(model, processor, head, val, config)
        save_checkpoint(stage, head, model, optimizer, state)
    index = {r['id']: i for i, r in enumerate(rows)}
    enabled = set(settings['arms'][arm]); acc = settings['gradient_accumulation']
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['next_epoch'], settings['epochs']):
        order = list(rows); random.Random(settings['seed'] + epoch).shuffle(order)
        values = {name: [] for name in ('caption', 'evidence', 'judgment', 'preference')}
        start = time.monotonic()
        for offset in range(0, len(order), acc):
            batch = order[offset:offset+acc]; optimizer.zero_grad(set_to_none=True)
            for row in batch:
                features = load_features(row)
                def text_loss(text):
                    return caption_loss(model, processor, head, fixed_prediction(row['keys']), features, dict(row, caption=text), config)
                ce = text_loss(row['caption']); values['caption'].append(float(ce.detach()))
                (ce/len(batch)).backward(); del ce
                available = informative_queries(row)
                if 'evidence' in enabled and available:
                    for q in available:
                        calls = [lambda i=i: choice_logits(model, processor, row, features, config, evidence_prompt(row, q, i), list(range(4)), 'ABCD') for i in range(len(q['phases']))]
                        value = streamed_backward(calls, lambda xs: ordered_evidence_loss(torch.stack(xs), ordered_paths(q)), settings['grounding_weight']/len(available)/len(batch), devices=(0,))
                        values['evidence'].append(value)
                claims = selected_claims(row, epoch)
                if 'judgment' in enabled and claims:
                    for claim in claims:
                        logits = choice_logits(model, processor, row, features, config, judgment_prompt(row, claim), claim['observations'], 'SCU')
                        target = torch.tensor([VERDICTS.index(claim['verdict'])], device=logits.device)
                        loss = torch.nn.functional.cross_entropy(logits.float()[None], target)
                        values['judgment'].append(float(loss.detach()))
                        (loss*settings['judgment_weight']/len(claims)/len(batch)).backward(); del loss, logits
                if 'preference' in enabled and row.get('preference_pairs'):
                    pair = row['preference_pairs'][(epoch + index[row['id']]) % len(row['preference_pairs'])]
                    value = streamed_backward([lambda: text_loss(row['caption']), lambda: text_loss(pair['rejected'])],
                        lambda xs: preference_loss(xs[0], xs[1], settings['preference_margin']), settings['preference_weight']/len(batch), devices=(0,))
                    values['preference'].append(value)
                del features
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not bool(torch.isfinite(norm)):
                raise ValueError('Nonfinite gradients; no optimizer update applied')
            optimizer.step(); state['updates'] += 1
            write_json(stage / 'status.json', dict(status='RUNNING', epoch=epoch+1, updates=state['updates']))
            print('TRAIN', arm, epoch+1, state['updates'], flush=True)
        if any(not torch.equal(original[k], v) for k, v in head.state_dict().items()):
            raise ValueError('Frozen selector changed')
        state['training_seconds'] += time.monotonic() - start
        state['next_epoch'] = epoch + 1
        state['history'].append(dict(epoch=epoch+1, losses={k: sum(v)/len(v) if v else None for k, v in values.items()}, validation=validate(model, processor, head, val, config)))
        state['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        save_checkpoint(stage, head, model, optimizer, state, best=epoch+1==settings['epochs'])
    write_json(stage / 'status.json', dict(status='COMPLETE', epochs=settings['epochs'], updates=state['updates']))
    write_json(stage / 'RESULT.json', dict(status='COMPLETE_TRAINING_SEMANTIC_EVALUATION_PENDING', state=state,
        checkpoint=str(checkpoint_path(stage)), default_pipeline_changed=False))
    del model, processor, head, optimizer, params
    release_gpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'verify', 'dry-run', 'audit-inputs'):
        sub.add_parser(name)
    opt = sub.add_parser('train'); opt.add_argument('arm', choices=['caption', 'evidence', 'judgment', 'preference'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(); print('PREPARED_NO_TRAINING')
    elif args.command == 'verify':
        verify(); print('VERIFIED_NO_TRAINING')
    elif args.command == 'dry-run':
        r = dry_run(); print({k: v for k, v in r.items() if k != 'cache_manifest'})
    elif args.command == 'audit-inputs':
        print(audit_inputs())
    else:
        with ExitStack() as stack:
            for name in ('/tmp/qwen35_multitask_gpu.lock', str(ROOT / '.local/etri_60s_check.lock')):
                lock = stack.enter_context(open(name, 'w')); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(args.arm)


if __name__ == '__main__':
    main()
