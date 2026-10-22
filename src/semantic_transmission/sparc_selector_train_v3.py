"""Budget-conditioned imitation training with real RD validation and exact resume."""
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import sparc_selector_train as supervision
from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_workers import REPO
from .sparc_selector_budget import (ARCHITECTURE, METRICS, BudgetSelector, contexts,
    budget_inputs, predict, quality_cost, order_key, cbr, search)
from . import sparc_selector_v3_data as data

CODE = ['src/semantic_transmission/'+name+'.py' for name in
        ('sparc_selector_budget', 'sparc_selector_v3_data', 'sparc_selector_train_v3', 'sparc_selector_dispatch')]


def allocate_cohort(clips, target_cbr):
    """Exact Pareto dynamic program for non-additive measured per-video costs.

    The paper target is an aggregate budget; hard videos may exceed its per-video
    share. No per-frame cost summation or unmeasured padding estimate is used.
    """
    budget = math.floor(target_cbr*3*sum(c['pixels'] for c in clips))
    frontier = [(0, 0., [])]
    for clip in clips:
        choices = sorted(clip['observations'], key=lambda o: o['channel']['complex_channel_uses'])
        compact, lowest = [], float('inf')
        for obs in choices:
            cost = quality_cost(obs, clip['baseline']) * clip['pixels']
            if cost < lowest:
                compact.append((obs['channel']['complex_channel_uses'], cost, obs))
                lowest = cost
        expanded = sorted(((r+rate, q+cost, picked+[obs]) for r, q, picked in frontier
                          for rate, cost, obs in compact if r+rate <= budget), key=lambda item:item[:2])
        frontier, best = [], float('inf')
        for rate, cost, picked in expanded:
            if cost < best:
                frontier.append((rate, cost, picked))
                best = cost
        if not frontier:
            minimum_uses = sum(min(o['channel']['complex_channel_uses'] for o in c['observations']) for c in clips)
            return dict(feasible=False, status='NO_FEASIBLE_COHORT_FOUND', budget_uses=budget,
                        minimum_observed_uses=minimum_uses,
                        excess_fraction=max(0., minimum_uses/budget-1), observed_clips=len(clips))
    rate, cost, picked = min(frontier, key=lambda value: (value[1], value[0]))
    weight = sum(c['pixels'] for c in clips)
    return dict(feasible=True, status='MEASURED_WITHIN_AGGREGATE_BUDGET', budget_uses=budget,
        channel_uses=rate, cbr=rate/(3*weight), quality_cost=cost/weight,
        psnr_delta=sum((o['quality']['psnr_db']-c['baseline']['quality']['psnr_db'])*c['pixels']
                       for c, o in zip(clips, picked))/weight,
        lpips_delta=sum((o['quality']['lpips_vgg']-c['baseline']['quality']['lpips_vgg'])*c['pixels']
                        for c, o in zip(clips, picked))/weight,
        selected=[dict(clip=c['id'], indices=o['identity']['indices'],
                       channel_uses=o['channel']['complex_channel_uses'], quality=o['quality'])
                  for c, o in zip(clips, picked)])


def evaluate(root, parent, rows, p, checkpoint, split, label, progress):
    """Fixed source-only candidate policy, then exact codec/channel/receiver checks."""
    path = root/'evaluation'/split/label/'result.json'
    identity = dict(checkpoint_sha256=sha(checkpoint), protocol=p['signature'], split=split,
                    rows=digest(rows))
    if path.exists():
        result = read(path)
        if result['identity'] != identity:
            raise ValueError('RD validation identity changed')
        return result
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = BudgetSelector(**p['model']).to(p['train']['device']).eval()
    model.load_state_dict(state['model'])
    chosen = data.sample_parents(rows, split, p['search'][split.replace('val','validation')+'_clips'], p['seed'])
    if not chosen:
        raise ValueError('no held-out evaluation parents')
    clips = []
    try:
        for i, row in enumerate(chosen):
            supervision.interrupted()
            saved = path.parent/(row['id']+'.json')
            if saved.exists():
                value = read(saved)
                if value['identity'] != identity:
                    raise ValueError('partial RD evidence changed')
                clips.append(value['clip'])
                continue
            progress('rd_'+split, i, len(chosen), clip=row['id'], checkpoint=str(checkpoint))
            folder = root/'clips'/row['id']
            folder.mkdir(parents=True, exist_ok=True)
            # Test inputs are first accessed here, after final model choice.
            if (not (folder/'features.npz').exists()
                    or split == 'test' and not (folder/'test_features.json').exists()):
                frames, ready = data.load_source(parent/'clips'/row['id'])
                extractor = data.Extractor(p['train']['device'])
                if extractor.identity != p['features']:
                    raise ValueError('test feature extractor changed')
                data.atomic_npz(folder/'features.npz', **extractor.extract(frames))
                save(folder/'prepared.json', ready)
                save(folder/'test_features.json', dict(protocol=p['signature'], sha256=sha(folder/'features.npz')))
                del extractor, frames
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if split == 'test' and read(folder/'test_features.json')['sha256'] != sha(folder/'features.npz'):
                raise ValueError('test feature cache changed')
            with np.load(folder/'features.npz') as z:
                features = (torch.from_numpy(z['patches'].copy()), torch.from_numpy(z['motion'].copy()))
                proxies = z['proxies'].copy()
            measure = data.Measurer(root, parent, row, p,
                lambda calls: progress('rd_'+split, i, len(chosen), clip=row['id'], candidate_measurements=calls))
            try:
                def rank(states, baseline):
                    out = []
                    with torch.inference_mode():
                        for start in range(0, len(states), 4):
                            ys, uncertainty, score = predict(model, features, states[start:start+4],
                                measure.ready, p['target_cbr'], cbr(baseline, measure.pixels),
                                state['normalization'], p['train']['device'])
                            # Penalize predicted budget excess and high rate uncertainty.
                            score = score-4*F.relu(ys[:,4]/p['target_cbr']-1)-.1*uncertainty[:,4]
                            out.extend(score.cpu().tolist())
                    return out
                decision, observations, baseline = search(proxies, measure.ready['keys'], measure.ready['cuts'],
                    math.floor(p['target_cbr']*3*measure.pixels), measure, rank=rank,
                    rounds=p['search']['rounds'], beam=p['search']['beam'],
                    per_round=p['search']['per_round'], stop=supervision.interrupted)
                clip = dict(id=row['id'], parent=row['parent'], pixels=measure.pixels,
                            baseline=baseline, observations=observations, decision=decision)
                save(saved, dict(identity=identity, clip=clip))
                clips.append(clip)
            finally:
                measure.close()
        cohort = allocate_cohort(clips, p['target_cbr'])
        goal = p['quality_goal']
        passes = (cohort['feasible'] and cohort['psnr_delta'] >= -goal['maximum_psnr_drop_db']
                  and cohort['lpips_delta'] <= goal['maximum_lpips_increase'])
        if cohort['feasible']:
            violations = (max(0., -cohort['psnr_delta']-goal['maximum_psnr_drop_db'])
                          +100*max(0., cohort['lpips_delta']-goal['maximum_lpips_increase']))
            selection_score = cohort['quality_cost']+100*violations
        else:
            selection_score = 1e6+1e3*cohort['excess_fraction']
        result = dict(identity=identity, cohort=cohort, target_met=passes,
            per_video_feasible=sum(c['decision']['feasible'] for c in clips), clips=len(clips),
            selection_score=selection_score,
            test_targets_used_for_training=False, teacher_candidates_used_for_evaluation=False)
        save(path, result)
        progress('rd_'+split, len(chosen), len(chosen), target_met=passes)
        return result
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def batch_loss(model, corpus, row, ids, budget, device):
    observations, targets, pixels = corpus.records[row['id']]
    selected = [observations[int(i)] for i in ids]
    ready = corpus.prepared[row['id']]
    features = [x.to(device) for x in corpus.features(row)]
    c, g, scenes = contexts(len(features[0]), [o['identity']['indices'] for o in selected],
        ready['keys'], ready['cuts'], ready['source']['video']['fps'], device)
    baseline = min(observations, key=lambda o: o['quality']['mse'])
    b = budget_inputs(len(selected), budget, cbr(baseline, pixels), device)
    prediction, log_var, score = model(*features, c, g, scenes, b)
    normalized = (targets[ids]-corpus.normalization['mean'])/corpus.normalization['std']
    residual = prediction-normalized.to(device)
    regression = F.smooth_l1_loss(prediction, normalized.to(device))
    calibration = .5*(residual.detach().square()*(-log_var).exp()+log_var).mean()
    # Demonstrations rank complete, actually measured selections at the requested budget.
    winner = min(range(len(selected)), key=lambda i: order_key(selected[i], baseline, budget*3*pixels))
    imitation = F.cross_entropy(score[None], torch.tensor([winner], device=device))
    return regression+.05*calibration+.5*imitation


def train(root, parent, rows, p, progress, *, max_steps=None, validate=evaluate,
          initial_checkpoint=None, initial_validation=None):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed'])
    cfg, device = p['train'], p['train']['device']
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(p['seed'])
    corpus = data.Corpus(root, rows, p)
    initial = None
    if initial_checkpoint is not None:
        identity = p['initialization']
        if sha(initial_checkpoint) != identity['sha256']:
            raise ValueError('initial checkpoint checksum changed')
        initial = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
        if (initial['architecture'] != ARCHITECTURE or initial['model_config'] != p['model']
                or initial['features'] != p['features'] or initial['protocol'] != identity['protocol']):
            raise ValueError('initial checkpoint architecture/features/protocol mismatch')
        # The pretrained regression head predicts in these coordinates. Keep them
        # fixed when adding a new corpus instead of silently changing its outputs.
        corpus.normalization = {k: v.clone() for k, v in initial['normalization'].items()}
        if (set(corpus.normalization) != {'mean', 'std'}
                or any(v.shape != (5,) or not torch.isfinite(v).all() for v in corpus.normalization.values())
                or not (corpus.normalization['std'] > 0).all()):
            raise ValueError('invalid initial target normalization')
    model = BudgetSelector(**p['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    def factor(e):
        if e < cfg['warmup_epochs']:
            return (e+1)/cfg['warmup_epochs']
        t = (e-cfg['warmup_epochs'])/max(1,cfg['max_epochs']-cfg['warmup_epochs'])
        return .05+.95*.5*(1+math.cos(math.pi*min(1,t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    generator = torch.Generator().manual_seed(p['seed']+1)
    training = corpus.rows['train']
    counts = Counter(r['parent'] for r in training)
    domains = {}
    for row in training:
        domains.setdefault(row.get('dataset','default'), set()).add(row['parent'])
    weights = torch.tensor([1/(len(domains[r.get('dataset','default')])*counts[r['parent']])
                            for r in training], dtype=torch.double)
    draws = max(cfg['accumulate'], math.ceil(sum(len(corpus.records[r['id']][0]) for r in training)/cfg['states_per_clip']))
    folder = root/'training'; folder.mkdir(exist_ok=True)
    latest = folder/'last.pt'
    epoch = position = steps = bad = 0
    best = float('inf'); order = None
    if latest.exists():
        s = torch.load(latest, map_location='cpu', weights_only=False)
        if s['protocol'] != p['signature'] or s['data_hash'] != corpus.data_hash or s['device'] != device:
            raise ValueError('v3 exact-resume identity changed')
        if any(not torch.equal(s['normalization'][k], v) for k, v in corpus.normalization.items()):
            raise ValueError('train-only normalization changed')
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        epoch, position, steps, bad, best, order = (s[k] for k in ('epoch','position','steps','bad','best','order'))
        random.setstate(s['python_rng']); np.random.set_state(s['numpy_rng'])
        torch.set_rng_state(s['torch_rng']); generator.set_state(s['sampler_rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all(s['cuda_rng'])
    elif initial is not None:
        model.load_state_dict(initial['model'], strict=True)
        if initial_validation is None or not math.isfinite(initial_validation['selection_score']):
            raise ValueError('warm start requires measured initial validation')
        best = initial_validation['selection_score']
    def export():
        return dict(architecture=ARCHITECTURE, model_config=p['model'], model=model.state_dict(),
            normalization=corpus.normalization, protocol=p['signature'], data_hash=corpus.data_hash,
            epoch=epoch, steps=steps, device=device, features=p['features'], target_cbr=p['target_cbr'])
    def persist():
        supervision.checkpoint(latest, dict(export(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            position=position, bad=bad, best=best, order=order, python_rng=random.getstate(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), sampler_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []))
    save(folder/'dataset.json', dict(data_hash=corpus.data_hash, test_targets_loaded=False,
        parents={s:len(v) for s,v in corpus.parents.items()}, train_clips=len(training),
        model_parameters=sum(v.numel() for v in model.parameters()), frozen_features=p['features'],
        best_checkpoint_criterion='actual held-out aggregate-budget reconstruction cost'))
    if initial is not None and not latest.exists():
        supervision.checkpoint(folder/'best.pt', dict(export(), validation=initial_validation,
            initialization=p['initialization'], retained_parent=True))
        persist()
    while epoch < cfg['max_epochs'] and bad < cfg['patience']:
        if order is None:
            order = []
            for index in torch.multinomial(weights, draws, replacement=True, generator=generator).tolist():
                row = training[index]; observations, _, pixels = corpus.records[row['id']]
                budget = p['training_budgets'][int(torch.randint(len(p['training_budgets']), (), generator=generator))]
                ids = torch.randperm(len(observations), generator=generator)[:cfg['states_per_clip']].tolist()
                baseline = min(observations, key=lambda o:o['quality']['mse'])
                winner = min(range(len(observations)), key=lambda i:order_key(observations[i],baseline,budget*3*pixels))
                if winner not in ids:
                    ids[-1] = winner
                order.append((index, ids, budget))
            position = 0
        model.train()
        while position < len(order):
            if supervision.STOP:
                persist(); return dict(status='INTERRUPTED', steps=steps)
            optimizer.zero_grad(set_to_none=True)
            end = min(len(order), position+cfg['accumulate']); loss_value = 0.
            for index, ids, budget in order[position:end]:
                loss = batch_loss(model, corpus, training[index], torch.tensor(ids), budget, device)/(end-position)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite v3 training loss')
                loss.backward(); loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); steps += 1; position = end
            progress('train_v3', epoch, cfg['max_epochs'], epoch=epoch+1, step=steps,
                     batch_done=position, batch_total=len(order), loss=loss_value,
                     resume='model/optimizer/scheduler/RNG/sampled order and budget restored together')
            if steps%10 == 0 or supervision.STOP or max_steps and steps >= max_steps:
                persist()
            if supervision.STOP or max_steps and steps >= max_steps:
                return dict(status='INTERRUPTED', steps=steps)
        if epoch == 0 or (epoch+1)%cfg['validation_every'] == 0 or epoch+1 == cfg['max_epochs']:
            model.eval()
            candidate = folder/'candidate.pt'; supervision.checkpoint(candidate, export())
            # Release optimizer/model GPU storage while the frozen codec/RIFE workers run.
            model.to('cpu')
            for entry in optimizer.state.values():
                for k, value in entry.items():
                    if torch.is_tensor(value):
                        entry[k] = value.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            try:
                actual = validate(root, parent, rows, p, candidate, 'val', f'epoch_{epoch+1:03d}', progress)
            except BaseException:
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                if device.startswith('cuda'):
                    torch.cuda.set_rng_state_all(rng[3])
                model.to(device)
                for entry in optimizer.state.values():
                    for k, value in entry.items():
                        if torch.is_tensor(value) and k != 'step':
                            entry[k] = value.to(device)
                persist()
                raise
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if device.startswith('cuda'):
                torch.cuda.set_rng_state_all(rng[3])
            model.to(device)
            for entry in optimizer.state.values():
                for k, value in entry.items():
                    if torch.is_tensor(value) and k != 'step':
                        entry[k] = value.to(device)
            value = actual['selection_score']
            improved = value < best-1e-5
            if improved:
                best, bad = value, 0
                supervision.checkpoint(folder/'best.pt', dict(export(), validation=actual))
            else:
                bad += 1
            with (folder/'history.jsonl').open('a') as log:
                log.write(json.dumps(dict(epoch=epoch+1, steps=steps, measured_validation=actual, improved=improved))+'\n')
        epoch += 1; scheduler.step(); order = None; position = 0; persist()
    return dict(status='TRAINED_REQUIRES_FINAL_TEST', checkpoint=str(folder/'best.pt'),
                checkpoint_sha256=sha(folder/'best.pt'), steps=steps, epochs=epoch)


def pipeline(root, parent, old, rows, config):
    """Called with the distributed parent and GPU locks already held."""
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import fcntl
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def progress(stage, done=None, total=None, **extra):
            supervision.status(root, stage, done, total, **extra)
            supervision.status(parent, stage, done, total, v3_output=str(root), **extra)
        progress('initialize_v3', resume='v2 labels preserved; new optimizer and model')
        # Build the feature identity without loading a second CUDA model.
        if (config['schema'] != 'sparc_budget_training_v3' or config['target_cbr'] <= 0
                or config['budget_scope'] != 'aggregate_rgb_source_samples'
                or config['automatic_default_replacement']):
            raise ValueError('invalid v3 training configuration')
        if sha(parent/'sources.json') != old['supervision_protocol']['source_manifest_sha256']:
            raise ValueError('parent source manifest changed')
        for name, expected in old['code'].items():
            if sha(REPO/name) != expected:
                raise ValueError('frozen measurement dependency changed')
        weights = data.backbone_weights()
        p = copy.deepcopy(config)
        p.update(architecture=ARCHITECTURE, parent=str(parent), supervision_protocol=old['supervision_protocol'],
            source_manifest_sha256=sha(parent/'sources.json'), code={f:sha(REPO/f) for f in CODE},
            features=dict(model='torchvision_resnet50_imagenet1k_v2_layer3', weights_sha256=sha(weights),
                          input_hw=[192,336], grid=[3,5], motion='frozen_source_features_v1'))
        p['signature'] = digest(p)
        if (root/'protocol.json').exists() and read(root/'protocol.json') != p:
            raise ValueError('v3 frozen training protocol changed; new output required')
        save(root/'protocol.json', p)
        save(root/'sources.json', rows)
        try:
            data.prepare(root, parent, rows, p, progress)
            data.teacher(root, parent, rows, p, progress)
            supervision.interrupted()
            trained = train(root, parent, rows, p, progress)
            if trained['status'] == 'INTERRUPTED':
                status = read(root/'status.json')
                status.update(status='INTERRUPTED', result=trained, updated_unix=time.time())
                save(root/'status.json', status)
                return trained
            tested = evaluate(root, parent, rows, p, Path(trained['checkpoint']), 'test', 'final', progress)
            result = dict(trained, status='TRAINED_RD_EVALUATED', heldout=tested,
                target_met=tested['target_met'], automatic_default_replacement=False,
                performance_improvement_proven=tested['target_met'],
                evidence_scope='fixed held-out parent cohort, AWGN10dB seed42; not full benchmark')
            save(root/'RESULT.json', result)
            progress('complete', result=result)
            status = read(root/'status.json'); status['status'] = result['status']; save(root/'status.json', status)
            return result
        except BaseException as error:
            current = read(root/'status.json') if (root/'status.json').exists() else {}
            current.update(status='INTERRUPTED' if supervision.STOP else 'FAILED', error=repr(error), updated_unix=time.time())
            save(root/'status.json', current)
            raise
