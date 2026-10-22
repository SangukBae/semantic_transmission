"""V4 risk supervision, independent calibration and resumable optimization."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import sparc_selector_budget as budget
from . import sparc_selector_train as supervision
from . import sparc_selector_v3_data as old_data
from . import sparc_selector_v4_data as data
from . import sparc_selector_search_v4 as search
from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_risk import (ARCHITECTURE, RiskSelector, Predictor, physical,
                                  fit_calibration, quality_feasible)


def batch_loss(model, corpus, row, ids, target, device):
    record = corpus.records[row['id']]
    selected = [record['observations'][int(i)] for i in ids]
    features, local = corpus.tensors(row)
    ready = corpus.prepared[row['id']]
    c, g, scenes = budget.contexts(len(features[0]), [o['identity']['indices'] for o in selected],
        ready['keys'], ready['cuts'], ready['source']['video']['fps'], device)
    output = model(*(x.to(device) for x in features), c, g, scenes,
        budget.budget_inputs(len(selected), target, budget.cbr(record['baseline'], record['pixels']), device))
    n = {k:v.to(device) for k,v in corpus.normalization.items()}
    absolute = (record['absolute'][ids].to(device)-n['mean'])/n['std']
    delta = (record['delta'][ids].to(device)-n['delta_mean'])/n['delta_std']
    local = (local[ids].to(device)-n['local_mean'])/n['local_std']
    weights = corpus.p['loss']
    value = weights['absolute']*F.smooth_l1_loss(output['absolute'], absolute)
    value += weights['delta']*F.smooth_l1_loss(output['delta'], delta)
    variance = sum(.5*((output[key]-truth).detach().square()*(-output[var]).exp()+output[var]).mean()
                   for key, var, truth in [('absolute','log_var',absolute),('delta','delta_log_var',delta)])
    value += weights['variance']*variance
    value += weights['violation']*F.binary_cross_entropy_with_logits(output['violation_logit'], record['unsafe'][ids].to(device))
    errors = F.smooth_l1_loss(output['local_log_mse'], local, reduction='none').flatten(1)
    tail = errors.topk(max(1, math.ceil(errors.shape[1]*weights['local_tail_fraction'])), dim=1).values.mean()
    value += weights['local']*(errors.mean()+tail)/2
    eligible = [i for i,o in enumerate(selected) if quality_feasible(o, record['baseline'], corpus.p['quality_goal'])
                and o['channel']['complex_channel_uses'] <= target*3*record['pixels']]
    # An impossible request supervises risk/regression, never a fictitious "safe" winner.
    if eligible:
        winner = min(eligible, key=lambda i: budget.quality_cost(selected[i], record['baseline']))
        value += weights['ranking']*F.cross_entropy(output['score'][None], torch.tensor([winner], device=device))
    return value


@torch.inference_mode()
def calibrate(model, corpus, p, device):
    model.eval()
    records = []
    for row in corpus.rows['calibration']:
        supervision.interrupted()
        record = corpus.records[row['id']]
        features, local = corpus.tensors(row)
        predictor = Predictor(model, features, corpus.prepared[row['id']], corpus.normalization, None, p['quality_goal'], device)
        observations = record['observations']
        for start in range(0, len(observations), 4):
            group = observations[start:start+4]
            values = predictor.values([o['identity']['indices'] for o in group], p['target_cbr'],
                                      budget.cbr(record['baseline'], record['pixels']))
            for i, obs in enumerate(group):
                index = start+i
                records.append(dict(parent=row['parent'], rate=float(values['rate'][i]),
                    rate_log_sigma=float(values['rate_log_sigma'][i]), delta=values['delta'][i].cpu().tolist(),
                    delta_sigma=values['delta_sigma'][i].cpu().tolist(),
                    actual_rate=budget.cbr(obs, record['pixels']), actual_delta=record['delta'][index].tolist(),
                    local_log_underprediction=float((local[index]-values['local_mse'][i].cpu().log()).max())))
    result = fit_calibration(records, p['calibration']['alpha'])
    result.update(data_hash=corpus.data_hash, candidates=len(records))
    return result


def summarize(clips, cohort, goal):
    selected = {o['clip']:o for o in cohort.get('selected', [])}
    metrics = ('psnr_db', 'lpips_vgg', 'mse', 'temporal_mse', 'local_mae95')
    baseline = {key:float(np.mean([c['baseline']['quality'][key] for c in clips])) for key in metrics}
    baseline_rate = sum(c['baseline']['channel']['complex_channel_uses'] for c in clips)
    result = dict(baseline=baseline, baseline_channel_uses=baseline_rate,
        videos=len(clips), selected_videos=len(selected), budget_met=bool(cohort['feasible']),
        aggregation='equal-video quality means; aggregate channel uses / RGB source samples')
    if not cohort['feasible']:
        result.update(per_video_quality_met=False, mean_quality_met=False, quality_failures=None)
        return result
    quality = {key:float(np.mean([selected[c['id']]['quality'][key] for c in clips])) for key in metrics}
    failures = [c['id'] for c in clips if not quality_feasible(selected[c['id']], c['baseline'], goal)]
    result.update(selected=quality, quality_failures=failures, per_video_quality_met=not failures,
        channel_reduction_percent=100*(1-cohort['channel_uses']/baseline_rate),
        worst_psnr_delta=min(selected[c['id']]['quality']['psnr_db']-c['baseline']['quality']['psnr_db'] for c in clips),
        worst_lpips_delta=max(selected[c['id']]['quality']['lpips_vgg']-c['baseline']['quality']['lpips_vgg'] for c in clips),
        mean_quality_met=(quality['psnr_db'] >= baseline['psnr_db']-goal['maximum_psnr_drop_db']
                          and quality['lpips_vgg'] <= baseline['lpips_vgg']+goal['maximum_lpips_increase']))
    return result


def evaluate(root, source_root, rows, p, checkpoint, split, label, progress):
    ids = set(p[{'val':'validation_ids', 'test':'test_ids', 'benchmark':'benchmark_ids'}[split]])
    chosen = [r for r in rows if r['id'] in ids]
    if not chosen or len(chosen) != len(ids):
        raise ValueError('empty or incomplete evaluation cohort')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['architecture'] != ARCHITECTURE or state['protocol'] != p['signature']:
        raise ValueError('v4 evaluation checkpoint identity mismatch')
    if 'calibration' not in state:
        raise ValueError('calibrated checkpoint required')
    calibrated = set(state['calibration']['parents'])
    if calibrated & {r['parent'] for r in chosen}:
        raise ValueError('calibration/evaluation parent leakage')
    output = root/'evaluation'/split/label
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=sha(checkpoint), protocol=p['signature'], split=split,
                    rows=digest(chosen), search=p['search'])
    result_path = output/'result.json'
    if result_path.exists():
        result = read(result_path)
        if result['identity'] != identity:
            raise ValueError('cached v4 evaluation identity changed')
        for clip in result['clips']:
            ready = read(output/'clips'/clip['id']/'prepared.json')
            frames, current_ready = old_data.load_source(source_root/'clips'/clip['id'])
            array_hash = data.source_array_hash(frames)
            if current_ready != ready:
                raise ValueError('cached evaluation source changed')
            for obs in clip['observations']:
                path = output/'clips'/clip['id']/'observations'/digest(obs['identity'])[:24]/'result.json'
                if old_data.verify_observation(path, p['supervision_protocol'], ready, array_hash) != obs:
                    raise ValueError('cached evaluation candidate changed')
        return result
    evaluation_started = time.monotonic()
    model = RiskSelector(**p['model']).to(p['train']['device']).eval()
    model.load_state_dict(state['model'])
    sessions, measures = [], []
    shared = supervision.Evaluator(output, p['supervision_protocol'])
    extractor = None
    try:
        for row in chosen:
            supervision.interrupted()
            measure = old_data.Measurer(output, source_root, row, p)
            measure.evaluator = shared
            measures.append(measure)
            folder = output/'clips'/row['id']
            save(folder/'prepared.json', measure.ready)
            # Features from arbitrary external roots are recomputed from verified source RGB.
            if extractor is None:
                extractor = old_data.Extractor(p['train']['device'])
                if extractor.identity != p['features']:
                    raise ValueError('feature extractor identity mismatch')
            extracted = extractor.extract(measure.frames)
            features = [torch.from_numpy(extracted[k]) for k in ('patches','motion')]
            predictor = Predictor(model, features, measure.ready, state['normalization'], state['calibration'],
                                  p['quality_goal'], p['train']['device'])
            sessions.append(search.Session(row['id'], extracted['proxies'], measure.ready['keys'], measure.ready['cuts'],
                measure.pixels, measure, p['search'], p['quality_goal'], predictor, supervision.interrupted))
        del extractor
        extractor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        begin = time.monotonic()
        result = search.search_cohort(sessions, p['target_cbr'], p['search'], p['quality_goal'],
            lambda **kw: progress('rd_'+split, kw.get('calls'), None, **kw))
        shared.close()
        shared.clients.clear()
        report = summarize(result['clips'], result['cohort'], p['quality_goal'])
        cohort = result['cohort']
        score = (cohort['quality_cost'] if cohort['feasible'] else
                 1e6 + (cohort.get('minimum_observed_safe_uses') or 10*cohort['budget_uses'])/cohort['budget_uses'])
        result.update(identity=identity, summary=report, selection_score=score,
            target_met=report['budget_met'] and report['per_video_quality_met'] and report['mean_quality_met'],
            evaluated_seconds=time.monotonic()-begin, source_reference_used_for_candidate_measurement=True,
            learned_one_pass_selection=False)
        result['search_seconds'] = result['elapsed_seconds']
        result['elapsed_seconds'] = time.monotonic()-evaluation_started
        save(result_path, result)
        return result
    finally:
        shared.close()
        for measure in measures:
            measure.array.unlink(missing_ok=True)
        del extractor, sessions, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train(root, source_root, rows, p, progress, *, max_steps=None, validate=evaluate, initial_checkpoint=None):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed'])
    device, cfg = p['train']['device'], p['train']
    corpus = data.Corpus(root, rows, p)
    model = RiskSelector(**p['model']).to(device)
    if initial_checkpoint:
        initial = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
        initial_config = dict(model.config, **initial['model_config'])
        if initial['architecture'] != budget.ARCHITECTURE or initial_config != model.config:
            raise ValueError('v3 initialization architecture mismatch')
        missing, unexpected = model.load_state_dict(initial['model'], strict=False)
        if unexpected or any(not key.startswith(('delta.', 'delta_log_variance.', 'violation.', 'local_error.')) for key in missing):
            raise ValueError('unexpected v3 initialization weights')
        corpus.normalization.update({k:initial['normalization'][k].clone() for k in ('mean','std')})
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    def factor(epoch):
        if epoch < cfg['warmup_epochs']:
            return (epoch+1)/cfg['warmup_epochs']
        t = (epoch-cfg['warmup_epochs'])/max(1,cfg['max_epochs']-cfg['warmup_epochs'])
        return .05 + .95*.5*(1+math.cos(math.pi*min(1,t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    generator = torch.Generator().manual_seed(p['seed']+1)
    training = corpus.rows['train']
    counts = Counter(r['parent'] for r in training)
    domains = {}
    for row in training:
        domains.setdefault(row.get('dataset','default'), set()).add(row['parent'])
    weights = torch.tensor([1/(len(domains[r.get('dataset','default')])*counts[r['parent']]) for r in training], dtype=torch.double)
    draws = max(cfg['accumulate'], math.ceil(sum(len(corpus.records[r['id']]['observations']) for r in training)/cfg['states_per_clip']))
    folder = root/'training'; folder.mkdir(exist_ok=True)
    latest = folder/'last.pt'
    epoch = position = steps = bad = 0
    best, order = float('inf'), None
    def restore_rng(s):
        random.setstate(s['python_rng']); np.random.set_state(s['numpy_rng'])
        torch.set_rng_state(s['torch_rng']); generator.set_state(s['sampler_rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all(s['cuda_rng'])
    def rng_state():
        return dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    sampler_rng=generator.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
    if latest.exists():
        s = torch.load(latest, map_location='cpu', weights_only=False)
        if s['protocol'] != p['signature'] or s['data_hash'] != corpus.data_hash or s['device'] != device:
            raise ValueError('v4 exact-resume identity changed')
        if any(not torch.equal(s['normalization'][k], v) for k,v in corpus.normalization.items()):
            raise ValueError('training normalization changed')
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        epoch, position, steps, bad, best, order = (s[k] for k in ('epoch','position','steps','bad','best','order'))
        restore_rng(s)
    def export():
        return dict(architecture=ARCHITECTURE, model_config=model.config, model=model.state_dict(), normalization=corpus.normalization,
            protocol=p['signature'], data_hash=corpus.data_hash, epoch=epoch, steps=steps, device=device, features=p['features'])
    def persist():
        supervision.checkpoint(latest, dict(export(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            position=position, bad=bad, best=best, order=order, **rng_state()))
    def move(target):
        model.to(target)
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if torch.is_tensor(value) and key != 'step':
                    entry[key] = value.to(target)
    save(folder/'dataset.json', dict(data_hash=corpus.data_hash, train_clips=len(training),
        calibration_parents=sorted({r['parent'] for r in corpus.rows['calibration']}),
        test_targets_loaded=False, criterion='measured quality-feasible aggregate allocation on separate validation parents'))
    while epoch < cfg['max_epochs'] and bad < cfg['patience']:
        if order is None:
            order = []
            for index in torch.multinomial(weights, draws, replacement=True, generator=generator).tolist():
                row = training[index]; record = corpus.records[row['id']]; observations = record['observations']
                target = p['training_budgets'][int(torch.randint(len(p['training_budgets']), (), generator=generator))]
                ids = torch.randperm(len(observations), generator=generator)[:cfg['states_per_clip']].tolist()
                eligible = [i for i,o in enumerate(observations) if quality_feasible(o,record['baseline'],p['quality_goal'])
                            and o['channel']['complex_channel_uses'] <= target*3*record['pixels']]
                if eligible:
                    winner = min(eligible, key=lambda i:budget.quality_cost(observations[i],record['baseline']))
                    if winner not in ids:
                        ids[-1] = winner
                order.append((index, ids, target))
            position = 0
        model.train()
        while position < len(order):
            if supervision.STOP:
                persist(); return dict(status='INTERRUPTED', steps=steps)
            optimizer.zero_grad(set_to_none=True)
            end = min(len(order), position+cfg['accumulate']); loss_value = 0.
            for index, ids, target in order[position:end]:
                loss = batch_loss(model, corpus, training[index], torch.tensor(ids), target, device)/(end-position)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite v4 training loss')
                loss.backward(); loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); steps += 1; position = end
            progress('train_v4', epoch, cfg['max_epochs'], epoch=epoch+1, step=steps, batch_done=position,
                batch_total=len(order), loss=loss_value, resume='weights/optimizer/scheduler/RNG/sampled order and budgets')
            if steps % 10 == 0 or supervision.STOP or max_steps and steps >= max_steps:
                persist()
            if supervision.STOP or max_steps and steps >= max_steps:
                return dict(status='INTERRUPTED', steps=steps)
        if epoch == 0 or (epoch+1) % cfg['validation_every'] == 0 or epoch+1 == cfg['max_epochs']:
            model.eval(); rng = rng_state()
            try:
                calibration = calibrate(model, corpus, p, device)
                candidate = folder/'candidate.pt'
                supervision.checkpoint(candidate, dict(export(), calibration=calibration))
                move('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                actual = validate(root, source_root, rows, p, candidate, 'val', f'epoch_{epoch+1:03d}', progress)
            except BaseException:
                restore_rng(rng); move(device); persist()
                raise
            restore_rng(rng); move(device)
            improved = actual['selection_score'] < best-1e-5
            if improved:
                best, bad = actual['selection_score'], 0
                supervision.checkpoint(folder/'best.pt', dict(export(), calibration=calibration, validation=actual))
            else:
                bad += 1
            with (folder/'history.jsonl').open('a') as log:
                log.write(json.dumps(dict(epoch=epoch+1, steps=steps, selection_score=actual['selection_score'], improved=improved))+'\n')
        epoch += 1; scheduler.step(); order = None; position = 0; persist()
    return dict(status='TRAINED_REQUIRES_FINAL_TEST', checkpoint=str(folder/'best.pt'),
                checkpoint_sha256=sha(folder/'best.pt'), steps=steps, epochs=epoch)
