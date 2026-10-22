"""Resumable CNN/attention selector training with frozen UF/channel/RIFE labels."""
import argparse
from collections import Counter, OrderedDict
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import sparc_selector_train as supervision
from .sparc_selector_cnn import (FRAME_HW, MODEL_CONFIG, SpatialTemporalSelector,
                                pair_context, thumbnails)
from .sparc_selector_data import digest, read, save, sha, status
from .sparc_selector_workers import REPO

DEFAULT = REPO/'outputs/sparc_selector_cnn_attention_v2_20261020'
PARENT = REPO/'outputs/sparc_selector_v1_20261020'
ARCHITECTURE = 'spatial_temporal_selector_v2'


def protocol(root, parent):
    """Keep measurement provenance independent of the new predictor."""
    old = read(parent/'protocol.json')
    if sha(parent/'sources.json') != old['source_manifest_sha256']:
        raise ValueError('parent source manifest changed')
    for name, expected in old['code'].items():
        if sha(REPO/'src/semantic_transmission'/f'{name}.py') != expected:
            raise ValueError('frozen supervision code changed: '+name)
    for path, expected in old['assets'].items():
        if sha(path) != expected:
            raise ValueError('frozen model changed: '+path)
    from .sparc_uf_transport import UPSTREAM
    import subprocess
    vendor = REPO/'.local/dcvc_uf_vendor'
    if subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM:
        raise ValueError('UF revision changed')
    if subprocess.check_output(['git','-C',str(vendor),'diff','HEAD']):
        raise ValueError('UF source changed')
    files = ['src/semantic_transmission/sparc_selector_cnn.py',
             'src/semantic_transmission/sparc_selector_train_v2.py']
    p = dict(architecture=ARCHITECTURE, model=MODEL_CONFIG, frame_hw=list(FRAME_HW),
        code={f:sha(REPO/f) for f in files}, parent=str(parent),
        supervision_protocol=old, seed=20261021, max_gap=None,
        train=dict(max_epochs=100, patience=12, pairs_per_clip=16, accumulate=4,
                   lr=.0003, weight_decay=.0001, warmup_epochs=3, device='cuda'),
        target_order=old['target_order'], cnn_trainable=True,
        labels_use_actual_channel=True, automatic_default_replacement=False,
        performance_improvement_proven=False)
    p['signature'] = digest(p)
    if (root/'protocol.json').exists() and read(root/'protocol.json') != p:
        raise ValueError('new training protocol changed; use a new run directory')
    save(root/'protocol.json', p)
    return p


def reuse_preparation(root, parent, p):
    """Copy receipts and hardlink immutable image files; never write into parent."""
    sources = read(parent/'sources.json')
    if (root/'sources.json').exists() and read(root/'sources.json') != sources:
        raise ValueError('source manifest changed')
    save(root/'sources.json', sources)
    reused = 0
    for row in sources:
        old, new = parent/'clips'/row['id'], root/'clips'/row['id']
        if not (old/'prepared.json').exists() or (new/'prepared.json').exists():
            continue
        ready = read(old/'prepared.json')
        if ready['input'] != row or ready['protocol'] != p['supervision_protocol']['signature']:
            raise ValueError('parent preparation provenance mismatch')
        new.mkdir(parents=True, exist_ok=True)
        folder = new/'frames'
        folder.mkdir(exist_ok=True)
        for name, expected in ready['source']['hashes'].items():
            source = Path(ready['source']['directory'])/name
            if sha(source) != expected:
                raise ValueError('parent source frame changed')
            target = folder/name
            if not target.exists():
                os.link(source, target)
        ready['source']['directory'] = str(folder)
        save(new/'prepared.json', ready)
        reused += 1
    if not (root/'reuse.json').exists():
        save(root/'reuse.json', dict(parent=str(parent), completed_clips_reused=reused,
            immutable_frames_hardlinked=True, parent_modified=False,
            old_optimizer_reused=False))
    return sources


def build_inputs(root, rows):
    from PIL import Image
    for i,row in enumerate(rows):
        supervision.interrupted()
        folder = root/'clips'/row['id']
        receipt = read(folder/'labels.json')
        if not receipt['samples']:
            continue
        prepared = read(folder/'prepared.json')
        identity = dict(labels=sha(folder/'labels.npz'), prepared=sha(folder/'prepared.json'),
                        shape=list(FRAME_HW), fps=prepared['source']['video']['fps'])
        if identity['labels'] != receipt['sha256']:
            raise ValueError('label checksum mismatch')
        cached = folder/'visual_inputs.json'
        if cached.exists():
            r = read(cached)
            if r['identity'] != identity or sha(folder/'visual_inputs.npz') != r['sha256']:
                raise ValueError('visual input cache changed')
            continue
        source = prepared['source']
        images = []
        for n in range(source['video']['frames']):
            path = Path(source['directory'])/f'{n}.png'
            if sha(path) != source['hashes'][f'{n}.png']:
                raise ValueError('source frame changed')
            images.append(np.asarray(Image.open(path).convert('RGB')))
        small = thumbnails(images)
        np.savez_compressed(folder/'visual_inputs.npz', frames=small)
        save(cached, dict(identity=identity, sha256=sha(folder/'visual_inputs.npz')))
        status(root, 'prepare_cnn_inputs', i+1, len(rows), clip=row['id'])


class Corpus:
    """Keep labels in RAM; bounded LRU for actual source thumbnails."""
    def __init__(self, root, rows):
        self.root = root
        self.groups = {s:[] for s in ('train','val','test')}
        parents = {s:set() for s in self.groups}
        self.metadata, self.targets, hashes = {}, {}, {}
        self.cache = OrderedDict()
        for row in rows:
            folder = root/'clips'/row['id']
            receipt = read(folder/'labels.json')
            if not receipt['samples']:
                continue
            if receipt.get('input', row) != row:
                raise ValueError('label input differs from source manifest')
            if sha(folder/'labels.npz') != receipt['sha256']:
                raise ValueError('label checksum mismatch')
            visual = read(folder/'visual_inputs.json')
            if (visual['identity']['labels'] != receipt['sha256']
                    or visual['identity']['prepared'] != sha(folder/'prepared.json')
                    or sha(folder/'visual_inputs.npz') != visual['sha256']):
                raise ValueError('visual input checksum mismatch')
            self.groups[row['split']].append(row)
            parents[row['split']].add(row['parent'])
            hashes[row['id']] = [receipt['sha256'], visual['sha256'],
                                 digest(receipt['records']), visual['identity']['prepared']]
            self.metadata[row['id']] = (receipt, read(folder/'prepared.json'))
            if row['split'] != 'test':
                with np.load(folder/'labels.npz') as d:
                    y = d['y'].copy()
                if y.shape != (receipt['samples'],5) or len(receipt['records']) != len(y):
                    raise ValueError('action/target alignment mismatch')
                if not np.isfinite(y).all():
                    raise ValueError('nonfinite targets')
                self.targets[row['id']] = torch.from_numpy(y)
        if any(parents[a]&parents[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('source leakage between splits')
        if not self.groups['train'] or not self.groups['val']:
            raise ValueError('source-disjoint train and validation sets required')
        self.data_hash = digest(hashes)
        self.parents = parents
        values = torch.cat([self.targets[r['id']] for r in self.groups['train']])
        # Scale only: preserving zero and exact reverse-action antisymmetry.
        self.scale = values.square().mean(0).sqrt().clamp_min(1e-6)
        self.weights = {}
        for split in ('train','val'):
            counts = Counter(r['parent'] for r in self.groups[split])
            domains = {}
            for r in self.groups[split]:
                domains.setdefault(r.get('dataset','default'), set()).add(r['parent'])
            self.weights[split] = torch.tensor([
                1/(len(domains[r.get('dataset','default')])*counts[r['parent']])
                for r in self.groups[split]], dtype=torch.double)

    def frames(self, row):
        key = row['id']
        if key not in self.cache:
            with np.load(self.root/'clips'/key/'visual_inputs.npz') as d:
                self.cache[key] = torch.from_numpy(d['frames'].copy()).permute(0,3,1,2)
        self.cache.move_to_end(key)
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return self.cache[key]

    def batch(self, row, ids, device):
        receipt, prepared = self.metadata[row['id']]
        records = [receipt['records'][int(i)] for i in ids]
        frames = self.frames(row)
        c,g,s = pair_context(len(frames), records, prepared['keys'], prepared['cuts'],
                            prepared['source']['video']['fps'])
        y = self.targets[row['id']][ids]/self.scale
        return [v.to(device) for v in (frames,c,g,s,y)]


def train(root, p, rows, *, max_steps=None):
    """Atomic full-state resume at optimizer boundaries, including CUDA RNG."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(p['seed'])
    np.random.seed(p['seed'])
    torch.manual_seed(p['seed'])
    cfg = p['train']
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(p['seed'])
    corpus = Corpus(root, rows)
    model = SpatialTemporalSelector(**p['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    def lr_factor(e):
        if e < cfg['warmup_epochs']:
            return (e+1)/cfg['warmup_epochs']
        t = (e-cfg['warmup_epochs'])/max(1,cfg['max_epochs']-cfg['warmup_epochs'])
        return .05+.95*.5*(1+math.cos(math.pi*min(1,t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    generator = torch.Generator().manual_seed(p['seed']+1)
    training_rows = corpus.groups['train']
    label_count = sum(len(corpus.targets[r['id']]) for r in training_rows)
    draws = max(cfg['accumulate'], math.ceil(label_count/cfg['pairs_per_clip']))
    folder = root/'training'
    folder.mkdir(exist_ok=True)
    latest = folder/'last.pt'
    epoch = position = steps = bad = 0
    best = float('inf')
    order = None
    if latest.exists():
        s = torch.load(latest, map_location='cpu', weights_only=False)
        if s['protocol'] != p['signature'] or s['data_hash'] != corpus.data_hash:
            raise ValueError('training resume identity mismatch')
        if s['device'] != str(device):
            raise ValueError('exact resume requires the same training device')
        if not torch.equal(s['target_scale'], corpus.scale):
            raise ValueError('train-only target scale changed')
        model.load_state_dict(s['model'])
        optimizer.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler'])
        epoch,position,steps,bad,best,order = (s[k] for k in ('epoch','position','steps','bad','best','order'))
        random.setstate(s['python_rng'])
        np.random.set_state(s['numpy_rng'])
        torch.set_rng_state(s['torch_rng'])
        generator.set_state(s['sampler_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(s['cuda_rng'])
    def export():
        return dict(architecture=ARCHITECTURE, model_config=p['model'],
            model=model.state_dict(), target_scale=corpus.scale, protocol=p['signature'],
            data_hash=corpus.data_hash, epoch=epoch, target_order=p['target_order'])
    def persist():
        supervision.checkpoint(latest, dict(export(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), position=position, steps=steps, bad=bad,
            best=best, order=order, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            sampler_rng=generator.get_state(), device=str(device)))
    save(folder/'dataset.json', dict(data_hash=corpus.data_hash, train_rows=label_count,
        validation_rows=sum(len(corpus.targets[r['id']]) for r in corpus.groups['val']),
        test_targets_loaded=False, source_counts={s:len(v) for s,v in corpus.parents.items()},
        parameters=sum(x.numel() for x in model.parameters()), model=p['model'],
        sampler='equal dataset, equal parent, equal clip within parent, uniform actions',
        cnn_trainable=True, temporal_attention_trainable=True))
    while epoch < cfg['max_epochs'] and bad < cfg['patience']:
        if order is None:
            clip_order = torch.multinomial(corpus.weights['train'], draws, replacement=True, generator=generator)
            order = [(int(i), torch.randint(len(corpus.targets[training_rows[int(i)]['id']]),
                (cfg['pairs_per_clip'],), generator=generator)) for i in clip_order]
            position = 0
        model.train()
        while position < len(order):
            if supervision.STOP:
                persist()
                return dict(status='INTERRUPTED', steps=steps)
            optimizer.zero_grad(set_to_none=True)
            end = min(len(order), position+cfg['accumulate'])
            loss_value = 0.
            for clip_id, ids in order[position:end]:
                inputs = corpus.batch(training_rows[clip_id], ids, device)
                loss = F.smooth_l1_loss(model(*inputs[:4]), inputs[4])/(end-position)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                loss.backward()
                loss_value += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            steps += 1
            position = end
            if steps%10 == 0 or supervision.STOP or max_steps and steps>=max_steps:
                persist()
            status(root,'train',epoch,cfg['max_epochs'],epoch=epoch+1,step=steps,
                batch_done=position,batch_total=len(order),loss=loss_value,
                best_val=best if math.isfinite(best) else None,
                resume='model + optimizer + scheduler + CPU/CUDA RNG + sampler order/offset')
            if supervision.STOP or max_steps and steps>=max_steps:
                return dict(status='INTERRUPTED',steps=steps)
        model.eval()
        total = denominator = 0.
        with torch.inference_mode():
            for j,row in enumerate(corpus.groups['val']):
                target = corpus.targets[row['id']]
                losses = []
                visual = None
                for offset in range(0,len(target),cfg['pairs_per_clip']):
                    ids = torch.arange(offset,min(len(target),offset+cfg['pairs_per_clip']))
                    frames,c,g,s,y = corpus.batch(row,ids,device)
                    if visual is None:
                        visual = model.encode_frames(frames,s)
                    losses.append(F.smooth_l1_loss(model.predict_encoded(visual,c,g,s),y,
                                                  reduction='none').mean(1).cpu())
                w = float(corpus.weights['val'][j])
                total += float(torch.cat(losses).mean())*w
                denominator += w
                if supervision.STOP:
                    persist()
                    return dict(status='INTERRUPTED',steps=steps)
        value = total/denominator
        if not math.isfinite(value):
            raise ValueError('nonfinite validation loss')
        improved = value < best-1e-5
        if improved:
            best,bad = value,0
            supervision.checkpoint(folder/'best.pt', dict(export(), epoch=epoch+1,val_loss=value))
        else:
            bad += 1
        with (folder/'history.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=epoch+1,steps=steps,val_loss=value,best=best,improved=improved))+'\n')
        epoch += 1
        scheduler.step()
        order,position = None,0
        persist()
        status(root,'train',epoch,cfg['max_epochs'],epoch=epoch,step=steps,best_val=best,
               early_stop_bad_epochs=bad)
    result = dict(status='TRAINED_NOT_YET_RD_VALIDATED', epochs=epoch,steps=steps,
        best_val=best,checkpoint=str(folder/'best.pt'),checkpoint_sha256=sha(folder/'best.pt'),
        architecture=ARCHITECTURE,performance_improvement_proven=False,automatic_default_replacement=False)
    save(root/'RESULT.json',result)
    return result


def pipeline(root, parent):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Also exclude any resumed v1 writer while sharing its immutable provenance.
    parent_lock = (parent/'run.lock').open('a')
    fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock = (REPO/'.local/etri_60s_check.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status(root,'verify_and_reuse',resume='verifying parent source/code/model identities')
    p = protocol(root,parent)
    rows = reuse_preparation(root,parent,p)
    prepared = supervision.prepare(root,rows,p['supervision_protocol'])
    supervision.interrupted()
    supervision.label(root,prepared,p['supervision_protocol'])
    supervision.interrupted()
    build_inputs(root,prepared)
    supervision.interrupted()
    result = train(root,p,prepared)
    status(root,'complete' if result['status']!='INTERRUPTED' else 'train',
           result=result,status_detail=result['status'])
    value = read(root/'status.json')
    value['status'] = result['status']
    save(root/'status.json',value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--parent',type=Path,default=PARENT)
    args = parser.parse_args()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,supervision.stop)
    try:
        pipeline(args.output.resolve(),args.parent.resolve())
    except BlockingIOError:
        raise SystemExit('Another selector/GPU job holds the lock; existing status preserved')
    except BaseException as error:
        root = args.output.resolve()
        value = read(root/'status.json') if (root/'status.json').exists() else {}
        value.update(status='INTERRUPTED' if supervision.STOP or isinstance(error,InterruptedError) else 'FAILED',
                     error=repr(error),exit_code=130 if supervision.STOP else 1,updated_unix=time.time())
        save(root/'status.json',value)
        raise


if __name__ == '__main__':
    main()
