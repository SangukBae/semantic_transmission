"""Resumable real-transport supervision and training for the SPARC auxiliary selector."""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from .sparc_selector_data import (action_features, candidates, digest, load_frames,
                                  read, save, sha, source_features, status)
from .sparc_selector_workers import Client, REPO

DEFAULT = REPO/'outputs/sparc_selector_v1_20261020'
STOP = False


def stop(*_):
    global STOP
    STOP = True


def interrupted():
    if STOP:
        raise InterruptedError('stop requested; completed clips/steps retained')


def protocol(root, smoke=False):
    from .sparc_uf import WEIGHTS
    from . import dense_temporal_receiver as temporal
    names = ['sparc_selector_data', 'sparc_selector_workers', 'sparc_selector_train',
             'dcvc_uf_codec', 'sparc_uf_rate_transport', 'sparc_uf_transport',
             'sparc_semantics', 'sparc_selection', 'sparc', 'dense_temporal_receiver',
             'metadata_channel', 'packets', 'qwen35_semantic_selection',
             'qwen35_feedback', 'qwen35_state_delta', 'qwen35_backend']
    code = {n:sha(REPO/'src/semantic_transmission'/f'{n}.py') for n in names}
    assets = {str(REPO/'.local/dcvc_uf_weights'/k):v for k,v in WEIGHTS.items()}
    assets.update({str(REPO/'.local/checkpoints/rife/v4.25/flownet.pkl'):temporal.WEIGHT_SHA,
                   str(REPO/'.local/checkpoints/rife/v4.25/IFNet_HDv3.py'):temporal.NETWORK_SHA,
                   str(REPO/'.local/vendor/Practical-RIFE/model/warplayer.py'):temporal.WARP_SHA})
    for path,h in assets.items():
        if sha(path) != h:
            raise ValueError('model identity mismatch: '+path)
    vendor = REPO/'.local/dcvc_uf_vendor'
    from .sparc_uf_transport import UPSTREAM
    if subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip() != UPSTREAM:
        raise ValueError('UF revision changed')
    if subprocess.check_output(['git','-C',str(vendor),'diff','HEAD']):
        raise ValueError('UF source is dirty')
    cfg = dict(schema='sparc_auxiliary_learned_v1', code=code, assets=assets,
        source_manifest_sha256=sha(root/'sources.json'), qp=30, max_gap=None,
        semantic_policy=read(REPO/'configs/sparc.json'), snr_db=10, channel_seed=42,
        seed=20261020, states=6, actions_per_state=4, reverse_labels=True,
        train=dict(max_epochs=200, patience=20, batch_size=256, lr=.001,
                   weight_decay=.0001, hidden=[128,128], device='cpu'),
        target_order=['mse_gain','lpips_gain','temporal_mse_gain','local_mae95_gain','channel_uses_per_source_pixel_delta'],
        label_quality='delivered H264 CRF18 MP4', no_oracle_in_inference=True,
        uf_rife_frozen=True, automatic_default_replacement=False, smoke=smoke)
    cfg['signature'] = digest(cfg)
    p = root/'protocol.json'
    if p.exists() and read(p) != cfg:
        raise ValueError('frozen protocol/source/code changed; use a new run directory')
    save(p, cfg)
    return cfg


def runtimes():
    return dict(semantic=str(REPO/'.local/qwen35_env/bin/python'),
                encoder=str(REPO/'.local/dcvc_uf_runtime/bin/python'),
                decoder=str(REPO/'.local/dcvc_uf_runtime/bin/python'),
                receiver=str(REPO/'.local/bsu_codec_eval_env/bin/python'),
                quality=str(REPO/'.local/bsu_codec_eval_env/bin/python'),
                channel=str(REPO.parent/'semantic_transmission/.local/envs/lgvsc-channel/bin/python'))


def prepare(root, sources, p):
    import numpy as np
    from PIL import Image
    client = None
    try:
        for n,row in enumerate(sources):
            interrupted()
            dest = root/'clips'/row['id']
            ready = dest/'prepared.json'
            if ready.exists():
                value = read(ready)
                if value['input'] != row or any(sha(Path(value['source']['directory'])/name) != h
                        for name,h in value['source']['hashes'].items()):
                    raise ValueError('prepared source changed')
                continue
            if (dest/'quarantine.json').exists():
                continue
            status(root, 'prepare_semantic_keys', n, len(sources), clip=row['id'],
                   log=str(root/'semantic.log'), resume='verified source hashes and cached Qwen calls')
            frames = load_frames(row)
            folder = dest/'frames'
            folder.mkdir(parents=True, exist_ok=True)
            for i,f in enumerate(frames):
                path = folder/f'{i}.png'
                if path.exists():
                    if not np.array_equal(np.asarray(Image.open(path)), f):
                        raise ValueError('partial prepared frame changed')
                else:
                    Image.fromarray(f).save(path)
            source = dict(directory=str(folder), video=dict(frames=len(frames), width=576,height=320,fps=24.),
                          hashes={f'{i}.png':sha(folder/f'{i}.png') for i in range(len(frames))})
            if p['smoke']:
                keys = [0,len(frames)-1]
                semantic = dict(keys=keys, smoke_endpoint_keys=True)
            else:
                if client is None:
                    client = Client('semantic',runtimes()['semantic'],root)
                try:
                    semantic = client.call(dict(source=source,output=str(dest)))
                except RuntimeError as error:
                    if 'semantic decision remains uncertain' not in str(error) and 'three invalid responses' not in str(error):
                        raise
                    save(dest/'quarantine.json',dict(input=row,reason=str(error),
                         use_for_training=False,source=source))
                    # A failed worker is restarted; successful call caches remain.
                    client.close()
                    client = None
                    continue
                keys = semantic['keys']
            _,_,cuts = source_features(frames)
            save(ready,dict(input=row,source=source,keys=keys,cuts=cuts,semantic=semantic,
                           protocol=p['signature']))
            status(root,'prepare_semantic_keys',n+1,len(sources),clip=row['id'])
    finally:
        if client:
            client.close()
    prepared = [r for r in sources if (root/'clips'/r['id']/'prepared.json').exists()]
    quarantine = [r['id'] for r in sources if r not in prepared]
    if len(prepared) < (.8*len(sources) if not p['smoke'] else 1):
        raise ValueError('more than 20 percent sources quarantined; investigate before training')
    save(root/'preparation.json',dict(prepared=len(prepared),quarantined=quarantine,
        split_parents={s:len({r['parent'] for r in prepared if r['split']==s}) for s in ('train','val','test')}))
    return prepared


class Evaluator:
    def __init__(self, root, p):
        self.root,self.p = root,p
        self.clients = {}

    def call(self, role, job):
        if role not in self.clients:
            self.clients[role] = Client(role,runtimes()[role],self.root)
        return self.clients[role].call(job)

    def evaluate(self, folder, source_array, indices, keys, cuts):
        identity = dict(protocol=self.p['signature'],source_sha256=sha(source_array),
                        indices=indices,keys=keys,cuts=cuts)
        out = folder/'observations'/digest(identity)[:24]
        result = out/'result.json'
        if result.exists():
            r = read(result)
            if r['identity'] != identity or any(sha(out/k)!=v for k,v in r['artifacts'].items()):
                raise ValueError('cached measurement identity/artifact mismatch')
            return r
        out.mkdir(parents=True,exist_ok=True)
        t = time.monotonic()
        send = self.call('encoder',dict(action='encode',frames=str(source_array),indices=indices,
                         keys=keys,cuts=cuts,packet=str(out/'sent.bin')))
        tx = time.monotonic()-t
        channel = self.call('channel',dict(input=str(out/'sent.bin'),output=str(out/'received.bin')))
        if send['complex_channel_uses'] != channel['complex_channel_uses']:
            raise ValueError('paid rate mismatch')
        t = time.monotonic()
        info = self.call('decoder',dict(action='decode',packet=str(out/'received.bin'),
                                       anchors=str(out/'anchors.npy')))
        if info['indices'] != indices or info['cuts'] != cuts:
            raise ValueError('receiver schedule mismatch')
        recv = self.call('receiver',dict(anchors=str(out/'anchors.npy'),info=info,
                                       movie=str(out/'delivered.mp4')))
        rx = time.monotonic()-t
        # Source pixels are passed only to this separate evaluator.
        quality = self.call('quality',dict(frames=str(source_array),movie=str(out/'delivered.mp4')))
        r = dict(identity=identity,sender=send,channel=channel,receiver=recv,quality=quality,
                 sender_seconds=tx,receiver_seconds=rx,
                 artifacts={k:sha(out/k) for k in ('sent.bin','received.bin','delivered.mp4')})
        save(result,r)
        (out/'anchors.npy').unlink()
        return r

    def close(self):
        for c in self.clients.values():
            c.close()


def label(root, sources, p):
    import numpy as np
    from PIL import Image
    evaluator = Evaluator(root,p)
    sample_count = 0
    try:
        for n,row in enumerate(sources):
            interrupted()
            folder = root/'clips'/row['id']
            done = folder/'labels.json'
            if done.exists():
                receipt = read(done)
                if receipt['protocol'] != p['signature'] or sha(folder/'labels.npz')!=receipt['sha256']:
                    raise ValueError('labels changed')
                sample_count += receipt['samples']
                continue
            prepared = read(folder/'prepared.json')
            source = prepared['source']
            frames = np.stack([np.asarray(Image.open(Path(source['directory'])/f'{i}.png').convert('RGB'))
                               for i in range(source['video']['frames'])])
            array = folder/'source.npy'
            np.save(array,frames)
            x,f,cuts = source_features(frames)
            if cuts != prepared['cuts']:
                raise ValueError('scene boundaries changed')
            keys = prepared['keys']
            seed = int(digest([p['seed'],row['id']])[:8],16)
            pairs = candidates(x,f,keys,cuts,seed,
                               states=2 if p['smoke'] else p['states'],
                               actions=2 if p['smoke'] else p['actions_per_state'])
            features,targets,records = [],[],[]
            for i,(before,after) in enumerate(pairs):
                interrupted()
                status(root,'measure_labels',n,len(sources),clip=row['id'],pair=i,
                       pairs_this_clip=len(pairs),samples=sample_count,
                       resume='whole-selection UF/channel/RIFE results cached by input and protocol hash')
                a = evaluator.evaluate(folder,array,before,keys,cuts)
                b = evaluator.evaluate(folder,array,after,keys,cuts)
                for left,right,l,r in [(before,after,a,b),(after,before,b,a)]:
                    gain = [l['quality'][k]-r['quality'][k] for k in
                            ('mse','lpips_vgg','temporal_mse','local_mae95')]
                    rate = (r['channel']['complex_channel_uses']-l['channel']['complex_channel_uses'])/math.prod(frames.shape[:3])
                    features.append(action_features(x,f,left,right,cuts,keys))
                    targets.append(gain+[rate])
                    records.append(dict(before=left,after=right,before_observation=digest(l['identity'])[:24],
                                        after_observation=digest(r['identity'])[:24]))
            if not features:
                save(folder/'no_actions.json',dict(reason='all frames protected or no valid change',keys=keys))
            np.savez_compressed(folder/'labels.npz',x=np.asarray(features,np.float32),
                                y=np.asarray(targets,np.float32))
            save(done,dict(protocol=p['signature'],input=row,samples=len(features),records=records,
                           sha256=sha(folder/'labels.npz')))
            sample_count += len(features)
            array.unlink()
            status(root,'measure_labels',n+1,len(sources),samples=sample_count)
    finally:
        evaluator.close()
    save(root/'label_summary.json',dict(clips=len(sources),samples=sample_count,
                                       simulated_channel=True,actual_entropy_coding=True))


def network(dim):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.GELU(),
                         nn.Linear(128,128),nn.GELU(),nn.Linear(128,5))


def checkpoint(path, record):
    import torch
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(record,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def train(root,p,sources, *, max_steps=None):
    """Complete optimizer/scheduler/RNG/order state; validation-only model choice."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    random.seed(p['seed'])
    np.random.seed(p['seed'])
    torch.manual_seed(p['seed'])
    groups = {s:[] for s in ('train','val','test')}
    parents = {s:set() for s in groups}
    receipts = {}
    for row in sources:
        folder = root/'clips'/row['id']
        if not (folder/'labels.json').exists():
            continue
        receipt = read(folder/'labels.json')
        if not receipt['samples']:
            continue
        if sha(folder/'labels.npz') != receipt['sha256']:
            raise ValueError('label checksum mismatch')
        receipts[row['id']] = receipt['sha256']
        groups[row['split']].append(row)
        parents[row['split']].add(row['parent'])
    if any(parents[a]&parents[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('source leakage between splits')
    if not groups['train'] or not groups['val']:
        raise ValueError('need source-disjoint train and validation data')
    data_hash = digest(receipts)
    # The test targets are not loaded for training, normalization or early stopping.
    def arrays(split_name):
        xs,ys,weights = [],[],[]
        source_counts = Counter(r['parent'] for r in groups[split_name])
        domain_parents = {}
        for row in groups[split_name]:
            domain_parents.setdefault(row.get('dataset','default'),set()).add(row['parent'])
        for row in groups[split_name]:
            d = np.load(root/'clips'/row['id']/'labels.npz')
            xs.append(d['x']);ys.append(d['y'])
            denominator = (len(domain_parents[row.get('dataset','default')]) *
                           source_counts[row['parent']]*len(d['x']))
            weights.extend([1/denominator]*len(d['x']))
        return torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys)),torch.tensor(weights)
    tx,ty,tw = arrays('train')
    vx,vy,vw = arrays('val')
    xm,xs = tx.mean(0),tx.std(0).clamp_min(1e-5)
    ym,ys = ty.mean(0),ty.std(0).clamp_min(1e-6)
    tx,vx = (tx-xm)/xs,(vx-xm)/xs
    ty,vy = (ty-ym)/ys,(vy-ym)/ys
    model = network(tx.shape[1])
    cfg = p['train']
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['max_epochs'])
    epoch=position=steps=bad=0
    best = float('inf')
    order = None
    generator = torch.Generator().manual_seed(p['seed']+1)
    folder = root/'training'
    folder.mkdir(exist_ok=True)
    latest = folder/'last.pt'
    normalization = dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['protocol']!=p['signature'] or state['data_hash']!=data_hash:
            raise ValueError('training resume identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch,position,steps,bad,best,order = (state[k] for k in ('epoch','position','steps','bad','best','order'))
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['sampler_rng'])
        if any(not torch.equal(normalization[k],state['normalization'][k]) for k in normalization):
            raise ValueError('train-only normalization changed')
    def persist():
        checkpoint(latest,dict(protocol=p['signature'],data_hash=data_hash,model=model.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,position=position,steps=steps,bad=bad,best=best,order=order,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            sampler_rng=generator.get_state(),normalization=normalization))
    save(folder/'dataset.json',dict(data_hash=data_hash,train_rows=len(tx),val_rows=len(vx),
         test_clips=len(groups['test']),test_targets_loaded=False,
         source_counts={k:len(v) for k,v in parents.items()},parameters=sum(q.numel() for q in model.parameters()),
         sampler='equal dataset probability, then equal parent probability; replacement; original-video grouped split'))
    while epoch < cfg['max_epochs'] and bad < cfg['patience']:
        if order is None:
            order = torch.multinomial(tw,len(tx),replacement=True,generator=generator)
            position = 0
        model.train()
        while position < len(order):
            ids = order[position:position+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(tx[ids]),ty[ids])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            steps += 1
            position += len(ids)
            if steps%20==0 or STOP or max_steps and steps>=max_steps:
                persist()
                status(root,'train',epoch,cfg['max_epochs'],epoch=epoch+1,step=steps,
                       batch_done=position,batch_total=len(order),loss=float(loss.item()),
                       best_val=None if not math.isfinite(best) else best,
                       early_stop_patience=cfg['patience'],resume='optimizer + scheduler + RNG + sampler order/offset')
            if STOP or max_steps and steps>=max_steps:
                return dict(status='INTERRUPTED',steps=steps)
        model.eval()
        with torch.inference_mode():
            per = torch.cat([F.smooth_l1_loss(model(x),y,reduction='none').mean(1)
                            for x,y in zip(vx.split(512),vy.split(512))])
            value = float((per*vw).sum()/vw.sum())
        improved = value < best-1e-5
        if improved:
            best,bad = value,0
            checkpoint(folder/'best.pt',dict(model=model.state_dict(),normalization=normalization,
                protocol=p['signature'],data_hash=data_hash,epoch=epoch+1,val_loss=value,
                input_dim=tx.shape[1],target_order=p['target_order']))
        else:
            bad += 1
        with (folder/'history.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=epoch+1,steps=steps,val_loss=value,best=best,improved=improved))+'\n')
        epoch += 1
        scheduler.step()
        order = None
        position = 0
        persist()
        status(root,'train',epoch,cfg['max_epochs'],step=steps,best_val=best,
               early_stop_bad_epochs=bad,resume='optimizer + scheduler + RNG + sampler order/offset')
    result = dict(status='TRAINED_NOT_YET_RD_VALIDATED',epochs=epoch,steps=steps,best_val=best,
                  checkpoint=str(folder/'best.pt'),checkpoint_sha256=sha(folder/'best.pt'),
                  automatic_default_replacement=False,performance_improvement_proven=False)
    save(root/'RESULT.json',result)
    return result


def select(frames, keys, cuts, checkpoint_path, rate_lambda):
    """Deployed selection reads source/context only; no trial neural reconstructions."""
    import numpy as np
    import torch
    from .sparc_selection import adaptive, validate_anchors
    if not math.isfinite(rate_lambda) or rate_lambda < 0:
        raise ValueError('invalid common validation-selected rate multiplier')
    saved = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model = network(saved['input_dim'])
    model.load_state_dict(saved['model']);model.eval()
    norm = saved['normalization']
    x,f,detected = source_features(frames)
    if detected != cuts:
        raise ValueError('frozen scene cuts differ')
    chosen = adaptive(x*255,keys,cuts,max_gap=None,threshold=6.)['indices']
    required = set(keys)|{i for c in cuts for i in (c-1,c)}
    visited = {tuple(chosen)}
    for _ in range(len(frames)):
        options = []
        for i in range(1,len(frames)-1):
            if i in required:
                continue
            proposal = sorted(set(chosen) ^ {i})
            if tuple(proposal) not in visited:
                options.append(proposal)
        if not options:
            break
        inputs = np.stack([action_features(x,f,chosen,a,cuts,keys) for a in options])
        with torch.inference_mode():
            y = model((torch.from_numpy(inputs)-norm['x_mean'])/norm['x_std'])*norm['y_std']+norm['y_mean']
            benefit = y[:,0]+.01*y[:,1]+.1*y[:,2]+.01*y[:,3]-rate_lambda*y[:,4]
        index = int(benefit.argmax())
        if benefit[index] <= 0:
            break
        chosen = options[index]
        visited.add(tuple(chosen))
    validate_anchors(chosen,len(frames),None,cuts)
    assert required <= set(chosen)
    return dict(indices=chosen,semantic_keys=keys,cuts=cuts,max_gap=None,
                selector='learned_v1_experimental',rate_lambda=rate_lambda,
                receiver_quality_used=False,selection_reads_source_only=True)


def pipeline(root, smoke=False):
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock = (REPO/'.local/etri_60s_check.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p = protocol(root,smoke)
    sources = read(root/'sources.json')
    status(root,'preflight',0,len(sources),resume='protocol and source hashes verified')
    prepared = prepare(root,sources,p)
    interrupted()
    # Qwen has exited before the codec/receiver workers claim GPU memory.
    label(root,prepared,p)
    gpu_lock.close()
    interrupted()
    result = train(root,p,prepared)
    status(root,'complete',1,1,result=result,status_detail=result['status'])
    current = read(root/'status.json')
    current['status'] = result['status']
    save(root/'status.json',current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)
    try:
        pipeline(args.output.resolve(),args.smoke)
    except BlockingIOError:
        raise SystemExit('Another selector/GPU experiment holds the lock; existing status preserved')
    except BaseException as error:
        root = args.output.resolve()
        value = read(root/'status.json') if (root/'status.json').exists() else {}
        value.update(status='INTERRUPTED' if STOP or isinstance(error,InterruptedError) else 'FAILED',
                     error=repr(error),exit_code=130 if STOP else 1,updated_unix=time.time())
        save(root/'status.json',value)
        raise


if __name__ == '__main__':
    main()
