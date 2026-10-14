"""Strict, resumable action evaluation with the authors' TimeSformer test loader."""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import sys
import time

from . import kinetics_assets as assets
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint

REPO = assets.REPO
CONFIG = REPO/'07_downstream/timesformer/TimeSformer_divST_8x32_224_TEST.yaml'
POLICY = 'author_timesformer_8x32_224_3crop_logits_v1'
ARMS = ('original', 'bsu_context')
read = lambda p: json.loads(Path(p).read_text())


def signed(value):
    return dict(value, signature=fingerprint(value))


def verify_signature(value):
    if fingerprint({k:v for k,v in value.items() if k!='signature'}) != value['signature']:
        raise ValueError('action artifact signature changed')


def validate_runtime(runtime):
    if runtime['upstream_commit'] != assets.COMMIT or runtime['checkpoint_sha256'] != assets.CHECKPOINT_SHA:
        raise ValueError('unrecognized TimeSformer source/checkpoint')
    receipt = Path(runtime['source_receipt'])
    if sha256(receipt) != runtime['source_receipt_sha256']:
        raise ValueError('TimeSformer source receipt changed')
    source = read(receipt)
    if (source['commit'] != assets.COMMIT or source['archive_sha256'] != assets.SOURCE_SHA
            or source['patch_sha256'] != sha256(REPO/'07_downstream/timesformer/patches/timesformer_modifications.patch')):
        raise ValueError('TimeSformer source provenance changed')
    for name,digest in source['files'].items():
        if sha256(Path(runtime['source'])/name) != digest:
            raise ValueError('TimeSformer implementation changed: '+name)
    if sha256(runtime['checkpoint']) != assets.CHECKPOINT_SHA:
        raise ValueError('TimeSformer checkpoint bytes changed')
    if not Path(runtime['python']).is_file():
        raise FileNotFoundError(runtime['python'])


def evaluation_job(plan, results):
    if len(plan['videos']) != 14 or len(results) != 14:
        raise ValueError('all 14 reconstructed clips are required for action evaluation')
    rows = []
    for video,result in zip(plan['videos'],results):
        if (result['id'] != video['id'] or result['signature'] != plan['signature']
                or result['frames'] != video['frames'] or result['mode'] != 'combined_context'
                or result['snr_db'] != 10):
            raise ValueError('action input differs from the frozen BSU condition')
        source = read(video['prepared'])
        for arm,path in [('original',source['normalized_video']),('bsu_context',result['video'])]:
            digest = sha256(path)
            if arm == 'original' and digest != source['normalized_sha256']:
                raise ValueError('original action input changed')
            rows.append(dict(id=video['id'],variant=arm,label=video['label'],path=str(Path(path).resolve()),
                             sha256=digest,frames=video['frames']))
    return signed(dict(policy=POLICY,benchmark_signature=plan['signature'],runtime=plan['action_runtime'],
        output=str(Path(plan['root']).parent/'action'),rows=rows,config=str(CONFIG),config_sha256=sha256(CONFIG),
        code={str(p):sha256(p) for p in [Path(__file__),REPO/'src/semantic_transmission/kinetics_assets.py']},
        evaluation=dict(num_frames=8,sampling_rate=32,spatial_crops=3,temporal_views=1,crop_size=224,
            target_fps=30,ensemble='sum_logits',precision='float32',videos_per_variant=14,
            original='full normalized 576x320 24fps source',reconstruction='delivered combined_context MP4')))


def validate_job(job):
    verify_signature(job)
    if job['policy'] != POLICY or sha256(job['config']) != job['config_sha256']:
        raise ValueError('action policy/config changed')
    for p,d in job['code'].items():
        if sha256(p) != d: raise ValueError('action evaluation code changed')
    rows=job['rows']
    identities={(r['id'],r['variant']) for r in rows}
    if len(rows)!=28 or len(identities)!=28:
        raise ValueError('expected exactly 28 unique original/BSU inputs')
    by_arm={a:{r['id']:(r['label'],r['frames']) for r in rows if r['variant']==a} for a in ARMS}
    if len(by_arm['original'])!=14 or by_arm['original']!=by_arm['bsu_context']:
        raise ValueError('original/BSU clips or labels do not match')
    for row in rows:
        if Path(row['id']).name!=row['id'] or row['id'] in ('.','..') or not 0<=row['label']<400:
            raise ValueError('invalid clip identity/label')
        if sha256(row['path'])!=row['sha256']:raise ValueError('action input bytes changed: '+row['path'])


def configure(runtime, csv_dir):
    # Import only inside the isolated environment. No fallback model or decoder.
    import importlib.metadata
    for spec in assets.PACKAGES:
        package,version=spec.split('==')
        if importlib.metadata.version(package)!=version:raise ValueError('runtime package changed: '+package)
    sys.path.insert(0,runtime['source'])
    from timesformer.config.defaults import get_cfg
    cfg=get_cfg();cfg.merge_from_file(str(CONFIG))
    cfg.DATA.PATH_TO_DATA_DIR=str(csv_dir);cfg.TEST.CHECKPOINT_FILE_PATH=runtime['checkpoint']
    cfg.OUTPUT_DIR=str(csv_dir);cfg.DATA_LOADER.NUM_WORKERS=0
    cfg.TEST.BATCH_SIZE=3  # One complete video's three views, FP32, eval mode.
    if (cfg.DATA.NUM_FRAMES,cfg.DATA.SAMPLING_RATE,cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.TEST.NUM_ENSEMBLE_VIEWS,cfg.DATA.TEST_CROP_SIZE,cfg.DATA.TARGET_FPS,
            cfg.DATA.ENSEMBLE_METHOD,cfg.MODEL.NUM_CLASSES)!=(8,32,3,1,224,30,'sum',400):
        raise ValueError('author evaluation geometry/ensemble changed')
    cfg.freeze()
    return cfg


def load_model(runtime, device):
    import torch
    from timesformer.models.vit import VisionTransformer
    # Same registered vit_base_patch16_224 model without its redundant ImageNet
    # download. Every parameter must then load from the pinned K400 checkpoint.
    model=VisionTransformer(img_size=224,num_classes=400,patch_size=16,embed_dim=768,depth=12,
        num_heads=12,mlp_ratio=4,qkv_bias=True,norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),
        drop_rate=0.,attn_drop_rate=0.,drop_path_rate=.1,num_frames=8,attention_type='divided_space_time')
    checkpoint=torch.load(runtime['checkpoint'],map_location='cpu')
    weights=checkpoint['model_state']
    if not weights or any(not k.startswith('model.') for k in weights):
        raise ValueError('unexpected official checkpoint parameter names')
    model.load_state_dict({k[len('model.'):]:v for k,v in weights.items()},strict=True)
    del checkpoint,weights
    return model.eval().to(device)


def classify(model, dataset, index, label, device):
    import torch
    views=[]
    for crop in range(3):
        frames,actual_label,actual_index,_=dataset[3*index+crop]
        if actual_label!=label or actual_index!=3*index+crop or tuple(frames.shape)!=(3,8,224,224):
            raise ValueError('TimeSformer returned a replacement clip or wrong crop geometry')
        views.append(frames)
    with torch.no_grad():
        preds=model(torch.stack(views).to(device)).detach().cpu()
    if tuple(preds.shape)!=(3,400) or not torch.isfinite(preds).all():
        raise ValueError('invalid TimeSformer predictions')
    # Match TestMeter's sequential FP32 addition, not mean crop accuracies or
    # averaging independently softmax-normalized scores.
    total=torch.zeros(400)
    for pred in preds:total+=pred
    return total.tolist(),total.topk(5).indices.tolist()


def record_path(job, row):
    return Path(job['output'])/'predictions'/f'{row["id"]}_{row["variant"]}.json'


def load_record(job, row):
    path=record_path(job,row)
    if not path.exists():return None
    value=read(path);verify_signature(value)
    if value['job_signature']!=job['signature'] or value['input']!=row or value['crops']!=3:
        raise ValueError('prediction belongs to another input or configuration')
    logits=value['summed_logits'];ranking=value['top5']
    if len(logits)!=400 or not all(math.isfinite(x) for x in logits) or len(set(ranking))!=5:
        raise ValueError('invalid stored prediction')
    if any(not 0<=n<400 for n in ranking):raise ValueError('invalid class index')
    # Accommodate tied logits without changing torch.topk's tie-breaking order.
    if [logits[i] for i in ranking]!=sorted(logits,reverse=True)[:5]:
        raise ValueError('stored ranking does not match logits')
    if value['top1_correct']!=(row['label']==ranking[0]) or value['top5_correct']!=(row['label'] in ranking):
        raise ValueError('stored accuracy does not match its label')
    return value


def summary(records):
    output={}
    for arm in ARMS:
        selected=[r for r in records if r['input']['variant']==arm]
        if len(selected)!=14:raise ValueError('cannot report full benchmark accuracy from partial predictions')
        top1=sum(r['top1_correct'] for r in selected);top5=sum(r['top5_correct'] for r in selected)
        output[arm]=dict(videos=14,top1_correct=top1,top1_percent=100*top1/14,
                        top5_correct=top5,top5_percent=100*top5/14)
    return output


def validate_evaluation(job):
    validate_job(job)
    records=[load_record(job,row) for row in job['rows']]
    if any(r is None for r in records):raise ValueError('action evaluation incomplete')
    result=read(Path(job['output'])/'RESULT.json');verify_signature(result)
    hashes={str(record_path(job,row).relative_to(job['output'])):sha256(record_path(job,row)) for row in job['rows']}
    if (result['job_signature']!=job['signature'] or result['summary']!=summary(records)
            or result['artifacts']!=hashes or result['status']!='COMPLETE_ACTION_EVALUATION'):
        raise ValueError('action result provenance/summary changed')
    return result


def evaluate(job):
    validate_job(job);validate_runtime(job['runtime'])
    output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    records=[load_record(job,row) for row in job['rows']]
    if (output/'RESULT.json').exists():
        result=validate_evaluation(job)
        assets.event(overall='복원 14/14 (100.0%) | 행동평가 28/28 (100.0%)',stage='완료된 행동 평가 검증·재사용')
        return result
    # The upstream loader splits on a comma; reject paths it cannot represent.
    if any(',' in r['path'] or '\n' in r['path'] for r in job['rows']):
        raise ValueError('TimeSformer CSV cannot represent a comma/newline in a path')
    csv_text=''.join(f'{r["path"]},{r["label"]}\n' for r in job['rows'])
    csv_path=output/'test.csv'
    if csv_path.exists() and csv_path.read_text()!=csv_text:raise ValueError('action CSV changed')
    csv_path.write_text(csv_text)
    cfg=configure(job['runtime'],output)
    import random
    import numpy as np
    import torch
    from timesformer.datasets.kinetics import Kinetics
    random.seed(cfg.RNG_SEED);np.random.seed(cfg.RNG_SEED);torch.manual_seed(cfg.RNG_SEED)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this action benchmark')
    assets.event(stage='TimeSformer 가중치 엄격 로딩 (진행률 미확정)')
    model=load_model(job['runtime'],'cuda');dataset=Kinetics(cfg,'test')
    if len(dataset)!=84:raise ValueError('expected 28 inputs x 3 spatial crops')
    done=sum(r is not None for r in records);started=time.monotonic();new=0
    for index,row in enumerate(job['rows']):
        if records[index] is not None:continue
        assets.event(current=row['id']+' '+row['variant'],stage=f'TimeSformer {done}/28 ({100*done/28:.1f}%)')
        tick=time.monotonic();logits,ranking=classify(model,dataset,index,row['label'],'cuda')
        record=signed(dict(job_signature=job['signature'],input=row,crops=3,summed_logits=logits,top5=ranking,
            top1_correct=row['label']==ranking[0],top5_correct=row['label'] in ranking,seconds=time.monotonic()-tick))
        freeze(record_path(job,row),record);records[index]=record;done+=1;new+=1
        write_json(output/'status.json',dict(status='RUNNING',complete=done,total=28))
        eta=(time.monotonic()-started)/new*(28-done)
        assets.event(overall=f'복원 14/14 (100.0%) | 행동평가 {done}/28 ({100*done/28:.1f}%)',
            stage=f'TimeSformer {done}/28 ({100*done/28:.1f}%) | ETA {eta:.0f}s')
    result=signed(dict(status='COMPLETE_ACTION_EVALUATION',job_signature=job['signature'],summary=summary(records),
        artifacts={str(record_path(job,r).relative_to(output)):sha256(record_path(job,r)) for r in job['rows']}))
    freeze(output/'RESULT.json',result)
    write_json(output/'status.json',dict(status='COMPLETE',complete=28,total=28))
    return validate_evaluation(job)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--job',type=Path);group.add_argument('--check-runtime',type=Path)
    args=parser.parse_args()
    if args.check_runtime:
        runtime=read(args.check_runtime);validate_runtime(runtime)
        configure(runtime,args.check_runtime.parent)
        model=load_model(runtime,'cpu')
        print('CHECK: official TimeSformer loaded strictly; parameters='+str(sum(p.numel() for p in model.parameters())))
    else:evaluate(read(args.job))


if __name__=='__main__':main()
