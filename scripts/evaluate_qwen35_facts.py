"""Generate source-only captions from validation-selected joint checkpoints."""
import argparse
import fcntl
from pathlib import Path
import time
import torch

from semantic_transmission.qwen35_ablation import load_inference,encode,generate
from semantic_transmission.qwen35_training import CONFIG,RUN,rows_for,checkpoint_path,release_gpu
from semantic_transmission.qwen35_training_data import ROOT,DEFAULT_DATA,read_json,write_json,sha256,verify
from semantic_transmission.qwen35_multitask import coverage


def evaluate(args,mode):
    folder=args.run/mode
    if read_json(folder/'RESULT.json')['status']!='COMPLETE':raise ValueError('Training incomplete')
    checkpoint=checkpoint_path(folder)
    config=read_json(CONFIG)
    protocol={'checkpoint':str(checkpoint),'manifest_sha256':sha256(checkpoint/'complete.json'),
              'config':config,'data_sha256':sha256(DEFAULT_DATA/'freeze.json'),
              'generation_module_sha256':sha256(ROOT/'src/semantic_transmission/qwen35_ablation.py'),
              'evaluation_code_sha256':sha256(Path(__file__)),
              'selection':'Existing validation-only best pointer, fixed before test generation',
              'scope':'Source-only hard four-key inference; no reference captions/facts/events enter the model'}
    out=args.run/'evaluation'/mode
    if (out/'protocol.json').exists() and read_json(out/'protocol.json')!=protocol:raise ValueError('Changed evaluation conditions')
    write_json(out/'protocol.json',protocol)
    rows=[r for split in ('validation','test') for r in rows_for(DEFAULT_DATA,split)]
    pending=[r for r in rows if not (out/'records'/f'{r["id"]}.json').exists()]
    if pending:
        model,processor,head,runtime=load_inference(config,checkpoint)
        for row in pending:
            start=time.monotonic()
            features=encode(model,processor,row,config)
            with torch.no_grad():keys=head(features['pooled'],config['temperature'])['indices'].tolist()
            call=generate(model,processor,row,keys,features,config,True)
            write_json(out/'records'/f'{row["id"]}.json',dict(call,id=row['id'],split=row['split'],keys=keys,
                source_key_indices=[row['frames'][i]['index'] for i in keys],
                source_hashes=[row['frames'][i]['sha256'] for i in keys],coverage=coverage(row,keys),
                total_seconds=time.monotonic()-start,protocol_sha256=sha256(out/'protocol.json'),runtime=runtime))
            print(mode,row['id'],keys,call['caption'],flush=True)
            del features
        del model,processor,head;release_gpu()
    files={f'records/{r["id"]}.json':sha256(out/'records'/f'{r["id"]}.json') for r in rows}
    for name in files:
        r=read_json(out/name)
        assert r['protocol_sha256']==sha256(out/'protocol.json') and r['status']=='COMPLETE'
        assert len(r['keys'])==len(set(r['keys']))==4 and r['keys']==sorted(r['keys']) and r['keys'][0]==0 and r['keys'][-1]==7
    write_json(out/'generation_freeze.json',{'files':files,'protocol_sha256':sha256(out/'protocol.json')})
    write_json(out/'RESULT.json',{'status':'GENERATED_CAPTION_REVIEW_PENDING','windows':len(rows)})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'outputs/qwen35_fact_joint_v2')
    p.add_argument('--mode',choices=['control','facts','both'],default='both')
    args=p.parse_args()
    verify(DEFAULT_DATA)
    with open('/tmp/qwen35_multitask_gpu.lock','w') as lock,(ROOT/'.local/etri_60s_check.lock').open('a+') as shared:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for mode in ['control','facts'] if args.mode=='both' else [args.mode]:evaluate(args,mode)
