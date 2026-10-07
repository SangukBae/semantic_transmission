"""Post-hoc development probe: use the exact trained action-QA prompt.

Uses only the known 14-window diagnostic. No new test or validation examples,
no checkpoint selection, no added facts in prompts, no additional training.
"""
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path
import time

from semantic_transmission.qwen35_caption_repair import RUN, verify
from semantic_transmission.qwen35_training import checkpoint_path,release_gpu
from semantic_transmission.qwen35_ablation import load_inference,encode,generate
from semantic_transmission.qwen35_training_data import ROOT,read_json,write_json,sha256


def run(root):
    p=verify(root);out=root/'action_probe';rows=read_json(root/'regression.json')['records']
    assert all(r['split']=='regression' for r in rows)
    selected={a:checkpoint_path(root/a) for a in p['settings']['arms']}
    config=dict(p['config'],caption_prompt=p['settings']['auxiliary_prompt'])
    protocol=dict(parent_result_sha256=sha256(root/'RESULT.json'),parent_freeze_sha256=sha256(root/'freeze.json'),
        cells=['base','caption_ce','caption_qa'],config=config,rows=rows,
        checkpoints={a:dict(path=str(c),sha256=sha256(c/'complete.json')) for a,c in selected.items()},
        implementation_sha256=sha256(Path(__file__)),scope='Post-hoc known-development action question probe, not a new test or model-selection run',
        comparison='Exact auxiliary training prompt reused at inference; caption prompt results remain unchanged',max_tokens_unchanged=True)
    path=out/'protocol.json'
    if path.exists() and read_json(path)!=protocol:raise ValueError('Probe conditions changed')
    write_json(path,protocol)
    pending=[r for r in rows if not (out/'records'/f'{r["id"]}.json').exists()]
    if pending:
        model,processor,head,runtime=load_inference(config,Path(p['checkpoint']))
        for a,c in selected.items():model.load_adapter(c/'adapter',adapter_name=a,is_trainable=False)
        model.eval()
        for row in pending:
            feat=encode(model,processor,row,config);cells={}
            names=protocol['cells'] if row['window']%2==0 else list(reversed(protocol['cells']))
            for name in names:
                model.set_adapter('default' if name=='base' else name)
                c=generate(model,processor,row,row['keys'],feat,config,adapter=name!='base')
                cells[name]=dict(c,keys=row['keys'],source_hashes=[row['frames'][i]['sha256'] for i in row['keys']])
                print('PROBE',row['id'],name,c['status'],flush=True)
            write_json(out/'records'/f'{row["id"]}.json',dict(id=row['id'],cells=cells,protocol_sha256=sha256(path)))
            del feat
        del model,processor,head
        release_gpu()
    write_json(out/'freeze.json',dict(protocol_sha256=sha256(path),files={f'records/{r["id"]}.json':sha256(out/'records'/f'{r["id"]}.json') for r in rows}))
    write_json(out/'RESULT.json',dict(status='GENERATED_REVIEW_PENDING',windows=len(rows),condition_windows=3*len(rows),purpose=protocol['scope']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN);args=parser.parse_args()
    with ExitStack() as stack:
        for name in ('/tmp/qwen35_multitask_gpu.lock',ROOT/'.local/etri_60s_check.lock'):
            lock=stack.enter_context(open(name,'w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.run)
