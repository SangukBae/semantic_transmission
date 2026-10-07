"""Source-disjoint caption holdout and separately authored source audits; no training."""
import argparse
from contextlib import ExitStack
import csv
import fcntl
from pathlib import Path
import random
import subprocess
import time

from .qwen35_training_data import ROOT, read_json, write_json, sha256, exclusion_keys
from .qwen35_astra_transfer import display_sheet, load_model
from .qwen35_ablation import encode, generate
from .qwen35_training import release_gpu
from .qwen35_evidence_repair_eval import check_checkpoint, input_row

RUN = ROOT/'outputs/qwen35_caption_holdout_20261007'
OLD = ROOT/'outputs/qwen35_evidence_repair_v3'
ARMS = ['initial', 'caption']


def save_frozen(path, obj):
    if path.exists() and read_json(path) != obj:
        raise ValueError(f'Frozen artifact changed: {path}')
    write_json(path, obj)


def source_exclusions():
    ids, hashes, evidence = exclusion_keys()
    manifest = ROOT/'outputs/qwen35_multitask_data_v1/source_split.json'
    for row in read_json(manifest)['sources']:
        ids.add(row['source_id']); hashes.add(row['source_sha256'])
    evidence.append(dict(path=str(manifest), sha256=sha256(manifest)))
    # All local Qwen experiment input manifests, including old tests/diagnostics.
    # Read metadata only; do not reuse any of their images as fresh holdout.
    names = {'train.json','validation.json','test.json','regression.json','train_teacher.json',
             'train_inputs.json','validation_inputs.json','test_inputs.json','protocol.json'}
    for folder in sorted((ROOT/'outputs').glob('qwen35*')):
        if folder == RUN or not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.json')):
            if path.name not in names:
                continue
            doc = read_json(path)
            rows = doc.get('records', doc.get('rows', doc.get('sources', [])))
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        if row.get('source_id'): ids.add(row['source_id'])
                        if row.get('source_sha256'): hashes.add(row['source_sha256'])
            evidence.append(dict(path=str(path), sha256=sha256(path)))
    return ids, hashes, evidence


def source_record(row, sheet):
    return dict(id=row['id'], fps=row['fps'], keys=row['keys'], sheet=str(sheet),
                sheet_sha256=sha256(sheet), frames=[dict(position=p, **row['frames'][k]) for p,k in enumerate(row['keys'])])


def prepare():
    if (RUN/'input_freeze.json').exists():
        return verify_inputs()
    ids, hashes, evidence = source_exclusions()
    paths = sorted(p for p in (ROOT/'data/etri_benchmark_v1_20260924/raw/clipshots').glob('*.mp4') if p.stem not in ids)
    random.Random(20261007).shuffle(paths)
    rows=[]; sources=[]; skipped=[]; seen=set(hashes)
    for path in paths:
        if len(sources) == 12: break
        h=sha256(path)
        if h in seen:
            skipped.append(dict(path=str(path),reason='duplicate or excluded source hash')); continue
        meta=read_json_string(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(path)],text=True))
        duration=float(meta['format']['duration'])
        if duration < 8:
            skipped.append(dict(path=str(path),reason='less than eight seconds')); continue
        number=len(sources); seen.add(h)
        src=dict(id=f'h{number:02d}',source_id=path.stem,source_path=str(path.resolve()),source_sha256=h,duration=duration,dataset='ClipShots')
        sources.append(src)
        for w,fraction in enumerate([.25,.75]):
            rid=f'h{number:02d}_w{w}'; start=round(max(0,min(duration-4,fraction*duration-1.75))*24)/24
            out=RUN/'frames'/rid; out.mkdir(parents=True,exist_ok=True)
            # Eight frames spaced at 0.5s; evaluation exposes only [0,2,5,7].
            subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','2','-ss',str(start),'-i',str(path),
                '-vf','fps=24,select=not(mod(n\\,12)),scale=576:-2','-vsync','0','-frames:v','8','-start_number','0','-threads','2','-y',str(out/'%d.png')],check=True)
            frames=[dict(index=round(start*24)+i*12,path=str(out/f'{i}.png'),sha256=sha256(out/f'{i}.png')) for i in range(8)]
            row=dict(id=rid,source_id=path.stem,source_sha256=h,fps=24,frames=frames,keys=[0,2,5,7],window=w,split='fresh_holdout',start_seconds=start)
            sheet=display_sheet(row,row['keys'],RUN/'sheets'/f'{rid}.png')
            rows.append(row)
        print('PREPARED_SOURCE',src['id'],flush=True)
    if len(sources)!=12: raise ValueError('Insufficient eligible sources')
    prior=read_json(OLD/'evaluation/protocol.json')
    protocol=dict(seed=20261007,arms=ARMS,rows=rows,sources=sources,config=prior['config'],
        checkpoints={a:prior['checkpoints'][a] for a in ARMS},exclusion_evidence=evidence,
        selection=dict(eligible_by_id=len(paths),selected_sources=12,skipped=skipped,
            rule='Seeded shuffle before viewing images or captions; first twelve decodable >=8s unique originals; two fixed windows per original.'),
        scope='Fresh to recorded Qwen caption training/development/evaluation and excluded ETRI evaluation originals. ClipShots domain shift; prior automated shot scanning/acquisition does not imply semantic review. Foundation pretraining overlap unknown.',
        limitations='Source ID and exact raw SHA256 exclusion; differently encoded or related-program near duplicates are not proven absent.',
        input_policy='Same four keys, pixels, timestamps, caption prompt, NF4 precision and160-token cap. Inference receives no references.',
        checkpoint_policy='Previous Astra rank32 parent versus completed caption-only epoch2; selected before this source review, no epoch tuning.',
        gate='Full facts/actions may not decrease and erroneous windows/failures may not increase; at least one strict gain. Report per-original uncertainty.',
        training_updates=0,default_pipeline_changed=False)
    save_frozen(RUN/'protocol.json',protocol)
    fresh=[source_record(r,RUN/'sheets'/f'{r["id"]}.png') for r in rows]
    save_frozen(RUN/'source_packet_fresh.json',dict(records=fresh,source_only=True,instructions='Build3-6 observable atomic facts per window; include actions/changes only if supported across images. Mark uncertain identity/motion; avoid inferring motion from one frame. Do not view model captions.'))
    # Audit the17 newly repaired training targets and every development window.
    train=[r for r in read_json(OLD/'train.json')['records'] if r.get('evidence_queries')]
    dev=read_json(OLD/'validation.json')['records']; refs={r['id']:r for r in read_json(OLD/'evaluation/reference.json')['records']}
    for name,items in [('train',train),('development',dev)]:
        packet=[]
        for r in items:
            sheet=display_sheet(r,r['keys'],RUN/'audit_sheets'/f'{r["id"]}.png')
            item=source_record(r,sheet)
            item.update(caption=r['caption'],evidence_queries=r.get('evidence_queries',[]),claims=r.get('claims',[]),preference_pairs=r.get('preference_pairs',[]),facts=refs[r['id']]['facts'] if name=='development' else [])
            packet.append(item)
        save_frozen(RUN/f'source_packet_{name}.json',dict(records=packet,source_only=True,
            scope='Independent reinspection of original source and labels; model outputs and prior scores withheld. Preserve uncertain observations. The17 repaired training windows are audited, not all120 inherited training windows.'))
    files=['protocol.json',*[f'source_packet_{name}.json' for name in ['fresh','train','development']]]
    sheets={r['sheet']:r['sheet_sha256'] for name in ['fresh','train','development'] for r in read_json(RUN/f'source_packet_{name}.json')['records']}
    dependencies={n:sha256(Path(__file__).with_name(n)) for n in ['qwen35_ablation.py','qwen35_astra_transfer.py','qwen35_training.py','qwen35_evidence_repair_eval.py']}
    save_frozen(RUN/'input_freeze.json',dict(files={n:sha256(RUN/n) for n in files},sheets=sheets,implementation_sha256=sha256(Path(__file__)),dependencies=dependencies))
    write_json(RUN/'RESULT.json',dict(status='INPUTS_FROZEN_INDEPENDENT_REVIEW_PENDING',fresh_sources=12,fresh_windows=24,training_labels_to_audit=17,development_windows_to_audit=16,training_updates=0,default_pipeline_changed=False))
    return verify_inputs()


def read_json_string(text):
    import json
    return json.loads(text)


def verify_inputs():
    f=read_json(RUN/'input_freeze.json')
    if f['implementation_sha256']!=sha256(Path(__file__)): raise ValueError('Changed frozen evaluation code')
    for name,h in f['files'].items():
        if sha256(RUN/name)!=h: raise ValueError('Changed input artifact')
    for name,h in f['sheets'].items():
        if sha256(name)!=h: raise ValueError('Changed source sheet')
    p=read_json(RUN/'protocol.json')
    for name,h in f['dependencies'].items():
        if sha256(Path(__file__).with_name(name))!=h: raise ValueError('Changed inference dependency')
    for source in p['sources']:
        if sha256(source['source_path'])!=source['source_sha256']: raise ValueError('Changed raw video')
    for ck in p['checkpoints'].values():
        path=Path(ck['path'])
        if sha256(path/'complete.json')!=ck['complete_sha256']: raise ValueError('Changed checkpoint manifest')
        check_checkpoint(path)
    for row in p['rows']:
        for frame in row['frames']:
            if sha256(frame['path'])!=frame['sha256']: raise ValueError('Changed pixels')
    return p


def validate_record(row,record,ph):
    if record['id']!=row['id'] or record['protocol_sha256']!=ph or set(record['cells'])!=set(ARMS):
        raise ValueError('Missing or wrong condition')
    first=record['cells']['initial']
    for c in record['cells'].values():
        if c['keys']!=row['keys'] or c['source_hashes']!=[row['frames'][k]['sha256'] for k in row['keys']]: raise ValueError('Mismatched source input')
        if c['input_tokens']!=first['input_tokens'] or c['grid']!=first['grid']: raise ValueError('Mismatched input budget')


def run():
    import torch
    p=verify_inputs(); ph=sha256(RUN/'protocol.json')
    # Freeze the independently authored reference before any new captions.
    reference=read_json(RUN/'reference_freeze.json')
    for name,h in reference['files'].items():
        if sha256(RUN/name)!=h: raise ValueError('Changed independent reference')
    pending=[]
    for row in p['rows']:
        path=RUN/'records'/f'{row["id"]}.json'
        if path.exists():
            if sha256(path)!=read_json(path.with_suffix('.receipt.json'))['sha256']: raise ValueError('Changed caption')
            validate_record(row,read_json(path),ph)
        else: pending.append(row)
    if pending:
        model,processor,runtime=load_model(p['config'],Path(p['checkpoints']['initial']['path'])/'adapter')
        model.load_adapter(str(Path(p['checkpoints']['caption']['path'])/'adapter'),adapter_name='caption',is_trainable=False)
        model.requires_grad_(False);model.gradient_checkpointing_disable();model.eval();torch.cuda.reset_peak_memory_stats()
        for row in pending:
            clean=input_row(row); start=time.monotonic(); features=encode(model,processor,clean,p['config']);torch.cuda.synchronize();elapsed=time.monotonic()-start
            order=ARMS[:] if p['rows'].index(row)%2==0 else ARMS[::-1]; cells={}
            for arm in order:
                model.set_adapter('default' if arm=='initial' else arm);model.requires_grad_(False);model.eval()
                if any(v.requires_grad for v in model.parameters()): raise ValueError('Evaluation must be frozen')
                c=generate(model,processor,clean,row['keys'],features,p['config'])
                if c['input_tokens']>p['config']['max_input_tokens']: raise ValueError('Input token limit exceeded')
                cells[arm]=dict(c,keys=row['keys'],source_hashes=[row['frames'][k]['sha256'] for k in row['keys']])
                print('EVAL',row['id'],arm,c['status'],flush=True)
            record=dict(id=row['id'],cells=cells,order=order,encode_seconds=elapsed,runtime=runtime,protocol_sha256=ph,reference_freeze_sha256=sha256(RUN/'reference_freeze.json'),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            validate_record(row,record,ph); path=RUN/'records'/f'{row["id"]}.json';write_json(path,record);write_json(path.with_suffix('.receipt.json'),dict(sha256=sha256(path)));del features
        del model,processor;release_gpu()
    verify_inputs()
    files={}
    for row in p['rows']:
        name=f'records/{row["id"]}.json';validate_record(row,read_json(RUN/name),ph);files[name]=sha256(RUN/name)
    save_frozen(RUN/'generation_freeze.json',dict(protocol_sha256=ph,reference_freeze_sha256=sha256(RUN/'reference_freeze.json'),files=files))
    print('GENERATED',len(p['rows'])*len(ARMS),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','verify','generate']);args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='verify':verify_inputs();print('VERIFIED')
    else:
        with ExitStack() as stack:
            for name in ['/tmp/qwen35_multitask_gpu.lock',str(ROOT/'.local/etri_60s_check.lock')]:
                lock=stack.enter_context(open(name,'w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run()
