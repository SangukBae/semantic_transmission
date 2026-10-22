#!/usr/bin/env python3
"""Build and verify a frozen, class-balanced Kinetics-400 evaluation subset.

Raw CVDF videos and normalized inputs remain outside Git. All paths in the
published manifest are relative so the same manifest works on both hosts.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request

BASE = 'https://s3.amazonaws.com/kinetics/400/'
DEFAULT = Path('/home/sangukbae/datasets/semantic_transmission/kinetics400_eval1200_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def identity(row):
    return f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"


def rank_candidates(rows, excluded_ids, seed, reserve=8):
    grouped = collections.defaultdict(list)
    for row in rows:
        if row['youtube_id'] not in excluded_ids:
            grouped[row['label']].append(row)
    selected = {}
    for label, items in sorted(grouped.items()):
        items.sort(key=lambda r: hashlib.sha256(f'{seed}:{identity(r)}'.encode()).hexdigest())
        unique, seen = [], set()
        for row in items:
            if row['youtube_id'] not in seen:
                unique.append(row); seen.add(row['youtube_id'])
        selected[label] = unique[:reserve]
    return selected


def plan(root, per_class=3, seed=20261022):
    metadata = root/'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    for name, suffix in [('val.csv','annotations/val.csv'), ('archives.txt','val/k400_val_path.txt')]:
        path = metadata/name
        if not path.exists():
            with urllib.request.urlopen(BASE+suffix, timeout=30) as response:
                path.write_bytes(response.read())
    previous = list(csv.DictReader((metadata/'existing_kinetics14.csv').open()))
    sources = read(metadata/'selector_v3_sources.json')
    excluded_ids = {r['filename'][:11] for r in previous}
    excluded_ids.update(r['youtube_id'] for r in sources if r.get('youtube_id'))
    excluded_hashes = {r['sha256'] for r in previous}
    excluded_hashes.update(r[k] for r in sources for k in ('sha256','raw_sha256') if r.get(k))
    rows = list(csv.DictReader((metadata/'val.csv').open()))
    groups = rank_candidates(rows, excluded_ids, seed, per_class+5)
    if len(groups) != 400 or any(len(v)<per_class for v in groups.values()):
        raise ValueError('400 classes with sufficient distinct sources required')
    value = dict(schema='kinetics400_eval_subset_v1', role='evaluation_only', upstream_split='validation',
        per_class=per_class, target_videos=400*per_class, seed=seed,
        labels=list(groups), candidates=groups, archives=(metadata/'archives.txt').read_text().splitlines(),
        source='https://github.com/cvdfoundation/kinetics-dataset',
        inputs={name:sha(metadata/name) for name in ('val.csv','archives.txt','existing_kinetics14.csv','selector_v3_sources.json')},
        excluded_youtube_ids=sorted(excluded_ids), excluded_sha256=sorted(excluded_hashes),
        selection='SHA256(seed:filename) order within each class; first technically valid distinct sources',
        preprocessing=dict(width=576,height=320,fps=24,filter='fps=24,scale=576:320:flags=lanczos,setsar=1',
            codec='libx264',crf=18,preset='medium',audio=False,min_duration_seconds=2,max_duration_seconds=16),
        limitations=['Not the full Kinetics benchmark or the LGVSC author 14-video cohort.',
            'Excludes exact raw hashes and known YouTube IDs; cross-dataset scene/near-duplicate overlap is not proven absent.',
            'LAVIB parent IDs do not supply a verified mapping to all original YouTube sources.',
            'Class balance is not a guarantee of crowd, occlusion, motion or camera diversity within each class.',
            'Pretrained backbone/metric model training overlap is not audited.'])
    if (root/'plan.json').exists() and read(root/'plan.json') != value:
        raise ValueError('Existing selection plan differs; choose a new output root')
    save(root/'plan.json',value)
    return value


def validate_plan(root):
    value = read(root/'plan.json')
    for name, digest in value['inputs'].items():
        if sha(root/'metadata'/name) != digest:
            raise ValueError('Frozen source metadata changed: '+name)
    return value


def status(root, stage, done, total, **extra):
    old = read(root/'status.json') if (root/'status.json').exists() else {}
    value = dict(status='RUNNING',stage=stage,done=done,total=total,pid=os.getpid(),
        started_unix=old.get('started_unix',time.time()),updated_unix=time.time(),**extra)
    save(root/'status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def fetch_archive(root, url):
    name = url.rsplit('/',1)[-1]
    dest = root/'.cache'/name
    receipt = root/'archive_records'/(name+'.json')
    if receipt.exists():
        old = read(receipt)
        if old['source'] != url:
            raise ValueError('Archive URL changed')
        if all((root/p).is_file() and sha(root/p)==h for p,h in old['extracted'].items()):
            return old
        raise ValueError('Previously extracted candidate was lost or changed')
    dest.parent.mkdir(parents=True,exist_ok=True)
    part = dest.with_suffix(dest.suffix+'.part')
    if not dest.exists():
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','5',
            '--retry-delay','2','--connect-timeout','20','--speed-time','90','--speed-limit','1024',
            '--continue-at','-','--output',str(part),url],check=True)
        part.replace(dest)
    p = read(root/'plan.json')
    wanted = {identity(row) for group in p['candidates'].values() for row in group}
    extracted = {}
    with tarfile.open(dest,'r:gz') as archive:
        for item in archive:
            name = Path(item.name).name
            if not item.isfile() or name not in wanted:
                continue
            output = root/'candidates'/name
            output.parent.mkdir(exist_ok=True)
            temp = output.with_suffix('.mp4.part')
            with archive.extractfile(item) as source, temp.open('wb') as target:
                shutil.copyfileobj(source,target)
            digest = sha(temp)
            if output.exists() and sha(output)!=digest:
                raise ValueError('Conflicting archive members: '+name)
            temp.replace(output)
            extracted[str(output.relative_to(root))] = digest
    value = dict(source=url,bytes=dest.stat().st_size,sha256=sha(dest),extracted=extracted,complete=True)
    save(receipt,value)
    return value


def probe(path, count=False):
    command = ['ffprobe','-v','error','-select_streams','v:0']
    if count:command += ['-count_frames']
    command += ['-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration',
                '-of','json',str(path)]
    value = json.loads(subprocess.check_output(command,text=True,stderr=subprocess.PIPE,timeout=90))
    stream = value['streams'][0]
    stream['duration'] = float(stream.get('duration',value.get('format',{}).get('duration',0)))
    return stream


def normalize(root, row, p):
    name = identity(row)
    source = root/'candidates'/name
    raw = root/'raw'/name
    output = root/'processed'/name
    receipt = root/'records'/(name+'.json')
    if receipt.exists():
        old = read(receipt)
        if old['plan_sha256'] != sha(root/'plan.json') or sha(raw)!=old['raw_sha256'] or sha(output)!=old['sha256']:
            raise ValueError('Previously normalized input changed: '+name)
        return old
    if not source.is_file():
        raise FileNotFoundError('not present in CVDF validation archives')
    digest = sha(source)
    if digest in p['excluded_sha256']:
        raise ValueError('raw hash overlaps training or previous evaluation')
    info = probe(source)
    spec = p['preprocessing']
    if not spec['min_duration_seconds'] <= info['duration'] <= spec['max_duration_seconds']:
        raise ValueError('duration outside registered technical bounds')
    for folder in ('raw','processed','records'):(root/folder).mkdir(exist_ok=True)
    temp = output.with_suffix('.part.mp4')
    command = ['ffmpeg','-v','error','-xerror','-nostdin','-y','-threads','2','-i',str(source),
        '-map','0:v:0','-an','-sn','-vf',spec['filter'],'-c:v',spec['codec'],'-crf',str(spec['crf']),
        '-preset',spec['preset'],'-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=180)
    actual = probe(temp,True)
    if (actual['width'],actual['height'],actual['avg_frame_rate']) != (576,320,'24/1'):
        raise ValueError('wrong normalized video geometry')
    subprocess.run(['ffmpeg','-v','error','-xerror','-nostdin','-threads','2','-i',str(temp),
        '-map','0:v:0','-an','-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=90)
    if int(actual['nb_read_frames'])<48:
        raise ValueError('normalized clip too short')
    shutil.copy2(source,raw)
    temp.replace(output)
    result = dict(id='kinetics_eval_'+Path(name).stem,parent='youtube:'+row['youtube_id'],
        dataset='Kinetics-400',role='test',upstream_split='validation',label=row['label'],
        label_id=p['labels'].index(row['label']),youtube_id=row['youtube_id'],
        time_start=int(row['time_start']),time_end=int(row['time_end']),
        raw_path='raw/'+name,path='processed/'+name,raw_sha256=digest,sha256=sha(output),
        frames=int(actual['nb_read_frames']),width=576,height=320,fps=24,
        duration_seconds=actual['duration'],raw_video=info,plan_sha256=sha(root/'plan.json'))
    save(receipt,result)
    return result


def verify(root):
    p = validate_plan(root)
    manifest = read(root/'manifest.json')
    rows = manifest['rows']
    counts = collections.Counter(r['label'] for r in rows)
    if (len(rows)!=p['target_videos'] or set(counts)!=set(p['labels'])
            or set(counts.values())!={p['per_class']}):
        raise ValueError('class balance or total differs from frozen plan')
    for key in ('id','youtube_id','raw_sha256','sha256'):
        if len({r[key] for r in rows})!=len(rows):
            raise ValueError('duplicate '+key)
    if {r['youtube_id'] for r in rows} & set(p['excluded_youtube_ids']):
        raise ValueError('excluded YouTube source selected')
    for row in rows:
        for key,digest in [('raw_path','raw_sha256'),('path','sha256')]:
            if sha(root/row[key])!=row[digest]:
                raise ValueError('input hash mismatch: '+row[key])
        if row['raw_sha256'] in p['excluded_sha256']:
            raise ValueError('excluded source hash selected')
    if manifest['plan_sha256']!=sha(root/'plan.json'):
        raise ValueError('manifest plan identity changed')
    return dict(status='VERIFIED',videos=len(rows),classes=len(counts),per_class=p['per_class'],
        total_frames=sum(r['frames'] for r in rows),raw_bytes=sum((root/r['raw_path']).stat().st_size for r in rows),
        processed_bytes=sum((root/r['path']).stat().st_size for r in rows),
        manifest_sha256=sha(root/'manifest.json'),plan_sha256=sha(root/'plan.json'))


def build(root, workers):
    p = validate_plan(root)
    status(root,'download_extract',0,len(p['archives']))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_archive,root,url) for url in p['archives']]
        for i,future in enumerate(cf.as_completed(futures),1):
            value = future.result()
            status(root,'download_extract',i,len(futures),archive=value['source'].rsplit('/',1)[-1])
    chosen, failures, seen_ids, seen_hashes = [], [], set(), set()
    status(root,'normalize_validate',0,p['target_videos'])
    def one_class(item):
        label,candidates = item
        selected,rejected=[],[]
        for row in candidates:
            try:
                result=normalize(root,row,p)
            except (ValueError,FileNotFoundError,subprocess.SubprocessError,IndexError,KeyError) as error:
                rejected.append(dict(filename=identity(row),label=label,error=str(error)))
                continue
            selected.append(result)
            if len(selected)==p['per_class']:break
        if len(selected)!=p['per_class']:
            raise ValueError('insufficient technically valid reserved candidates: '+label+' '+str(rejected))
        return selected,rejected
    with cf.ThreadPoolExecutor(max_workers=min(workers,4)) as pool:
        for selected,rejected in pool.map(one_class,p['candidates'].items()):
            for row in selected:
                if row['youtube_id'] in seen_ids or row['raw_sha256'] in seen_hashes:
                    raise ValueError('cross-class duplicate source; expand the deterministic reserve')
                seen_ids.add(row['youtube_id']);seen_hashes.add(row['raw_sha256'])
            chosen.extend(selected);failures.extend(rejected)
            status(root,'normalize_validate',len(chosen),p['target_videos'],label=selected[0]['label'])
    chosen.sort(key=lambda r:(r['label_id'],r['id']))
    save(root/'manifest.json',dict(schema=p['schema'],plan_sha256=sha(root/'plan.json'),role=p['role'],
        upstream_split=p['upstream_split'],selection=p['selection'],preprocessing=p['preprocessing'],
        limitations=p['limitations'],rows=chosen))
    save(root/'technical_rejections.json',failures)
    save(root/'environment.json',dict(ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        builder_sha256=sha(__file__)))
    with (root/'videos.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','path','label_id','label','sha256','raw_sha256'],lineterminator='\n')
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in chosen)
    result=verify(root)
    save(root/'verification.json',result)
    status(root,'complete',len(chosen),p['target_videos'],result=result)
    s=read(root/'status.json');s.update(status='COMPLETE',ended_unix=time.time());save(root/'status.json',s)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','build','verify','status'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--per-class',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='plan':
        value=plan(root,args.per_class,args.seed)
        print(json.dumps(dict(target_videos=value['target_videos'],classes=len(value['labels']),plan_sha256=sha(root/'plan.json'))))
    elif args.command=='verify':print(json.dumps(verify(root)))
    elif args.command=='status':print(json.dumps(read(root/'status.json'),indent=2,ensure_ascii=False))
    else:
        try:print(json.dumps(build(root,args.workers)))
        except BaseException as error:
            s=read(root/'status.json');s.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
            save(root/'status.json',s);raise


if __name__=='__main__':main()
