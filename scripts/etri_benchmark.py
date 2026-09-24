#!/usr/bin/env python3
"""Source-only ETRI benchmark curation, preparation, and audit utilities.

This tool does not infer hallucination truth from quality metrics or source cuts.
Existing pilot artifacts are read-only inputs.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "data/etri_long_video_20260924"
DATA = REPO / "data/etri_benchmark_v1_20260924"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace")[-2000:])
    return p


def probe(path, count=False):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count:
        cmd.append("-count_frames")
    cmd += ["-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,start_time:format=duration",
            "-of", "json", str(path)]
    return json.loads(run(cmd).stdout)


def all_sources():
    sources = [("TVSum", p) for p in sorted((PILOT / "raw/tvsum").glob("*.mp4"))]
    clips = {p.name: p for p in (PILOT / "raw/clipshots").glob("*.mp4")}
    clips.update({p.name: p for p in (DATA / "raw/clipshots").glob("*.mp4")})
    sources += [("ClipShots", p) for p in sorted(clips.values())]
    return sources


def extract_prefix():
    """Recover complete MP4 tar members only; partial gzip is intentionally bounded."""
    archive = DATA/'archives/clipshots_a_prefix_2g.gz.part'
    receipts=[]; incomplete=None
    try:
        with tarfile.open(archive,mode='r|gz') as tar:
            for member in tar:
                name=Path(member.name)
                if not member.isfile() or name.suffix.lower()!='.mp4': continue
                if name.is_absolute() or '..' in name.parts: raise ValueError(member.name)
                dest=DATA/'raw/clipshots'/name.name
                old=PILOT/'raw/clipshots'/name.name
                if old.exists() and old.stat().st_size==member.size:
                    receipts.append({'member':member.name,'path':str(old.resolve()),'bytes':member.size,'reuse_pilot':True})
                    continue
                if dest.exists() and dest.stat().st_size==member.size:
                    receipts.append({'member':member.name,'path':str(dest.resolve()),'bytes':member.size,'reuse_pilot':False})
                    continue
                temp=dest.with_suffix('.mp4.part'); incomplete=str(temp)
                with tar.extractfile(member) as src, temp.open('wb') as target:
                    while True:
                        block=src.read(4*1024**2)
                        if not block: break
                        target.write(block)
                if temp.stat().st_size!=member.size: raise EOFError('Incomplete final video')
                temp.replace(dest); incomplete=None
                receipts.append({'member':member.name,'path':str(dest.resolve()),'bytes':member.size,'reuse_pilot':False})
    except (EOFError,tarfile.ReadError) as error:
        termination=f'EXPECTED_BOUNDED_ARCHIVE_END: {error}'
    else:
        termination='ARCHIVE_END'
    save(DATA/'reports/extraction.json',{'archive':str(archive.resolve()),'bytes':archive.stat().st_size,
        'sha256':sha(archive),'termination':termination,'incomplete_excluded':incomplete,
        'complete_videos':receipts,'complete_count':len(receipts)})
    print(json.dumps({'complete':len(receipts),'incomplete':incomplete}),flush=True)


def scan_one(job):
    import cv2
    dataset, path = job
    dest = DATA / "reports/scans" / dataset / (path.stem + ".json")
    if dest.exists():
        return json.loads(dest.read_text())
    p = probe(path)
    duration = float(p["format"]["duration"])
    row = {"dataset": dataset, "source_id": path.stem, "source_path": str(path.resolve()),
           "source_sha256": sha(path), "probe": p, "duration_sec": duration}
    if duration < 60:
        row["eligibility"] = "SHORTER_THAN_60_SECONDS"
        save(dest, row)
        return row
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-threads", "2", "-filter_threads", "2", "-i", str(path),
           "-an", "-vf", "scale=256:-2,select='gt(scene,0.18)',showinfo", "-vsync", "vfr", "-f", "null", "-"]
    log = run(cmd).stderr.decode(errors="replace")
    candidates = [float(t) for t in re.findall(r"pts_time:([\d.]+)", log)]
    grouped = []
    for t in candidates:
        if not grouped or t - grouped[-1] >= .5:
            grouped.append(t)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-filter_threads", "2", "-i", str(path),
           "-vf", "fps=2,scale=32:32", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    frames = np.frombuffer(run(cmd).stdout, np.uint8).reshape(-1, 32, 32)
    gray = frames.astype(np.float32)
    change = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    phashes = []
    for frame in frames[::4]:
        block = cv2.dct(frame.astype(np.float32))[:8, :8].flatten()
        bits = block > np.median(block[1:])
        phashes.append(f"{sum(int(b) << i for i,b in enumerate(bits)):016x}")
    windows = []
    for start in range(0, int(duration - 60) + 1, 10):
        c = [t-start for t in grouped if start < t < start+60]
        sample = frames[start*2:(start+60)*2]
        motion = change[start*2:(start+60)*2-1]
        black_fraction = float(np.mean(sample.mean(axis=(1,2)) < 8))
        flat_fraction = float(np.mean(sample.std(axis=(1,2)) < 5))
        static_fraction = float(np.mean(motion < .1))
        level = "low" if len(c) <= 1 else ("medium" if len(c) <= 8 else "high")
        windows.append({"start_sec": start, "end_sec": start+60, "candidate_count": len(c),
            "candidate_times_sec": c, "automatic_level": level, "median_motion": float(np.median(motion)),
            "black_fraction": black_fraction, "flat_fraction": flat_fraction,
            "static_fraction": static_fraction,
            "technical_eligible": black_fraction <= .05 and flat_fraction <= .05 and static_fraction < .8})
    row.update(eligibility="CANDIDATE_REQUIRES_REVIEW", candidate_times_source_sec=grouped,
               phash_2sec=phashes, windows=windows)
    save(dest, row)
    return row


def scan(dataset=None):
    jobs = [(d,p) for d,p in all_sources() if dataset is None or d == dataset]
    rows = []
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(scan_one, jobs):
            rows.append(row)
            print(json.dumps({"dataset":row["dataset"], "id":row["source_id"],
                              "duration":round(row["duration_sec"],2)}), flush=True)
    save(DATA / "reports" / ("scan_" + (dataset or "all") + ".json"), rows)


def curate(dataset, per_level=10):
    """Deterministic, source-only candidate allocation, pending visual review."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    info={r['video_id']:r for r in csv.DictReader((DATA/'metadata/tvsum_source_info.tsv').open(),delimiter='\t')}
    excluded_path=DATA/'metadata/exclusions.json'
    excluded=json.loads(excluded_path.read_text()) if excluded_path.exists() else {}
    options=[]
    for scanpath in sorted((DATA/'reports/scans'/dataset).glob('*.json')):
        row=json.loads(scanpath.read_text())
        if row['source_id'] in excluded: continue
        predpath=DATA/'reports/transnet'/dataset/scanpath.name
        if not predpath.exists(): continue
        pred=json.loads(predpath.read_text())
        a,b=map(float,row['probe']['streams'][0]['avg_frame_rate'].split('/'))
        if a/b < 23.9: continue
        best={}
        for window in row.get('windows',[]):
            if not window['technical_eligible']: continue
            start,end=window['start_sec'],window['end_sec']
            events=[e for e in pred['events'] if start<e['peak_sec']<end]
            weak=[e for e in pred['low_confidence_events'] if start<e['peak_sec']<end]
            count=len(events)
            level='low' if count<=1 else ('medium' if count<=8 else 'high')
            if level=='low' and len(weak)>2: continue
            # Prefer well inside count strata and avoid the opening title period.
            target={'low':0,'medium':5,'high':20}[level]
            cost=abs(count-target)*.3+max(0,20-start)*.1+window['black_fraction']*20+window['static_fraction']*2
            cost+=(len(weak)-count)*.15 + start*.0005
            jitter=int(hashlib.sha256(('20260924'+row['source_id']).encode()).hexdigest()[:8],16)/2**32
            candidate={k:row[k] for k in ['dataset','source_id','source_path','source_sha256','duration_sec','probe']}
            candidate.update(level=level,window=window,transitions=events,weak_transition_count=len(weak),
                shot_count=count,classification_status='MODEL_CANDIDATE_PENDING_VISUAL_REVIEW',cost=cost+jitter,
                category=info.get(row['source_id'],{}).get('category','UNREVIEWED'),
                title=info.get(row['source_id'],{}).get('title',row['source_id']))
            if level not in best or candidate['cost']<best[level]['cost']: best[level]=candidate
        options.extend(best.values())
    levels=['low','medium','high']; sources=sorted({r['source_id'] for r in options})
    matrix=[];lower=[];upper=[]
    for level in levels:
        matrix.append([r['level']==level for r in options]);lower.append(per_level);upper.append(per_level)
    for source in sources:
        matrix.append([r['source_id']==source for r in options]);lower.append(0);upper.append(1)
    # At most three videos from a TVSum category in one transition stratum.
    if dataset=='TVSum':
        for level in levels:
            for cat in sorted({r['category'] for r in options}):
                matrix.append([r['level']==level and r['category']==cat for r in options]);lower.append(0);upper.append(3)
    result=milp(np.array([r['cost'] for r in options]),integrality=np.ones(len(options)),
        bounds=Bounds(0,1),constraints=LinearConstraint(np.array(matrix,float),lower,upper),options={'time_limit':60})
    if not result.success:
        counts={level:sum(r['level']==level for r in options) for level in levels}
        raise ValueError(f'No feasible balanced allocation: {dataset} {counts} {result.message}')
    chosen=[r for r,x in zip(options,result.x) if x>.5]
    for level in levels:
        for i,r in enumerate(sorted([x for x in chosen if x['level']==level],key=lambda x:x['source_id'])):
            r['curation_id']=('tv' if dataset=='TVSum' else 'cs')+'_'+level+f'_{i+1:02}'
    save(DATA/'metadata'/(dataset.lower()+'_candidates_v2.json'),sorted(chosen,key=lambda r:r['curation_id']))
    save(DATA/'reports'/(dataset.lower()+'_candidate_pool_v2.json'),options)
    print(json.dumps({'dataset':dataset,'selected':len(chosen),'eligible_source_level_options':len(options)}))


def sheet(path, output, times, width=240):
    height = 155
    result = Image.new("RGB", (width*4, height*math.ceil(len(times)/4)), "#1d2430")
    draw = ImageDraw.Draw(result)
    for i,t in enumerate(times):
        cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-filter_threads", "1", "-ss", str(max(0,t)),
            "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:130:force_original_aspect_ratio=decrease,pad={width}:130:(ow-iw)/2:(oh-ih)/2",
            "-f", "image2pipe", "-vcodec", "png", "-"]
        img = Image.open(io.BytesIO(run(cmd).stdout)).convert("RGB")
        x,y = i%4*width, i//4*height
        result.paste(img,(x,y)); draw.text((x+4,y+132),f"{t:.3f}s",fill="white")
    output.parent.mkdir(parents=True,exist_ok=True)
    result.save(output)


def fps_time_mapping(log, start, frame_count):
    """Read the actual FPS filter's source-frame choices, including CFR repeats."""
    recent={}; rows=[]
    for line in log.splitlines():
        match=re.search(r'Read frame with in pts (-?\d+), out pts (-?\d+)',line)
        if match: recent[int(match[2])]=int(match[1])/1_000_000
        match=re.search(r'Writing frame with pts (-?\d+) to pts (-?\d+)',line)
        if match:
            frame=int(match[2])
            if 0<=frame<frame_count:
                relative=recent[int(match[1])]
                rows.append({'output_frame':frame,'output_time_sec':frame/24,
                             'source_time_sec':start+relative,
                             'sampling_error_sec':relative-frame/24})
    if [r['output_frame'] for r in rows]!=list(range(frame_count)):
        raise ValueError('FPS filter mapping does not cover every output frame')
    return rows


def build_one(row):
    dest=DATA/'processed'/(row['id']+'.mp4')
    report=DATA/'reports/preparation'/(row['id']+'.json')
    if report.exists():
        old=json.loads(report.read_text())
        same_media=all(old[k]==row[k] for k in ['source_sha256','clip_start_sec','clip_duration_sec'])
        if same_media and dest.exists() and sha(dest)==old['processed_sha256']:
            old.update(row)
            old['selection_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            save(report,old)
            return old
    source=Path(row['source_path']); start=row['clip_start_sec']; duration=row['clip_duration_sec']
    assert sha(source)==row['source_sha256'], source
    assert float(probe(source)['format']['duration'])>=start+duration
    source_report=DATA/'reports/source_validation'/row['dataset']/(row['source_id']+'.json')
    if not source_report.exists():
        validation=run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(source),'-map','0:v:0','-f','null','-'])
        assert not validation.stderr.strip(), validation.stderr
        save(source_report,{'source_sha256':row['source_sha256'],'full_source_decode':'PASS'})
    else:
        assert json.loads(source_report.read_text())['source_sha256']==row['source_sha256']
    pts=np.load(DATA/'reports/transnet'/row['dataset']/(row['source_id']+'.npz'))['pts']
    selected_pts=pts[(pts>=start)&(pts<=start+duration)]
    assert len(selected_pts)>1 and np.diff(selected_pts).max()<.125, 'Source timestamp gap'
    frames=round(duration*24)
    cmd=['ffmpeg','-loglevel','debug','-nostdin','-y','-threads','2','-filter_threads','2',
         '-ss',str(start),'-i',str(source),'-t',str(duration),'-frames:v',str(frames),'-an',
         '-vf','settb=AVTB,fps=fps=24:start_time=0:round=near,scale=576:320:force_original_aspect_ratio=decrease,pad=576:320:(ow-iw)/2:(oh-ih)/2,setsar=1',
         '-c:v','libx264','-threads','2','-preset','fast','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    log=run(cmd).stderr.decode(errors='replace')
    report.parent.mkdir(parents=True,exist_ok=True)
    report.with_suffix('.log').write_text(log)
    mapping=fps_time_mapping(log,start,frames)
    save(DATA/'metadata/time_mapping'/(row['id']+'.json'),mapping)
    p=probe(dest,count=True); stream=p['streams'][0]
    assert int(stream['nb_read_frames'])==frames, (dest,p)
    assert abs(float(p['format']['duration'])-duration)<.001, (dest,p)
    assert stream['width']==576 and stream['height']==320 and stream['avg_frame_rate']=='24/1'
    validation=run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(dest),'-map','0:v:0','-f','null','-'])
    assert not validation.stderr.strip(), validation.stderr
    times=[r['source_time_sec'] for r in mapping]
    assert all(b>=a for a,b in zip(times,times[1:])), 'Non-monotonic mapping'
    assert max(abs(r['sampling_error_sec']) for r in mapping)<.09
    assert times[-1]-times[0]>=duration-.15
    duplicates=len(times)-len(set(times))
    assert duplicates<=max(2,math.ceil(frames*.003)), 'Excessive frame duplication'
    result=dict(row,selection_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
        processed_path=str(dest.resolve()),processed_sha256=sha(dest),processed_bytes=dest.stat().st_size,
        processed_probe=p,frames=frames,frame_rate=24,width=576,height=320,
        output_duration_sec=float(p['format']['duration']),cfr_repeated_source_frames=duplicates,
        source_time_span_sec=times[-1]-times[0],max_sampling_error_sec=max(abs(r['sampling_error_sec']) for r in mapping),
        source_max_frame_gap_sec=float(np.diff(selected_pts).max()),full_output_decode='PASS',
        encoding='H.264 CRF 0, YUV420p, lossless codec stage after spatial/time normalization',
        command=cmd,source_annotation_status='PROVISIONAL_NOT_INDEPENDENT_TRUTH',
        reconstruction_status='NOT_RUN',hallucination_annotation_status='NOT_STARTED')
    save(report,result)
    print(json.dumps({'id':row['id'],'frames':frames,'cfr_repeats':duplicates,'decode':'PASS'}),flush=True)
    return result


def build():
    rows=json.loads((DATA/'metadata/selections.json').read_text())
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(build_one,rows))
    save(DATA/'manifest.json',results)
    fields=['id','dataset','source_id','content_group','split','level','role','clip_start_sec','clip_duration_sec',
            'shot_count','source_sha256','processed_sha256','processed_path','output_duration_sec','frames',
            'cfr_repeated_source_frames','source_annotation_status','reconstruction_status']
    with (DATA/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["scan","sheet","extract-prefix","curate","build"])
    p.add_argument("--dataset",choices=["TVSum","ClipShots"])
    p.add_argument("--video",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--times",default="0,10,20,30,40,50")
    a=p.parse_args()
    if a.command=="scan":scan(a.dataset)
    elif a.command=='extract-prefix':extract_prefix()
    elif a.command=='curate':curate(a.dataset)
    elif a.command=='build':build()
    else:sheet(a.video,a.output,[float(x) for x in a.times.split(",")])


if __name__=="__main__":main()
