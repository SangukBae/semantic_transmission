"""Audit full-video schedule repair and prepare frame evidence without inference."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from semantic_transmission import schedule_repair as repair, short_schedule as fix
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.temporal import segment_lengths
from semantic_transmission.webvid5 import fingerprint, read_json
from semantic_transmission.webvid_ablation import snapshot
from semantic_transmission.video_io import probe

METRICS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    new = args.output.resolve()
    result = read_json(new / "RESULT.json")
    before = Path(result["previous"])
    original = Path(read_json(before / "run/receiver_policy.json")["paired_source"])
    roots = dict(original=original, before=before, after=new)
    dest = new / "analysis"
    dest.mkdir(exist_ok=True)
    verification = {}
    for label, root in roots.items():
        protocol = read_json(root / "execution_protocol.json")
        assert fingerprint({k:v for k,v in protocol.items() if k!="signature"}) == protocol["signature"]
        receipts = {}
        for p in (root / "stages").glob("*.json"):
            r = read_json(p)
            assert r["status"] == "PASSED" and r["identity"] == protocol["signature"]
            assert snapshot(root, r["required"]) == r["artifacts"], p
            receipts[p.stem] = r
        for r in receipts.values():
            assert all(fingerprint(receipts[k]) == v for k,v in r["dependencies"].items())
        q = read_json(root / "run/quality.json")
        video = root / "run/receiver/reconstruction/sample_0000.mp4"
        assert q["status"] == "PASSED" and sha256(video) == q["video_sha256"]
        assert sha256(root / "run/data/normalized.mp4") == q["source_sha256"]
        info = probe(video)
        assert (info["frames"],info["fps"],info["width"],info["height"]) == (240,24,576,320)
        repair.base.text.validate_usage(root / "run")
        verification[label] = dict(stage_seconds={k:r["seconds"] for k,r in receipts.items()},
            checked_stages=len(receipts), video_sha256=q["video_sha256"], video=info)
    identity, _ = repair.preflight(before, new)
    assert snapshot(new / "run", repair.INPUTS) == identity["inputs"]
    traces = [read_json(p / "run" / repair.base.noise.TRACE) for p in (before,new)]
    assert traces[1]["matched_reference"]
    assert all(traces[0][k] == traces[1][k] for k in ("records","runtime","contract"))
    assert read_json(before / "run" / repair.base.fix.TRACE) == read_json(new / "run" / repair.base.fix.TRACE)
    keys = read_json(new / "run/keyframes.json")["indices"]
    schedule = read_json(new / "run" / fix.TRACE)
    fix.validate_trace(schedule,keys,30)
    assert [s["loop"] for s in schedule if s["repaired"]] == [0]
    short_probe = repair.REPO / "outputs/etri_person_walk_vae_diagnosis_20261001_v3/schedule_probe"
    matches = [np.array_equal(np.asarray(Image.open(short_probe/f"frames/{i:02d}.png")),
        np.asarray(Image.open(new/f"run/receiver/reconstruction/sample_0000_frames/{i:05d}.png"))) for i in range(9)]
    assert all(matches), "Full-run beginning differs from the paired short probe"
    boundaries = {}
    for boundary in ("delivered_mp4","lossless_frames"):
        tables = {}
        for label,root in roots.items():
            with (root/f"run/quality_{boundary}.csv").open() as f:
                rows = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
            assert [r["frame"] for r in rows] == list(range(240))
            q = read_json(root/"run/quality.json")[boundary]
            assert all(abs(float(np.mean([r[k] for r in rows]))-q[k])<1e-8 for k in METRICS)
            tables[label] = rows
        boundaries[boundary] = tables
    values = boundaries["delivered_mp4"]
    def mean(label, ids):
        return {k:float(np.mean([values[label][i][k] for i in ids])) for k in METRICS}
    groups = {"first_0_8":range(9), "following_9_32":range(9,33), "rest_33_239":range(33,240),
              "second_half_120_239":range(120,240), "whole_0_239":range(240)}
    regions = {name:{label:mean(label,ids) for label in roots} for name,ids in groups.items()}
    segments=[]
    for i,(a,b) in enumerate(zip(keys,keys[1:])):
        ids=list(range(a if i==0 else a+1,b+1))
        q={label:mean(label,ids) for label in roots}
        segments.append(dict(loop=i,frames=[ids[0],ids[-1]],metrics=q,
            lpips_delta=q['after']['lpips_vgg']-q['before']['lpips_vgg']))
    changes = {i:values['after'][i]['lpips_vgg']-values['before'][i]['lpips_vgg'] for i in range(33,240)}
    def spread(reverse):
        chosen=[]
        for i in sorted(changes,key=changes.get,reverse=reverse):
            if all(abs(i-j)>=8 for j in chosen):chosen.append(i)
            if len(chosen)==6:break
        return sorted(chosen)
    frames = dict(first=list(range(9)), following=[9,12,24], known_errors=[81,128,133,176,196,208],
                  coverage=[48,96,144,168,216,239], losses=spread(True), gains=spread(False))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    evidence=[]
    for group,ids in frames.items():
        for page,start in enumerate(range(0,len(ids),3)):
            these=ids[start:start+3]
            pic=Image.new('RGB',(1536,320*len(these)),(20,24,30));d=ImageDraw.Draw(pic)
            for row,f in enumerate(these):
                paths=[original/f"run/data/frames/sample/{f}.png",*[root/f"run/receiver/reconstruction/sample_0000_frames/{f:05d}.png" for root in (before,new)]]
                for col,(title,path) in enumerate(zip(('SOURCE','BEFORE SCHEDULE FIX','AFTER SCHEDULE FIX'),paths)):
                    pic.paste(Image.open(path).convert('RGB').resize((512,284)),(col*512,row*320+36))
                    d.text((col*512+5,row*320+2),f'{title} | f{f} | {f/24:.3f}s',font=font,fill='white')
                    if col:
                        v=values['before' if col==1 else 'after'][f]
                        d.text((col*512+5,row*320+18),f"MP4 PSNR {v['psnr_db']:.2f}  LPIPS {v['lpips_vgg']:.3f}",font=font,fill='white')
            path=dest/f'{group}_{page:02d}.jpg';pic.save(path,quality=95)
            evidence.append(dict(path=path.name,frames=these,sha256=sha256(path),viewed=False))
    timing={label:dict(generation_seconds=v['stage_seconds']['reconstruct'],
        stage_seconds_sum=sum(v['stage_seconds'].values()),t5_preparation_seconds=v['stage_seconds'].get('prepare-text',0))
        for label,v in verification.items()}
    analysis=dict(status='NUMERIC_AUDIT_COMPLETE_VISUAL_REVIEW_PENDING',previous=str(before),current=str(new),original=str(original),
        verification=verification,regions=regions,segments=segments,timing=timing,
        noise_scopes=len(traces[1]['records']),noise_draws=sum(len(r['draws']) for r in traces[1]['records']),
        condition_masks_identical=True,received_inputs_identical=True,prepared_text_identical=True,
        only_schedule_repaired_segments=[0],first_nine_match_short_probe=matches,
        quality={label:read_json(root/'run/quality.json')['delivered_mp4'] for label,root in roots.items()},
        segments_lpips_improved=sum(s['lpips_delta']<0 for s in segments),
        segments_lpips_regressed=sum(s['lpips_delta']>0 for s in segments),
        evidence=evidence,review_frames=sorted(set(i for ids in frames.values() for i in ids)),
        frame_metrics={str(i):{label:values[label][i] for label in roots} for i in range(240)},
        additional_channel_uses=0,independent_semantic_review='PENDING',hallucination_mitigation_verified=False,
        limitations=['One video and seed; frame selection combines previous errors, coverage and metric extremes.',
            'Later output changes through generated history although later schedules and all noise draws are unchanged.',
            'Single-run timing is not a controlled speed benchmark.'])
    write_json(dest/'analysis.json',analysis)
    print(dict(quality=analysis['quality'],regions=regions,timing=timing,segments_improved=analysis['segments_lpips_improved'],
               segments_regressed=analysis['segments_lpips_regressed'],frames=frames))


if __name__=='__main__':
    main()
