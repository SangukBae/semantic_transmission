#!/usr/bin/env python3
"""Generate provisional source shot candidates; these are not human ground truth.

Uses the upstream TransNetV2 PyTorch architecture with a recorded converted
checkpoint. Original PTS are kept, and only the central 50 frames of each
100-frame inference window are used, as in the upstream inference procedure.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from etri_benchmark import DATA, all_sources, probe, run, save, sha


def intervals(pred, pts, threshold):
    mask = pred > threshold
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [{"start_frame": int(a), "end_frame": int(b),
             "start_sec": float(pts[a]), "end_sec": float(pts[b]),
             "peak_sec": float(pts[a + np.argmax(pred[a:b+1])]),
             "probability": float(pred[a:b+1].max())} for a,b in zip(starts,ends)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['TVSum','ClipShots'])
    parser.add_argument('--source-id')
    args = parser.parse_args()
    root = DATA / 'metadata/transnetv2'
    spec = importlib.util.spec_from_file_location('transnetv2_upstream', root/'transnetv2_pytorch.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.TransNetV2()
    model.load_state_dict(torch.load(root/'weights.pth', map_location='cpu', weights_only=True))
    model.eval().cuda()
    torch.set_num_threads(4)
    weights_hash = sha(root/'weights.pth')
    for dataset,path in all_sources():
        if args.dataset and dataset != args.dataset: continue
        if args.source_id and path.stem != args.source_id: continue
        dest = DATA/'reports/transnet'/dataset/(path.stem+'.json')
        if dest.exists(): continue
        if float(probe(path)['format']['duration']) < 60: continue
        started = time.monotonic()
        frames = np.frombuffer(run(['ffmpeg','-v','error','-threads','2','-filter_threads','2',
            '-i',str(path),'-vf','scale=48:27','-vsync','0','-pix_fmt','rgb24','-f','rawvideo','-']).stdout,
            np.uint8).reshape(-1,27,48,3)
        pts_data = json.loads(run(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]).stdout)
        pts = np.array([float(f['best_effort_timestamp_time']) for f in pts_data['frames']])
        assert len(pts) == len(frames), (path,len(pts),len(frames))
        assert np.all(np.diff(pts)>0), path
        pts -= pts[0]
        padded = np.pad(frames,((25,25+(-len(frames))%50),(0,0),(0,0),(0,0)),mode='edge')
        pred=[]; many=[]
        with torch.inference_mode():
            for i in range(0,len(padded)-99,50):
                one,aux=model(torch.from_numpy(padded[i:i+100][None]).cuda())
                pred.append(torch.sigmoid(one)[0,25:75,0].cpu().numpy())
                many.append(torch.sigmoid(aux['many_hot'])[0,25:75,0].cpu().numpy())
        pred=np.concatenate(pred)[:len(frames)]
        many=np.concatenate(many)[:len(frames)]
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest.with_suffix('.npz'),pts=pts,single=pred,many=many)
        row={'dataset':dataset,'source_id':path.stem,'source_sha256':sha(path),
             'weights_sha256':weights_hash,'status':'MODEL_CANDIDATES_NOT_GROUND_TRUTH',
             'decoded_frames':len(frames),'last_pts_sec':float(pts[-1]),
             'events':intervals(pred,pts,.5),'low_confidence_events':intervals(pred,pts,.2),
             'wall_seconds':time.monotonic()-started}
        save(dest,row)
        print(json.dumps({k:row[k] for k in ['dataset','source_id','decoded_frames','wall_seconds']} |
                         {'events':len(row['events'])}),flush=True)


if __name__=='__main__':main()
