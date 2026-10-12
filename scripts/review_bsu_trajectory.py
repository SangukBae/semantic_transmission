#!/usr/bin/env python3
"""Export measured cadence curves and predetermined failure-review strips."""
import argparse
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))


def main():
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from semantic_transmission.artifacts import sha256,write_json
    from semantic_transmission.webvid5 import read_json
    from semantic_transmission.research_quality import read_frames
    from semantic_transmission.bsu_motion_experiment import source_window
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_trajectory_cadence_20261012_v2')
    args=parser.parse_args();root=args.root
    p=read_json(root/'protocol.json');r=read_json(root/'RESULT.json')
    if p['signature']!=r['signature']:raise ValueError('protocol mismatch')
    lookup={(x['case'],x['arm']):x for x in r['rows']};start=time.monotonic();review=[]
    figure,axes=plt.subplots(len(p['cases']),2,figsize=(15,3*len(p['cases'])),constrained_layout=True)
    for number,c in enumerate(p['cases']):
        name=c['name'];base=lookup[name,'baseline'];new=lookup[name,'combined']
        ax=axes[number,0];source=base['temporal']['per_transition']['source_motion']
        ax.plot(np.arange(1,len(source)+1)/24,source,label='Source',color='black',linewidth=1.)
        for arm,color in [('baseline','tab:orange'),('cadence','tab:green'),('combined','tab:blue')]:
            series=lookup[name,arm]['temporal']['per_transition']['generated_motion']
            ax.plot(np.arange(1,len(series)+1)/24,series,label=arm,color=color,linewidth=.9,alpha=.8)
        ax.set(title=f'{name}: motion speed',xlabel='Time (s)',ylabel='Analysis pixels / frame');ax.legend(fontsize=8)
        ax=axes[number,1]
        for arm,color in [('baseline','tab:orange'),('combined','tab:blue')]:
            series=lookup[name,arm]['temporal']['per_transition']['flow_error']
            ax.plot(np.arange(1,len(series)+1)/24,series,label=arm,color=color,linewidth=.9)
        ax.set(title=f'{name}: source-paired flow error',xlabel='Time (s)',ylabel='Analysis pixels / frame');ax.legend(fontsize=8)
        changes=np.asarray(new['temporal']['per_transition']['flow_error'])-np.asarray(base['temporal']['per_transition']['flow_error'])
        worst=int(np.argmax(changes))+1
        first=max(0,min(c['frames']-9,worst-4));indices=list(range(first,min(c['frames'],first+9)))
        arrays=[source_window(c['source_video'],*c['keys']),read_frames(root/'generation'/name/'baseline/frames'),read_frames(root/'generation'/name/'combined/frames')]
        canvas=np.zeros((3*184,288*len(indices),3),np.uint8)
        for j,(array,label) in enumerate(zip(arrays,['Source','Baseline','Combined'])):
            for k,t in enumerate(indices):
                canvas[j*184+24:(j+1)*184,k*288:(k+1)*288]=cv2.resize(array[t],(288,160))
                cv2.putText(canvas,f'{label} {t}',(k*288+6,j*184+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        path=root/'review'/name/'worst_flow_strip.png';Image.fromarray(canvas).save(path)
        review.append(dict(case=name,selection='posthoc maximum combined-minus-baseline flow-error transition',
                           transition=worst,frames=indices,path=str(path),sha256=sha256(path)))
        print(f'REVIEW {number+1}/{len(p["cases"])} ({100*(number+1)/len(p["cases"]):.0f}%) elapsed={time.monotonic()-start:.1f}s',flush=True)
    for suffix in ['png','svg']:figure.savefig(root/'review'/f'cadence_curves.{suffix}',dpi=140)
    plt.close(figure)
    write_json(root/'review'/'posthoc_sample_plan.json',dict(signature=p['signature'],samples=review,
               source_result_sha256=sha256(root/'RESULT.json'),analysis_code_sha256=sha256(__file__),
               note='These strips are selected for failure review after measurements, not an independent test set.'))
    print('REVIEW exports complete',flush=True)


if __name__=='__main__':main()
