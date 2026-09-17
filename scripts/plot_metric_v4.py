#!/usr/bin/env python3
"""Descriptive figures from fixed scores; no fitting or metric changes."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):return json.loads(path.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);root=ap.parse_args().output
    result=read(root/'summary.json')
    names=list(result['event']['ghost']['metrics'])
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,key in zip(axes,('auc','rate')):
        matrix=[]
        for name in names:
            values=[]
            for target in ('ghost','delay'):
                v=result['event'][target]['metrics'][name]
                value=v['auc'] if key=='auc' else v['positive']['rate']
                values.append(np.nan if value is None else value)
            matrix.append(values)
        matrix=np.array(matrix)
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for y in range(len(names)):
            for x in range(2):
                value=matrix[y,x];ax.text(x,y,'null' if np.isnan(value) else f'{value:.3f}',ha='center',va='center',color='white' if value>.65 else 'black')
        ax.set_yticks(range(len(names)),names);ax.set_xticks([0,1],['Post-exit ghost','Event delay/miss'])
        ax.set_title('ROC AUC (measurable cases)' if key=='auc' else 'Detection rate (abstentions are misses)')
    fig.subplots_adjust(left=.18,right=.86,wspace=.15)
    scale_ax=fig.add_axes([.91,.23,.015,.55]);fig.colorbar(im,cax=scale_ax)
    fig.suptitle('New scene families; thresholds frozen on previous data',y=.98)
    for suffix in ('png','svg'):fig.savefig(root/('event_comparison.'+suffix),dpi=160)
    plt.close(fig)
    methods=('sta','sta_without_rejection','stage','stage_without_rejection')
    labels=('STA + reject','STA raw 5-class','Stage distances + reject','Stage distances raw')
    fields=('macro_accuracy','coverage','normal_false_alarm_rate','normal_abstention_rate')
    fig,axes=plt.subplots(1,4,figsize=(17,5),sharey=True)
    for ax,key,title in zip(axes,fields,('5-class macro accuracy','Coverage','Normal false alarms','Normal abstentions')):
        vals=[result['STA'][method][key] for method in methods]
        bars=ax.bar(range(4),vals,color=['#277da1','#90be6d','#f8961e','#f9c74f'])
        for i,(bar,value,method) in enumerate(zip(bars,vals,methods)):
            ci=result['STA'][method][key+'_source_95ci']
            ax.plot([i,i],ci,color='black',linewidth=1.5)
            ax.text(i,min(1.16,ci[1]+.025),f'{value:.1%}',ha='center',fontsize=9)
        ax.set_xticks(range(4),labels,rotation=30,ha='right',fontsize=8);ax.set_ylim(0,1.2);ax.set_title(title)
    axes[0].axhline(.8,color='gray',linestyle='--');axes[1].axhline(.95,color='gray',linestyle='--');axes[2].axhline(.1,color='gray',linestyle='--')
    fig.suptitle('STA normal and abstention evaluation; known fault classes plus normal')
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(root/('sta_normal_rejection.'+suffix),dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    ghost=result['event']['ghost']['numerical_agreement']['rows']
    delay=result['event']['delay']['numerical_agreement']['rows']
    for ax,group,key,title in ((axes[0],ghost,'ghost_auc','Ghost occupancy in 0.5 s'),(axes[1],delay,'delay_penalty','Delay/1 s; deadline miss = 1')):
        pairs=[(r['oracle'][key],r['observed'][key]) for r in group if r['measured']]
        if pairs:
            x,y=np.array(pairs).T
            ax.scatter(x,y,alpha=.25,s=65)
        ax.plot([0,1],[0,1],'k--',linewidth=1);ax.set_xlim(-.04,1.04);ax.set_ylim(-.04,1.04)
        ax.set_xlabel('Independent simulation truth');ax.set_ylabel('RGB-derived candidate');ax.set_title(f'{title}\nmeasured {len(pairs)}/{len(group)}')
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(root/('numeric_agreement.'+suffix),dpi=160)
    plt.close(fig)
    sources=read(root/'sources.json');fig,axes=plt.subplots(3,4,figsize=(12,6))
    for row,index in enumerate((0,8,16)):
        source=sources[index]
        with np.load(root/'sources'/source['source_stem']/'source.npz') as f:frames=f['frames']
        for column,t in enumerate((0,9,20,27)):
            axes[row,column].imshow(frames[t]);axes[row,column].set_title(f"{source['family']}\nt={t/8:.3f}s",fontsize=9);axes[row,column].axis('off')
    fig.tight_layout();fig.savefig(root/'new_scene_examples.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
