"""Materialize the 2026-10-05 source re-review without modifying frozen v1.

Recipes are manually source-reviewed AI annotations, not generated model labels.
Only train/validation are written to this training sidecar; test is excluded.
"""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semantic_transmission.qwen35_training_data import DEFAULT_DATA, ROOT, read_json, write_json, sha256, verify

ALL = list(range(8))
LINKS = {
 '005_w0': {1:[0,1]}, '005_w2': {0:[0]},
 '007_w0': {0:[0]}, '007_w2': {1:[0,1]}, '007_w3': {1:[0]},
 '013_w0': {1:[0]}, '013_w1': {1:[0]}, '013_w3': {0:[0,1]},
 '014_w1': {0:[0]}, '014_w2': {0:[0]},
 '021_w2': {0:[0]}, '021_w3': {0:[0]},
 '025_w0': {1:[0]}, '025_w1': {1:[0]}, '025_w3': {0:[0]},
 '039_w1': {1:[0]}, '051_w0': {1:[0]}, '051_w3': {0:[0]},
 '001_w2': {1:[0]}, '001_w3': {1:[0,1]},
 '024_w0': {1:[0]}, '024_w1': {0:[0],1:[1]},
 '032_w0': {1:[0]}, '032_w1': {0:[0],1:[0]},
 '047_w1': {1:[0]}, '047_w3': {0:[0,1]},
}
# Continuous motion needs separated observations even without a discrete event.
CONTINUOUS = {'006':[1], '010':[0], '012':[1], '016':[0], '017':[0],
              '026':[0], '039':[0]}
EXTRAS = {
 '006': ('Dark parallel tracks run between the golden crop rows.', ALL),
 '007': ('The person behind the branches has long dark hair and a red flower near the side of their head.', ALL),
 '010': ('White spray collects below the falling water.', ALL),
 '012': ("The cat's striped tail lies across the grass.", ALL),
 '016': ('Red items are visible at the top of the dark bucket.', ALL),
 '021': ('Bright reflections run along the aisle floor.', ALL),
 '025': ("A white decorated pillar stands beside the person's dress.", ALL),
 '028': ('Red-roofed buildings and palm trees stand in the foreground.', ALL),
 '036': ('Ripples spread across the foreground water.', ALL),
}


def build():
    audit=verify(DEFAULT_DATA)
    records={}
    changes=[]
    for split in ('train','validation'):
        for original in read_json(DEFAULT_DATA/f'{split}.json')['records']:
            row=copy.deepcopy(original)
            vid=row['video_id'].split('_')[1]
            short=f'{vid}_w{row["window"]}'
            events=row['events']
            if short=='013_w3':
                changes.append({'id':row['id'],'field':'events[0].after', 'before':events[0]['after'], 'after':[6],
                    'reason':'Native f931 is still on the griddle; clear lift is f966 only. f1002 is lowered again.'})
                events[0]['after']=[6]
            if short=='014_w1':
                changes.append({'id':row['id'],'field':'events[0]', 'before':copy.deepcopy(events[0]),
                    'reason':'Helicopter body/landing gear remains at the upper edge in i5/i6. Complete absence first in i7.'})
                events[0].update(before=[4,5,6],after=[7])
            for e in events:
                e['start_frame']=row['frames'][max(e['before'])]['index']
                e['end_frame']=row['frames'][min(e['after'])]['index']
            facts=[]
            for j,old in enumerate(row['facts']):
                groups=[old['evidence']]
                kind='state'
                linked=LINKS.get(short,{}).get(j,[])
                if linked:
                    groups=[]
                    for k in linked:
                        groups.extend([events[k]['before'],events[k]['after']])
                    kind='transition'
                elif j in CONTINUOUS.get(vid,[]):
                    groups=[[0,1,2],[5,6,7]]
                    kind='continuous_motion'
                if short=='005_w1' and j==0 or short=='014_w3' and j==1 or vid=='021' and row['window']<2 and j==0 or short=='051_w2' and j==0:
                    groups=[[0,1,2],[5,6,7]];kind='continuous_motion'
                if short=='024_w3' and j==1:
                    groups=[[0,1,2],[4,5,6,7]];kind='transition'
                if short in ('032_w2','032_w3') and j==0 or short=='047_w0' and j==0 or short=='047_w2' and j==0:
                    groups=[[0,1,2],[5,6,7]];kind='continuous_motion'
                facts.append(dict(old,evidence_groups=groups,kind=kind,linked_events=[events[k]['id'] for k in linked]))
            # Add only details actually rechecked in the source images.
            extra=EXTRAS.get(vid) if split=='train' else None
            if extra:
                text,evidence=extra
                if short=='007_w0':evidence=[5,6,7]
                facts.append({'id':row['id']+'_detail0','text':text,'evidence':evidence,
                              'evidence_groups':[evidence],'kind':'state','linked_events':[]})
            if split=='train' and vid=='013' and row['window'] in (0,1,3):
                ev={0:[5,6,7],1:[0,1,2,3,4],3:ALL}[row['window']]
                facts.append({'id':row['id']+'_detail0','text':'Two round dark griddles sit beside ingredient trays.',
                              'evidence':ev,'evidence_groups':[ev],'kind':'state','linked_events':[]})
            if split=='train' and vid=='014' and row['window'] in (2,3):
                ev=[3,4,5,6,7] if row['window']==2 else ALL
                facts.append({'id':row['id']+'_detail0','text':'Several straps connect the green load to the cable.',
                              'evidence':ev,'evidence_groups':[ev],'kind':'state','linked_events':[]})
            if split=='train' and vid=='051':
                facts.append({'id':row['id']+'_detail0','text':'Two large round cacti in orange-brown pots occupy the foreground.',
                              'evidence':ALL,'evidence_groups':[ALL],'kind':'state','linked_events':[]})
            # Resolve pronouns for independently masked fact sentences.
            for f in facts:
                t=f['text']
                if vid=='007' and t.startswith('They '):t='The person in patterned clothing '+t[5:]
                if vid=='025' and t.startswith('They '):t='The person '+t[5:]
                if vid=='032' and t.startswith('They '):t='The two cable-car cabins '+t[5:]
                f['text']=t
            records[row['id']]={'split':split,'source_sha256':row['source_sha256'],
                'source_frame_hashes':[f['sha256'] for f in row['frames']],
                'review_sheet_sha256':row['review_sheet_sha256'], 'events':events,'facts':facts,
                'caption':' '.join(f['text'] for f in facts),
                'review':'AI source re-review of all eight sampled frames; native enlargement for ambiguous action boundaries; not independent human ground truth'}
    out=ROOT/'configs/qwen35_fact_annotations_v2.json'
    if out.exists():raise ValueError('Version exists; do not overwrite reviewed labels')
    write_json(out, {'version':2,'base_freeze_sha256':audit['freeze_sha256'],
        'provenance':'Codex source-only re-review; generated ablation captions not used as annotation targets',
        'scope':'64 training and 16 validation windows; original source splits unchanged; no test annotations in training sidecar',
        'limitations':['Sparse samples rather than exhaustive video truth','Compound fact units retained; not all atomic',
                       'Additional descriptions and temporal evidence groups do not increase the number of independent source videos'],
        'boundary_corrections':changes,'records':records})
    print(out, 'facts',sum(len(r['facts']) for r in records.values()),'temporal',sum(len(f['evidence_groups'])>1 for r in records.values() for f in r['facts']))


def refine_native_review():
    """Preserve v2; publish source-only corrections before any v3 evaluation."""
    audit=verify(DEFAULT_DATA)
    source=ROOT/'configs/qwen35_fact_annotations_v2.json'
    out=ROOT/'configs/qwen35_fact_annotations_v3.json'
    if out.exists():raise ValueError('Version exists; do not overwrite reviewed labels')
    meta=read_json(source)
    assert meta['base_freeze_sha256']==audit['freeze_sha256']
    changes=[]
    for ident,row in meta['records'].items():
        vid=ident.split('_')[1]
        short=f'{vid}_w{ident.rsplit("_w",1)[1]}'
        for fact in list(row['facts']):
            if not fact['id'].endswith('_detail0'):continue
            before=copy.deepcopy(fact)
            if short=='007_w0':
                row['facts'].remove(fact)
                reason='The red flower is not reliably visible in enlarged early frames; omit this uncertain target.'
            elif short=='007_w1':
                fact.update(evidence=list(range(2,8)),evidence_groups=[list(range(2,8))])
                reason='Red flower is clear from native f135 (i2); the earlier background view is blurred.'
            elif vid=='012':
                fact['text']="The cat's body has dark stripes."
                reason='Striped body is clear; the lower appendage cannot reliably be distinguished as tail rather than hind leg.'
            else:continue
            changes.append({'id':ident,'before':before,'after':fact if fact in row['facts'] else None,'reason':reason})
        row['caption']=' '.join(f['text'] for f in row['facts'])
    meta.update(version=3,parent_annotation_sha256=sha256(source),native_detail_corrections=changes)
    write_json(out,meta)
    print(out,'facts',sum(len(r['facts']) for r in meta['records'].values()),'native corrections',len(changes))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision',type=int,choices=[2,3],default=3)
    args=parser.parse_args()
    build() if args.revision==2 else refine_native_review()
