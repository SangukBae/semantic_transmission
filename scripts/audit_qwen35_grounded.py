"""Source, pixel and supervision coverage audit; no model or optimizer."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from semantic_transmission.qwen35_grounded_data import RUN,PARENT,verify_data
from semantic_transmission.qwen35_training_data import ROOT,read_json,write_json,sha256


def dhash(path):
    with Image.open(path) as im:a=np.asarray(im.convert('L').resize((9,8)))
    bits=(a[:,1:]>a[:,:-1]).flatten();return sum(int(v)<<i for i,v in enumerate(bits))


def audit():
    verify_data();rows={s:read_json(RUN/f'{s}.json')['records'] for s in ['train','validation','test']}
    protected=[]
    for s in ['validation','test','regression']:
        protected+=read_json(ROOT/'outputs/qwen35_caption_repair_v1'/f'{s}.json')['records']
    for s in ['validation','test']:protected+=read_json(PARENT/f'{s}_inputs.json')['records']
    # Old train rows reappear intentionally as replay, never as evaluation.
    train_ids={r['source_id'] for r in rows['train']};assert not train_ids&{r['source_id'] for r in protected}
    all_groups=dict(rows,protected_development=protected);by_hash={};near=[];exact=[]
    for split,rr in all_groups.items():
        frames={f['path']:f for r in rr for f in r['frames']}
        by_hash[split]=[(p,f['sha256'],dhash(p)) for p,f in frames.items()]
    for left,right in combinations(by_hash,2):
        for pa,ha,da in by_hash[left]:
            for pb,hb,db in by_hash[right]:
                if ha==hb:exact.append(dict(splits=[left,right],paths=[pa,pb]))
                if (da^db).bit_count()<=4:near.append(dict(splits=[left,right],paths=[pa,pb],distance=(da^db).bit_count()))
    if exact:raise ValueError('Cross-split exact frame duplicates')
    coverage={}
    for split,rr in rows.items():
        new=[r for r in rr if r['origin']=='grounded_expansion']
        coverage[split]=dict(originals=len({r['source_id'] for r in rr}),windows=len(rr),new_windows=len(new),
            facts=sum(len(r['facts']) for r in rr),fact_kinds=dict(Counter(f['kind'] for r in rr for f in r['facts'])),
            events=sum(len(r['events']) for r in rr),new_camera_motion=dict(Counter(r['motion_review']['camera'] for r in new)),
            new_object_motion=dict(Counter(r['motion_review']['objects'] for r in new)),
            new_entity_visibility=dict(Counter(e['visibility'] for r in new for e in r['entities'])),
            new_uncertainty_windows=sum(bool(r['uncertainties']) for r in new),
            confirmed_negative_indices=sum(len(r.get('confirmed_no_transition_indices',[])) for r in rr),
            caption_words_mean=sum(len(r['caption'].split()) for r in rr)/len(rr))
    panels=0
    for a in ['A','B']:
        for row in read_json(RUN/'teacher'/f'packet_{a}.json')['records']:
            images=[Image.open(f['path']).convert('RGB') for f in row['frames']]
            width=max(im.width for im in images);height=max(im.height for im in images)
            expected=Image.new('RGB',(2*width,2*(height+26)),'white');draw=ImageDraw.Draw(expected)
            for n,(f,im) in enumerate(zip(row['frames'],images)):
                x=(n%2)*width;y=(n//2)*(height+26)
                draw.text((x+4,y+4),f"i{f['i']} | frame {f['index']} | {f['index']/row['fps']:.3f}s",fill='black');expected.paste(im,(x,y+26));panels+=1
            with Image.open(row['sheet']) as actual:
                assert actual.size==expected.size and actual.convert('RGB').tobytes()==expected.tobytes()
    reviewed=False
    adjudication=RUN/'duplicate_adjudication.json'
    if near and adjudication.exists():
        judgment=read_json(adjudication);by_paths={tuple(r['paths']):r for r in judgment['records']}
        if set(by_paths)!={tuple(r['paths']) for r in near}:raise ValueError('Incomplete duplicate review')
        for paths,r in by_paths.items():
            if r['sha256']!=[sha256(p) for p in paths] or r['judgment']!='distinct' or not r['reason']:raise ValueError('Duplicate review not resolved')
        reviewed=True
    result=dict(status='PASS' if not near or reviewed else 'NEAR_DUPLICATES_REQUIRE_REVIEW',coverage=coverage,
        source_disjoint=True,exact_cross_split_duplicates=exact,dhash_threshold=4,near_duplicates=near,
        near_duplicate_review_sha256=sha256(adjudication) if reviewed else None,near_duplicates_resolved_as_distinct=reviewed,
        native_pixels_and_labels_verified=panels,data_freeze_sha256=sha256(RUN/'data_freeze.json'),
        limitations=['AI source-only labels, not human ground truth. No complete event boundaries or track IDs.',
            'Exact source/PNG and sampled dHash screening does not exclude every re-edit or semantic duplicate.',
            'Random source split, deterministic four windows per video; not a balanced hard-case benchmark.',
            'Explicit transition negatives remain separate from stable camera/object descriptions.'])
    write_json(RUN/'data_audit.json',result)
    write_json(RUN/'source_usage.json',dict(parent_usage_sha256=sha256(PARENT/'source_usage.json'),
        newly_allocated=read_json(RUN/'protocol.json')['sources'],remaining_reserve_sources=[],
        training_originals=sorted(train_ids),held_out_originals={s:sorted({r['source_id'] for r in rows[s]}) for s in ['validation','test']},
        old_evaluation_remains_protected=True))
    print(result)


if __name__=='__main__':audit()
