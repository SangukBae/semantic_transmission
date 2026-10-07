"""Supplementary paired contrasts; never changes the predeclared adoption gate."""
from collections import Counter,defaultdict
from semantic_transmission.qwen35_astra_transfer import RUN
from semantic_transmission.qwen35_training_data import read_json,write_json,sha256


def main():
    result=read_json(RUN/'RESULT.json')
    if result['status']!='COMPLETE_CROSS_ANNOTATOR_ASTRA_PILOT':raise ValueError('Review not complete')
    totals={s:defaultdict(Counter) for s in ['validation','test','combined']}
    per_source=defaultdict(lambda:defaultdict(Counter))
    examples=[]
    for row in result['rows']:
        eligible=set(row['four_frame_fact_ids']);by_id={f['id']:f for f in row['facts']}
        actions={f for f in eligible if by_id[f]['kind']!='state'}
        video=row['id'].rsplit('_w',1)[0];details={}
        for name,cell in row['cells'].items():
            review=cell['review'];ok=cell['output']['status']=='COMPLETE'
            hit={f for f,v in review['fact_hits'].items() if v} if ok else set()
            partial=set(review.get('partial_fact_ids',[])) if ok else set()
            counts=Counter(fact_hits=len(hit&eligible),action_hits=len(hit&actions),
                full_or_partial_facts=len((hit|partial)&eligible),full_or_partial_actions=len((hit|partial)&actions),
                error_windows=int(bool(review['errors'])),extra_eight_frame_hits=len(hit-eligible),
                facts=len(eligible),actions=len(actions),extra_eight_frame_facts=len(by_id)-len(eligible),windows=1)
            for fact in eligible:
                kind=by_id[fact]['kind'];counts[kind+'_facts']+=1;counts[kind+'_hits']+=int(fact in hit)
            for scope in [row['split'],'combined']:totals[scope][name].update(counts)
            per_source[video][name].update(counts)
            details[name]=dict(hits=sorted(hit&eligible),missed=sorted(eligible-hit),partial=sorted(partial&eligible))
        examples.append(dict(id=row['id'],split=row['split'],facts=by_id,conditions=details))
    contrasts={}
    comparisons=[('teacher_r8','control_r8'),('teacher_r32','control_r32'),
        ('teacher_r32','teacher_r8'),('control_r32','control_r8'),
        ('teacher_r8','base'),('teacher_r32','base'),('base_8frames','base')]
    for a,b in comparisons:
        label=a+'_minus_'+b
        contrasts[label]={s:{k:arms[a][k]-arms[b][k] for k in ['fact_hits','action_hits','full_or_partial_facts','full_or_partial_actions','error_windows']}
            for s,arms in totals.items()}
        contrasts[label]['source_differences']={video:{k:arms[a][k]-arms[b][k] for k in ['fact_hits','action_hits','error_windows']}
            for video,arms in per_source.items()}
    out=dict(status='SUPPLEMENTARY_PAIRED_ANALYSIS',result_sha256=sha256(RUN/'RESULT.json'),
        totals=totals,contrasts=contrasts,examples=examples,
        scope='Complete+partial and per-source contrasts supplement the predeclared complete-fact gate. No changed judgments, checkpoint selection or statistical significance claim.')
    write_json(RUN/'paired_analysis.json',out)
    print({s:{a:dict(c) for a,c in arms.items()} for s,arms in totals.items() if s=='combined'})


if __name__=='__main__':main()
