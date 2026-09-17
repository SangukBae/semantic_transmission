"""Annotation-only auditing of RGB outputs; never called by a candidate scorer."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .metric_v3_cases import annotation_events


def track_truth_map(observation, truth):
    tracks=observation['tracks']; labels=truth['masks']; identities=range(1,truth['visible'].shape[1]+1)
    affinities=np.zeros((len(tracks),len(identities)))
    for i,track in enumerate(tracks):
        for j,identity in enumerate(identities):
            values=[]
            for t,r in track.items():
                gt=labels[t]==identity; union=(gt|r['mask']).sum()
                if gt.any():values.append((gt&r['mask']).sum()/max(1,union))
            affinities[i,j]=np.mean(values) if values else 0.
    # Different nonoverlapping inferred fragments may belong to the same true entity.
    mapping={}
    for i in range(len(tracks)):
        j=int(np.argmax(affinities[i]))
        if affinities[i,j]>=.25:mapping[i]=j+1
    return mapping


def audit_events(observation, metric, truth):
    mapping=track_truth_map(observation,truth)
    true_events=annotation_events(truth['centers'],truth['visible'])
    predicted=metric['events'];cost=np.full((len(true_events),len(predicted)),1e6)
    for i,a in enumerate(true_events):
        for j,b in enumerate(predicted):
            dt=abs(a['time_s']-b['source_time_s'])
            if a['object_id']==mapping.get(b['source_track_id']) and a['type']==b['type'] and dt<=.25+1e-9:
                cost[i,j]=dt
    pairs={int(i):int(j) for i,j in zip(*linear_sum_assignment(cost)) if cost[i,j]<1e6}
    rows=[]
    for i,event in enumerate(true_events):
        rows.append({**event,'inferred_event_index':pairs.get(i),
                     'observed':predicted[pairs[i]] if i in pairs else None})
    return {'truth_events':len(true_events),'matched_source_events':len(pairs),'predicted_source_events':len(predicted),
            'source_events':rows,'scope':'External annotation audit; missing source queries remain in coverage denominator'}


def source_paired_additions(a,b,metric,truth):
    """Known source-entity vocabulary only; not open-world object hallucination.

GT absent AND source detector negative conditions the paired denominator. Truth
only associates the source query with annotated entity ID in this audit.
"""
    mapping=track_truth_map(a,truth);groups=metric['entity_groups'];v=truth['visible']
    counts={'absent_opportunities':0,'unpairable':0,'raw_additions':0,'raw_pairable_opportunities':0,
            'source_positive_excluded':0,'paired_opportunities':0,'paired_additions':0,
            'present_opportunities':int(v.sum()),'source_detected_present':0}
    for k in range(v.shape[1]):
        ids=[i for i,identity in mapping.items() if identity==k+1]
        at={t for i in ids for t in a['tracks'][i]}
        bt={t for i in ids for j in groups.get(i,[]) for t in b['tracks'][j]}
        counts['source_detected_present']+=sum(t in at for t in range(len(v)) if v[t,k])
        for t in range(len(v)):
            if v[t,k]:continue
            counts['absent_opportunities']+=1
            if not ids:counts['unpairable']+=1;continue
            counts['raw_pairable_opportunities']+=1;counts['raw_additions']+=int(t in bt)
            if t in at:counts['source_positive_excluded']+=1;continue
            counts['paired_opportunities']+=1;counts['paired_additions']+=int(t in bt)
    return counts | {'raw_h_add':counts['raw_additions']/counts['raw_pairable_opportunities'] if counts['raw_pairable_opportunities'] else None,
                     'source_paired_h_add':counts['paired_additions']/counts['paired_opportunities'] if counts['paired_opportunities'] else None,
                     'scope':'known source entities, source-GT-absent opportunities; same RGB extractor as candidates, not an independent validator'}
