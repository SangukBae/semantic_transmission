"""Local, resumable selection-only comparison against a blinded Astra reference.

Four predeclared complete Kinetics clips. Historical baseline inference is reused
only after checking matching source hashes; it is never claimed as a local rerun.
"""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import signal
import time

from . import auto_extraction as ae, qwen35_event_selection as event
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO/'outputs/bsu_event_selection_20261019_v2'
CODE = ['src/semantic_transmission/'+name+'.py' for name in
        ('qwen35_event_selection','bsu_event_validation','qwen35_semantic_selection',
         'qwen35_feedback','qwen35_backend','bsu_kinetics_base')]


def prepare(root):
    packet=ae.read(root/'reference/packet.json')
    imported=ae.read(root/'baseline_lgvsc_import.json')
    rows=[]
    for row in packet['records']:
        source=ae.read(row['prepared']); vid=row['id']
        if sha256(row['prepared']) != row['prepared_sha256']:
            raise ValueError('reference source metadata changed')
        if source['source_frame_hashes'] != row['source_frame_hashes']:
            raise ValueError('reference frame identity changed')
        for i in range(source['frames']):
            ae.check_frame(source,i)
        if sha256(source['normalized_video']) != source['normalized_sha256']:
            raise ValueError('normalized source changed')
        base=REPO/'outputs/bsu_kinetics_context/benchmark/snr_10'/vid/'base'
        before=ae.unseal(base/'selection.json'); prior=ae.read(base/'plan.json')
        if (before['signature'] != prior['signature'] or prior['source'] != source
                or prior['selector'] != 'qwen35_sequential_semantic_v1'
                or prior['candidate_stride'] != 4):
            raise ValueError('BSU baseline/source/policy mismatch')
        for call in before['calls']:
            if sha256(call['path']) != call['sha256']:
                raise ValueError('baseline raw call changed')
        other=imported[vid]; audit=other['files']['input_audit.json']['data']
        if (other['video']['prepared_sha256'] != row['prepared_sha256']
                or audit['source_sha256'] != source['normalized_sha256']
                or not audit['all_png_pixels_checked'] or audit['frames'] != source['frames']):
            raise ValueError('LGVSC source does not match the reference/BSU source')
        lgkeys=other['files']['keyframes.json']['data']['indices']
        event.baseline.validate_keys(lgkeys,source['frames'])
        event.baseline.validate_keys(before['keys'],source['frames'])
        rows.append(dict(id=vid,source=source,prepared_sha256=row['prepared_sha256'],
            before_selection=str(base/'selection.json'),before_sha256=sha256(base/'selection.json'),
            baseline_keys=dict(lgvsc=lgkeys,bsu_before=before['keys']),
            qwen={k:v for k,v in prior['caption'].items() if k not in ('caption_adapter','prompts')},
            baseline_origin='Previously executed selectors; identical prepared/source frame hashes verified'))
    protocol=dict(policy='bsu_event_selection_comparison_v1',rows=rows,
        detector=asdict(event.DetectorConfig()),reference_model='gpt-6-astra',
        reference_packet_sha256=sha256(root/'reference/packet.json'),
        reference_annotation_sha256=sha256(root/'reference/astra_reference.json'),
        baseline_import_sha256=sha256(root/'baseline_lgvsc_import.json'),
        code={p:sha256(REPO/p) for p in CODE},
        cohort='First four lexicographic existing Kinetics clips; 4 x 240 frames, full timelines.',
        tuning='Engineering thresholds fixed before reading Astra labels or scoring outputs; synthetic tests only.',
        primary='One-to-one matching of internal keys to Astra acceptable state ranges; micro precision/recall/F1.',
        secondary='Exact and +/-1,2,4-frame one-to-one matching to representative IDs; macro F1.',
        endpoint_policy='Both endpoints retained by all models, excluded from all agreement scores.',
        budget='Natural key counts; precision/F1 penalize extra unmatched keys. No equal-rate superiority claim.',
        limitations=['Four-clip pilot, not 14-clip full benchmark','Astra reference is not human ground truth',
                    'Selection only; no new channel/reconstruction quality measurements',
                    'Astra sees contact sheets and chosen native closeups; Qwen sees native frames plus crops',
                    'Baseline timings from different host cannot measure speed improvement'])
    protocol['signature']=ae.digest(protocol)
    freeze(root/'protocol.json',protocol)
    return protocol


def verify(root):
    p=ae.read(root/'protocol.json')
    if ae.digest({k:v for k,v in p.items() if k!='signature'}) != p['signature']:
        raise ValueError('protocol changed')
    for rel,h in p['code'].items():
        if sha256(REPO/rel) != h:
            raise ValueError(f'frozen experiment code changed: {rel}')
    if (sha256(root/'reference/packet.json') != p['reference_packet_sha256']
            or sha256(root/'reference/astra_reference.json') != p['reference_annotation_sha256']
            or sha256(root/'baseline_lgvsc_import.json') != p['baseline_import_sha256']):
        raise ValueError('reference packet or baseline provenance changed')
    for row in p['rows']:
        if sha256(row['before_selection']) != row['before_sha256']:
            raise ValueError('baseline selection changed')
        for i in range(row['source']['frames']):
            ae.check_frame(row['source'],i)
    return p


def run(root):
    from .bsu_kinetics_base import current_pipeline
    from .qwen35_pairs_comparison import lock
    from .validation_progress import emit
    root.mkdir(parents=True,exist_ok=True)
    with lock(root/'execution.lock'), lock(REPO/'.local/etri_60s_check.lock'):
        p=prepare(root) if not (root/'protocol.json').exists() else verify(root)
        total=sum(r['source']['frames']-1 for r in p['rows']); done=0; model=None
        started=time.monotonic()
        try:
            for row in p['rows']:
                dest=root/'videos'/row['id'];dest.mkdir(parents=True,exist_ok=True)
                source=row['source'];n=source['frames'];path=dest/'selection.json'
                if path.exists():
                    saved=ae.unseal(path)
                    if saved['signature']!=p['signature']:
                        raise ValueError('selection cache mismatch')
                    for call in saved['calls']:
                        if sha256(call['path'])!=call['sha256']:
                            raise ValueError('cached raw model response changed')
                    done+=n-1;continue
                scan_path=dest/'scan.json'
                if scan_path.exists():
                    scan=ae.unseal(scan_path)
                    if scan['signature']!=p['signature']:
                        raise ValueError('scan cache mismatch')
                else:
                    write_json(root/'status.json',dict(status='RUNNING',stage='all_frame_scan',video=row['id'],done=done,total=total))
                    t=time.monotonic()
                    scan=event.detect([Path(source['source_frames'])/f'{i}.png' for i in range(n)],event.DetectorConfig(**p['detector']))
                    scan=dict(scan,signature=p['signature'],seconds=time.monotonic()-t)
                    freeze(scan_path,ae.seal(scan))
                plan=dict(source,**row['qwen'],signature=p['signature'],max_images=16)
                session=event.make_session(dest,plan,model)
                def progress(current,_,start):
                    elapsed=time.monotonic()-started
                    write_json(root/'status.json',dict(status='RUNNING',stage='qwen_event_selection',video=row['id'],
                        done=done+current,total=total,percent=100*(done+current)/total,elapsed_seconds=elapsed,
                        eta_seconds=None,resume='Same run command; source/code/cached model responses are verified'))
                    emit(done+current,total)
                    print(f'SELECTION {done+current}/{total} ({100*(done+current)/total:.1f}%) {row["id"]}',flush=True)
                t=time.monotonic()
                with current_pipeline():
                    keys,trace=event.select(n,session,scan,progress=progress)
                if session.model is not None:
                    model=session.model.base
                freeze(path,ae.seal(dict(signature=p['signature'],id=row['id'],keys=keys,trace=trace,
                    calls=[{k:v for k,v in c.items() if k!='reused'} for c in session.calls],
                    selection_seconds=time.monotonic()-t,scan_seconds=scan['seconds'],
                    scan_sha256=sha256(scan_path),runtime=session.model.runtime if session.model else None)))
                done+=n-1
            write_json(root/'status.json',dict(status='SELECTION_COMPLETE_REFERENCE_EVALUATION_PENDING',done=total,total=total,
                                             elapsed_seconds=time.monotonic()-started))
        except BaseException as exc:
            write_json(root/'status.json',dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                error=str(exc),resume='Repeat the same command; completed scans and valid raw responses retained.'))
            raise


def match(keys,events,tolerance=None):
    """Maximum-cardinality bipartite match; every predicted/reference key used once."""
    owner={}
    def edges(index):
        k=keys[index]
        return [i for i,e in enumerate(events) if
                (e['acceptable_start']<=k<=e['acceptable_end'] if tolerance is None
                 else abs(k-e['frame'])<=tolerance)]
    def augment(k,seen):
        for j in sorted(edges(k),key=lambda j:abs(keys[k]-events[j]['frame'])):
            if j in seen:continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):
                owner[j]=k;return True
        return False
    for k in range(len(keys)):augment(k,set())
    tp=len(owner);pred=len(keys);ref=len(events)
    precision=tp/pred if pred else (1. if not ref else 0.)
    recall=tp/ref if ref else 1.
    return dict(tp=tp,predicted=pred,reference=ref,precision=precision,recall=recall,
        f1=2*tp/(pred+ref) if pred+ref else 1.,
        pairs=[dict(prediction=keys[k],reference=events[j]['frame']) for j,k in sorted(owner.items())])


def reference(root,p):
    path=root/'reference/astra_reference.json';a=ae.read(path)
    packet=ae.read(root/'reference/packet.json')
    if (a['annotator_model']!='gpt-6-astra' or not a['source_only']
            or a['packet_sha256']!=p['reference_packet_sha256']):
        raise ValueError('reference model/provenance mismatch')
    sheets={s['path']:s['sha256'] for r in packet['records'] for s in r['sheets']}
    if not set(sheets)<=set(a['viewed_sheet_paths']):
        raise ValueError('reference did not inspect all supplied sheets')
    for path,h in sheets.items():
        if sha256(path)!=h:raise ValueError('reference image changed')
    rows={r['id']:r for r in a['records']}
    if len(rows)!=len(a['records']) or set(rows)!={r['id'] for r in p['rows']}:
        raise ValueError('incomplete reference')
    for row in p['rows']:
        r=rows[row['id']];n=row['source']['frames'];event.baseline.validate_keys(r['keys'],n)
        if r['frames']!=n or [e['frame'] for e in r['events']]!=r['keys'][1:-1]:
            raise ValueError('reference event/keys mismatch')
        for e in r['events']:
            if not 0<e['acceptable_start']<=e['frame']<=e['acceptable_end']<n-1:
                raise ValueError('reference event range invalid')
    freeze(root/'reference/freeze.json',dict(model='gpt-6-astra',sha256=sha256(root/'reference/astra_reference.json'),
        packet_sha256=p['reference_packet_sha256'],scope='Independent source-only annotation; not human ground truth'))
    return rows


def report(root):
    p=verify(root);refs=reference(root,p);rows=[]
    for row in p['rows']:
        after=ae.unseal(root/'videos'/row['id']/'selection.json')
        if after['signature']!=p['signature']:raise ValueError('selection protocol mismatch')
        for call in after['calls']:
            if sha256(call['path'])!=call['sha256']:raise ValueError('raw selection evidence changed')
        if sha256(root/'videos'/row['id']/'scan.json')!=after['scan_sha256']:
            raise ValueError('detector evidence changed')
        arms=dict(row['baseline_keys'],bsu_after=after['keys']);ref=refs[row['id']]
        for name,keys in arms.items():
            event.baseline.validate_keys(keys,row['source']['frames'])
            rows.append(dict(id=row['id'],model=name,keys=keys,key_count=len(keys),
                reference_key_count=len(ref['keys']),event_match=match(keys[1:-1],ref['events']),
                temporal={str(t):match(keys[1:-1],ref['events'],t) for t in (0,1,2,4)}))
    summary={}
    for name in ('lgvsc','bsu_before','bsu_after'):
        group=[r for r in rows if r['model']==name]
        def aggregate(scores):
            tp=sum(r['tp'] for r in scores);pred=sum(r['predicted'] for r in scores);ref=sum(r['reference'] for r in scores)
            return dict(tp=tp,predicted=pred,reference=ref,precision=tp/pred if pred else 0.,recall=tp/ref if ref else 1.,
                micro_f1=2*tp/(pred+ref) if pred+ref else 1.,macro_f1=sum(r['f1'] for r in scores)/len(scores))
        summary[name]=dict(keys=sum(r['key_count'] for r in group),event_match=aggregate([r['event_match'] for r in group]),
                          temporal={str(t):aggregate([r['temporal'][str(t)] for r in group]) for t in (0,1,2,4)})
    result=dict(status='COMPLETE_SELECTION_COMPARISON',signature=p['signature'],reference_sha256=sha256(root/'reference/astra_reference.json'),
        videos=len(p['rows']),frames=sum(r['source']['frames'] for r in p['rows']),reference_keys=sum(len(r['keys']) for r in refs.values()),
        summary=summary,rows=rows,limitations=p['limitations'],new_reconstruction_evaluation=False)
    write_json(root/'RESULT.json',result)
    page=['<!doctype html><meta charset="utf-8"><title>BSU keyframe comparison</title>',
          '<style>body{font-family:sans-serif;margin:24px;background:#111;color:#ddd}img{width:192px}figure{display:inline-block;margin:5px}td,th{padding:8px;border:1px solid #888}</style>',
          '<h1>Independent Astra keyframe agreement — four complete clips</h1>',
          '<p>Source-only reference. Endpoints excluded from scores. Natural key counts, not matched bitrate. Selection only.</p>',
          '<table><tr><th>Model</th><th>Keys</th><th>Precision</th><th>Recall</th><th>F1</th></tr>']
    for name,s in summary.items():
        m=s['event_match'];page.append(f'<tr><td>{name}</td><td>{s["keys"]}</td><td>{m["precision"]:.3f}</td><td>{m["recall"]:.3f}</td><td>{m["micro_f1"]:.3f}</td></tr>')
    page.append('</table>')
    for row in p['rows']:
        page.append(f'<h2>{row["id"]}</h2>')
        arms={'GPT6-Astra':refs[row['id']]['keys'],**{r['model']:r['keys'] for r in rows if r['id']==row['id']}}
        for name,keys in arms.items():
            page.append(f'<h3>{name} — {len(keys)} keys</h3>')
            for key in keys:
                path=Path(row['source']['source_frames'])/f'{key}.png'
                import base64
                from PIL import Image
                from io import BytesIO
                with Image.open(path) as im:
                    buffer=BytesIO();im.resize((288,160)).save(buffer,format='JPEG',quality=85)
                encoded=base64.b64encode(buffer.getvalue()).decode()
                page.append(f'<figure><img src="data:image/jpeg;base64,{encoded}"><figcaption>frame {key} / {key/24:.3f}s</figcaption></figure>')
    page.append('<p>'+html.escape(' | '.join(p['limitations']))+'</p>')
    (root/'comparison.html').write_text('\n'.join(page))
    write_json(root/'status.json',dict(status=result['status'],done=4,total=4))
    print(json.dumps(dict(status=result['status'],reference_keys=result['reference_keys'],summary=summary),indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','report'))
    parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    {'prepare':prepare,'run':run,'report':report}[args.action](args.root.resolve())


if __name__=='__main__':main()
