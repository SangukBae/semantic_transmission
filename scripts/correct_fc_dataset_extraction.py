#!/usr/bin/env python3
"""Correct assistant proposals with genuine sequential FC-LGVSC SKEM scoring.

Original artifacts stay immutable. Classification decisions are authored by the
assistant from prior source observations, with additional visual review as needed.
Only exactly matching source intervals/samples may reuse a prior assistant caption.
Changed intervals require new direct review and assistant-authored captions.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json
from semantic_transmission import hybrid_selection as hybrid
from semantic_transmission.assisted_captions import expected_samples, validate_bundle

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / 'outputs/fc_lgvsc_webvid_tvsum_full_20261001'
ROOT = REPO / 'outputs/fc_lgvsc_webvid_tvsum_faithful_20261002'
SELF = Path(__file__).resolve()


def now():
    return datetime.now(timezone.utc).isoformat()


def locate(video, root=ROOT):
    matches = list(root.glob(f'*/{video}'))
    if len(matches) != 1:
        raise ValueError(f'unknown or ambiguous video {video}')
    return matches[0]


def init():
    ROOT.mkdir(exist_ok=True)
    inventory = read_json(OLD / 'inventory.json')
    target = ROOT / 'inventory.json'
    if target.exists() and read_json(target) != inventory:
        raise ValueError('preserve another inventory')
    write_json(target, inventory)
    for source in inventory['videos']:
        directory = ROOT / source['dataset'].lower() / source['id']
        directory.mkdir(parents=True, exist_ok=True)
        prepared = OLD / source['dataset'].lower() / source['id'] / 'source_prepared.json'
        target = directory / 'source_prepared.json'
        if target.exists() and sha256(target) != sha256(prepared):
            raise ValueError('source preparation changed')
        if not target.exists():
            target.write_bytes(prepared.read_bytes())


def classify(video, decisions_path):
    directory = locate(video)
    old = locate(video, OLD)
    decisions = read_json(decisions_path)
    previous = read_json(old / 'extraction/protocol.json')
    frames = [c['frame'] for c in previous['candidates']]
    optional = decisions['optional_frames']
    protected = decisions['protected_frames']
    if (len(optional) != len(set(optional)) or len(protected) != len(set(protected))
            or set(optional) & set(protected) or set(optional) | set(protected) != set(frames)):
        raise ValueError('every prior candidate needs exactly one explicit classification')
    if 0 not in protected or previous['frames'] - 1 not in protected:
        raise ValueError('first and final frame are mandatory')
    if not decisions['rationale'] or not decisions['author']:
        raise ValueError('classification needs actual assistant rationale and provenance')
    # Decisions are supplied explicitly; this script does not infer classifications
    # from keywords or mark any image as reviewed.
    candidates = [dict(c, mandatory=c['frame'] in protected) for c in previous['candidates']]
    note = dict(decisions, old_protocol_sha256=sha256(old / 'extraction/protocol.json'),
                old_proposal_sha256=previous['proposal_sha256'],
                candidates=candidates, source_id=video, created_utc=now())
    extraction = directory / 'extraction'
    extraction.mkdir(exist_ok=True)
    if (extraction / 'protocol.json').exists():
        saved = read_json(directory / 'reclassification.json')
        if any(saved[k] != note[k] for k in note if k != 'created_utc'):
            raise ValueError('classification already frozen; use a new revision')
        return
    write_json(directory / 'reclassification.json', note)
    old_observations = read_json(previous['proposal_path'])
    observations = dict(old_observations, candidates=candidates,
                        correction='Explicit protected/optional reclassification; original source review retained.',
                        reclassification_sha256=sha256(directory / 'reclassification.json'))
    proposal_path = extraction / 'assistant_observations.json'
    write_json(proposal_path, observations)
    protocol = copy.deepcopy(previous)
    protocol.pop('signature')
    protocol.update(created_utc=now(), baseline=str(directory), candidates=candidates,
                    proposal_path=str(proposal_path), proposal_sha256=sha256(proposal_path),
                    scope='Original FC-LGVSC procedure: protected events, genuine sequential SKEM for optional proposals, max-gap anchors, assistant captions.',
                    correction_from=str(old / 'extraction'),
                    reclassification_path=str(directory / 'reclassification.json'),
                    reclassification_sha256=sha256(directory / 'reclassification.json'))
    protocol['code'][str(SELF.relative_to(REPO))] = sha256(SELF)
    protocol['signature'] = fingerprint(protocol)
    write_json(extraction / 'protocol.json', protocol)
    print(json.dumps(dict(video=video, protected=len(protected), optional=len(optional))), flush=True)


def caption_identity(row):
    return row['start'], row['end_exclusive'], tuple(row['source_indices'])


def reusable_caption_records(old_bundle, samples, source_hashes):
    old_rows = {caption_identity(row): row for row in old_bundle['records']}
    reused, changed = {}, []
    for sample in samples:
        previous = old_rows.get(caption_identity(sample))
        same = previous is not None and all(
            old_bundle['source_frame_hashes'].get(str(i)) == source_hashes[str(i)]
            for i in sample['source_indices'])
        if same:
            reused[str(sample['segment'])] = previous['text']
        else:
            changed.append(sample)
    return reused, changed


def prepare_captions(directory, protocol, selection):
    from PIL import Image, ImageDraw, ImageFont
    extraction = directory / 'extraction'
    folder = extraction / 'assistant_captions'
    folder.mkdir(exist_ok=True)
    old_bundle_path = locate(directory.name, OLD) / 'extraction/assistant_captions/captions_bundle.json'
    old_bundle = validate_bundle(old_bundle_path, old_bundle_path.parent.parent)
    if old_bundle['input_sha256'] != protocol['input_sha256']:
        raise ValueError('caption reuse source mismatch')
    samples = expected_samples(protocol, selection)
    hashes = {str(i): protocol['source_frame_hashes'][str(i)] for row in samples for i in row['source_indices']}
    reused, changed = reusable_caption_records(old_bundle, samples, hashes)
    metadata = dict(selection_freeze_sha256=sha256(extraction/'selection_freeze.json'),
                    input_sha256=protocol['input_sha256'], samples=samples, source_frame_hashes=hashes,
                    old_bundle_path=str(old_bundle_path), old_bundle_sha256=sha256(old_bundle_path),
                    reused_captions=reused, changed_samples=changed,
                    reuse_rule='Exact [start,end) interval, four source indices and corresponding PNG hashes; same source video.',
                    sampling='Original PLLaVA four samples from half-open interval; assistant caption authorship.')
    target = folder / 'samples.json'
    if target.exists() and read_json(target) != metadata:
        raise ValueError('caption preparation already exists with different evidence')
    write_json(target, metadata)
    compact = folder / 'changed_compact'
    compact.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    pages = []
    for page, start in enumerate(range(0, len(changed), 12)):
        values = changed[start:start+12]
        canvas = Image.new('RGB', (1536, ((len(values)+1)//2)*151+24), '#171717')
        draw = ImageDraw.Draw(canvas)
        draw.text((4,3),directory.name+' | CHANGED INTERVALS | source samples for DIRECT REVIEW',font=font,fill='white')
        for j, sample in enumerate(values):
            x, y = j%2*768, j//2*151+24
            draw.text((x+3,y+1),f"SEG {sample['segment']:03d} | [{sample['start']},{sample['end_exclusive']})",font=font,fill='white')
            for k, i in enumerate(sample['source_indices']):
                with Image.open(Path(protocol['source_frames']) / f'{i}.png') as im:
                    canvas.paste(im.convert('RGB').resize((192,107)),(x+k*192,y+41))
                draw.text((x+k*192+3,y+21),f'f{i}',font=font,fill='white')
        path=compact/f'{page:03d}.jpg'
        if not path.exists():
            canvas.save(path,quality=96)
        pages.append(str(path.relative_to(folder)))
    write_json(folder / 'review_required.json', dict(status='PENDING_DIRECT_REVIEW' if changed else 'EXACT_REUSE_ONLY',
                changed_segments=[r['segment'] for r in changed], pages=pages,
                exact_reused_segments=len(reused), source_bundle_sha256=metadata['old_bundle_sha256']))
    if not changed:
        freeze(directory.name, None)
    print(json.dumps(dict(video=directory.name, keyframes=len(selection['indices']),
                          skem_pairs=selection['skem_comparisons'], reused=len(reused), changed=len(changed))),flush=True)


def freeze(video, authored_path):
    directory = locate(video)
    extraction = directory / 'extraction'
    folder = extraction / 'assistant_captions'
    protocol, selection = hybrid.verify_selection(extraction)
    metadata = read_json(folder / 'samples.json')
    request = read_json(folder / 'review_required.json')
    expected = expected_samples(protocol, selection)
    if metadata['samples'] != expected:
        raise ValueError('source samples changed')
    old_path = Path(metadata['old_bundle_path'])
    if sha256(old_path) != metadata['old_bundle_sha256']:
        raise ValueError('reused source caption bundle changed')
    authored = read_json(authored_path) if authored_path else dict(captions={}, reviewed_pages=[])
    required = {str(r['segment']) for r in metadata['changed_samples']}
    if set(authored['captions']) != required or sorted(authored['reviewed_pages']) != sorted(request['pages']):
        raise ValueError('every changed interval and actual review page is required')
    texts = dict(metadata['reused_captions'], **authored['captions'])
    records = [dict(row, text=texts[str(row['segment'])]) for row in expected]
    evidence = {'samples.json':sha256(folder/'samples.json'),
                'review_required.json':sha256(folder/'review_required.json')}
    evidence.update({p:sha256(folder/p) for p in request['pages']})
    payload = dict(version=1,status='CAPTIONS_PREPARED_RECONSTRUCTION_NOT_STARTED',
        input_sha256=protocol['input_sha256'], selection_freeze_sha256=sha256(extraction/'selection_freeze.json'),
        source_frame_hashes=metadata['source_frame_hashes'],records=records,
        author='Codex assistant after direct source review; prior captions reused only for identical source intervals and samples.',
        protocol='FC-LGVSC faithful caption criteria: visible objects, location, size, direction, visible portions, occlusion, observed changes and blur; no invented identities or unseen details.',
        scope='Assistant-authored captions with genuine sequential InternVL SKEM selection; not independent ground truth.',
        sampling='Original four PLLaVA source sample indices; no PLLaVA caption inference.',
        caption_reuse=dict(source_bundle=str(old_path),source_bundle_sha256=metadata['old_bundle_sha256'],
                           reused_segments=sorted(map(int,metadata['reused_captions'])),
                           newly_reviewed_segments=sorted(map(int,authored['captions']))),
        evidence=evidence, reviewed_caption_pages=authored['reviewed_pages'],
        hallucination_mitigation_verified=False,reconstruction_started=False)
    result=dict(payload,checksum=fingerprint(payload))
    target=folder/'captions_bundle.json'
    if target.exists() and read_json(target)!=result:
        raise ValueError('preserve frozen corrected captions')
    write_json(folder/'authored_correction.json',authored)
    write_json(target,result)
    validate_bundle(target,extraction)


def run():
    from semantic_transmission.etri_60s_check import lock
    with lock(REPO/'.local/etri_60s_check.lock'), lock(ROOT/'.runner.lock'):
        scorer = None
        scorer_identity = None
        checked_models = None
        while True:
            directories = sorted(ROOT.glob('*/*/extraction/protocol.json'),
                                 key=lambda p:(p.parent.parent.parent.name!='webvid',str(p)))
            pending = [p for p in directories if not (p.parent/'selection_freeze.json').exists()
                       or not (p.parent/'assistant_captions/review_required.json').exists()]
            if not pending:
                if (ROOT/'classification_complete.json').exists():
                    if len(directories)!=103:
                        raise ValueError('classification marked complete without 103 protocols')
                    write_json(ROOT/'selection_batch_complete.json',dict(status='ALL_103_SELECTED',completed_utc=now()))
                    return
                time.sleep(5)
                continue
            for path in pending:
                extraction, directory = path.parent, path.parent.parent
                protocol=hybrid.validate_protocol(extraction)
                if sha256(Path(protocol['reclassification_path'])) != protocol['reclassification_sha256']:
                    raise ValueError('classification evidence changed')
                model_identity=fingerprint(protocol['model_files'])
                if checked_models != model_identity:
                    for p,info in protocol['model_files'].items():
                        if sha256(p)!=info['sha256']:
                            raise ValueError('model weights changed')
                    checked_models=model_identity
                current_identity=fingerprint(dict(model_files=protocol['model_files'],prompts=protocol['prompts'],
                    config={k:protocol['config'][k] for k in ['seed','flash_attn','internvl_gpu_head','internvl_offload_layers','internvl_compact_kv_cache','max_tiles','max_new_tokens']}))
                if scorer is None:
                    scorer=hybrid.InternVLScorer(protocol)
                    scorer_identity=current_identity
                elif scorer_identity != current_identity:
                    raise ValueError('cannot share scorer across different inference configurations')
                else:
                    scorer.protocol=protocol
                write_json(ROOT/'progress.json',dict(stage='SELECTION',video=directory.name,updated_utc=now()))
                if (extraction/'selection_freeze.json').exists():
                    _,selection=hybrid.verify_selection(extraction)
                else:
                    begin=time.perf_counter()
                    def score(a,b):
                        value=hybrid.cached_score(extraction,protocol,a,b,scorer)
                        write_json(ROOT/'progress.json',dict(stage='SEQUENTIAL_SKEM',video=directory.name,
                                   reference=a,candidate=b,p_yes=value['p_yes'],p_no=value['p_no'],updated_utc=now()))
                        return value
                    keys,records=hybrid.choose_keys(protocol['frames'],[r['frame'] for r in protocol['candidates']],
                        [r['frame'] for r in protocol['candidates'] if r['mandatory']],score,
                        max_gap=protocol['max_gap_frames'],threshold=protocol['threshold'])
                    selection=hybrid.export_selection(extraction,protocol,keys,records,time.perf_counter()-begin,scorer.load_seconds)
                    hybrid.verify_selection(extraction)
                prepare_captions(directory,protocol,selection)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','classify','run','freeze','classification-complete'])
    p.add_argument('--video');p.add_argument('--authored',type=Path)
    args=p.parse_args()
    if args.action=='init':init()
    elif args.action=='classify':classify(args.video,args.authored)
    elif args.action=='freeze':freeze(args.video,args.authored)
    elif args.action=='classification-complete':
        assert len(list(ROOT.glob('*/*/extraction/protocol.json')))==103
        write_json(ROOT/'classification_complete.json',dict(completed_utc=now(),videos=103))
    else:run()


if __name__=='__main__':
    main()
