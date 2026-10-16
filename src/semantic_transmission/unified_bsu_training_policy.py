"""Source-disjoint, full-timeline sampling and resumable ETRI training policy.

Long sources supply adjacent receiver windows throughout their duration. This
does not add long-term memory to the frozen 17-frame generator. Legacy TVSum AI
captions are explicitly auxiliary conditions, never independent semantic truth.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import unified_bsu_core as core
from .artifacts import sha256
from .webvid5 import fingerprint
from .wire import unpack


def config(path):
    c = core.read(path)
    if c['policy'] != 'bsu_unified_etri_training_v1':
        raise ValueError('unsupported training policy')
    for k in ('train_sequences_per_source', 'validation_sequences_per_source',
              'max_steps', 'min_joint_steps', 'min_coverage_epochs', 'gradient_accumulation',
              'validate_every', 'save_every', 'loss_patience', 'quality_every', 'quality_patience',
              'quality_sources_per_dataset', 'quality_budget_per_frame'):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError('invalid training setting: '+k)
    if (c['validate_every'] % c['save_every'] or c['quality_every'] % c['validate_every']
            or c['min_joint_steps'] >= c['max_steps']):
        raise ValueError('invalid training/validation cadence')
    if (not 0 < c['min_learning_rate'] <= c['learning_rate'] < 1 or not 0 < c['relative_min_delta'] < 1
            or not 0 < c['additional_validation_fraction'] < 1
            or not 0 <= c['overlap_consistency_weight'] <= 1):
        raise ValueError('invalid training weights')
    for key in ('budgets_per_frame', 'validation_seeds', 'quality_durations_seconds', 'sequence_window_counts'):
        if not c[key] or any(type(v) is not int or v < 1 for v in c[key]):
            raise ValueError('invalid '+key)
    if (not c['validation_noise_fractions']
            or any(not 0 < v < 1 for v in c['validation_noise_fractions'])):
        raise ValueError('invalid validation noise fractions')
    for key in ('quality_psnr_tolerance_db', 'quality_lpips_tolerance', 'quality_temporal_tolerance'):
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError('invalid quality guard')
    return c


def crop_context(context, start, count):
    """Keep actual caption changes and paid timestamps on the cropped timeline."""
    stop = start+count-1
    if not 0 <= start < stop < context['video']['frames']:
        raise ValueError('crop outside source timeline')
    keys = [start]+[k for k in context['indices'] if start < k < stop]+[stop]
    segments = []
    for a, b in zip(keys, keys[1:]):
        i = next(i for i, (x, y) in enumerate(zip(context['indices'], context['indices'][1:]))
                 if x <= a < y)
        segments.append(dict(context['segments'][i], path=f'clips/sample/{a-start}_{b-start}.mp4'))
    result = dict(video=dict(context['video'], frames=count), indices=[k-start for k in keys],
                  segments=segments, decoder={})
    core.validate_context(result)
    return result


def sequence_starts(frames, length, limit, boundaries):
    """Stratify over the entire timeline; reserve slots for semantic transitions."""
    if frames < 17 or length < 17 or limit < 2:
        raise ValueError('need >=17 frames and at least two temporal samples')
    length = min(length, frames)
    count = min(limit, frames-length+1)
    # Endpoints are unconditional; neither motion nor one long scene can
    # suppress the middle/tail of a source.
    uniform = np.linspace(0, frames-length, max(2, (count+1)//2)).round().astype(int).tolist()
    transitions = sorted({max(0, min(frames-length, int(k)-length//2)) for k in boundaries[1:-1]})
    selected = list(dict.fromkeys(uniform))[:count]
    if transitions:
        for i in np.linspace(0, len(transitions)-1, min(count-len(selected), len(transitions))).round().astype(int):
            if transitions[i] not in selected:
                selected.append(transitions[i])
    for s in np.linspace(0, frames-length, count).round().astype(int):
        if len(selected) >= count:
            break
        if int(s) not in selected:
            selected.append(int(s))
    return sorted(selected)


def source_catalog(selection, c):
    """Retain old source splits and exclude ETRI originals, including extensions."""
    root = core.REPO
    evidence = {str(Path(selection).resolve()): sha256(selection)}
    old = core.read(selection)
    excluded_hashes, excluded_ids = set(), set()
    for name in c['excluded_manifests']:
        path = root / name
        evidence[str(path)] = sha256(path)
        for row in csv.DictReader(path.open()):
            excluded_hashes.add(row.get('source_sha256', ''))
            excluded_ids.update(row.get(k, '') for k in ('source_id', 'source_video_id'))
    excluded_hashes.discard(''); excluded_ids.discard('')
    known_roles = dict(old.get('roles', {}))
    for e in old.get('cases', [])+old['examples']:
        key = e['source_sha256']
        if key in known_roles and known_roles[key] != e['role']:
            raise ValueError('conflicting previous source split')
        known_roles[key] = e['role']
    usage_path = root / c['previous_source_usage']
    evidence[str(usage_path)] = sha256(usage_path)
    usage = core.read(usage_path)
    extra_roles = {e['source_sha256']: e['split'] for e in usage['newly_used_evaluation_sources']}
    by_hash = {}
    for e in old['examples']:
        key = e['source_sha256']
        if e['role'] not in ('train', 'validation') or key in excluded_hashes:
            continue
        if key in by_hash:
            continue
        header, _ = unpack(Path(e['packet']).read_bytes())
        context = dict(video=header['video'], indices=[r['index'] for r in header['keyframes']],
                       segments=header['segments'], decoder={})
        core.validate_context(context)
        evidence[e['packet']] = sha256(e['packet'])
        by_hash[key] = dict(source_id=e['id'], source_sha256=key, dataset=e['dataset'].lower(),
            role=e['role'], directory=e['source_directory'], context=context,
            caption_provenance='existing_received_semantic_packet')
    inventory_path = root / c['additional_inventory']
    evidence[str(inventory_path)] = sha256(inventory_path)
    candidates, excluded = [], []
    for v in core.read(inventory_path)['videos']:
        if v['dataset'].lower() != 'tvsum':
            continue
        key = v['source_sha256']
        if (key in excluded_hashes or v['source_id'] in excluded_ids or v['benchmark_memberships']
                or known_roles.get(key) == 'test' or extra_roles.get(key) == 'test'):
            excluded.append(dict(id=v['id'], source_sha256=key, reason='reserved_evaluation_original'))
            continue
        if key in by_hash:
            raise ValueError('duplicate additional source')
        folder = inventory_path.parent / v['dataset'].lower() / v['id']
        prepared_path = folder / 'source_prepared.json'
        caption_path = folder / 'extraction/assistant_captions/captions_bundle.json'
        prepared, bundle = core.read(prepared_path), core.read(caption_path)
        if (prepared['source_sha256'] != key or bundle['input_sha256'] != prepared['normalized_sha256']
                or sha256(v['source_path']) != key
                or sha256(prepared['normalized_video']) != prepared['normalized_sha256']):
            raise ValueError('additional source/caption provenance mismatch: '+v['id'])
        records = bundle['records']; frames = prepared['frames']
        if (not records or records[0]['start'] != 0 or records[-1]['end_exclusive'] not in (frames-1, frames)
                or any(a['end_exclusive'] != b['start'] for a, b in zip(records, records[1:]))):
            raise ValueError('caption timeline incomplete: '+v['id'])
        keys = [r['start'] for r in records]+[frames-1]
        if keys[-2] == keys[-1]:
            records = records[:-1]; keys = keys[:-1]
        context = dict(video={k: prepared[k] for k in ('frames', 'fps', 'width', 'height')},
            indices=keys, segments=[dict(path=f'clips/sample/{a}_{b}.mp4', text=r['text'], flow=1.)
                for a, b, r in zip(keys, keys[1:], records)], decoder={})
        core.validate_context(context)
        for path in (prepared_path, caption_path):
            evidence[str(path)] = sha256(path)
        for index, digest in bundle['source_frame_hashes'].items():
            if prepared['source_frame_hashes'].get(index) != digest:
                raise ValueError('caption samples differ from normalized source')
        candidates.append(dict(source_id=v['id'], source_sha256=key, dataset='tvsum',
            role=known_roles.get(key, extra_roles.get(key)), directory=prepared['source_frames'],
            context=context, expected_frame_hashes=prepared['source_frame_hashes'],
            caption_provenance='legacy_source_reviewed_AI_auxiliary_not_independent_truth',
            caption_bundle=str(caption_path)))
    undecided = sorted([r for r in candidates if r['role'] is None], key=lambda r: r['source_sha256'])
    nval = max(1, round(len(candidates)*c['additional_validation_fraction'])-
               sum(r['role'] == 'validation' for r in candidates))
    for i, row in enumerate(undecided):
        row['role'] = 'validation' if i < nval else 'train'
    for row in candidates:
        by_hash[row['source_sha256']] = row
    rows = sorted(by_hash.values(), key=lambda r: (r['dataset'], r['source_sha256']))
    if any(r['role'] not in ('train', 'validation') for r in rows):
        raise ValueError('reserved source entered training catalog')
    if not any(r['dataset'] == 'tvsum' and r['role'] == 'train' for r in rows):
        raise ValueError('no independent long-video training sources')
    return rows, evidence, excluded


def materialize_example(source, start, count, budget, name):
    files = [Path(source['directory']) / f'{i}.png' for i in range(start, start+count)]
    hashes = [sha256(p) for p in files]
    expected = source.get('expected_frame_hashes')
    if expected and any(expected.get(str(i)) != h for i, h in zip(range(start, start+count), hashes)):
        raise ValueError('normalized training frame changed')
    return dict(name=name, role=source['role'], source_id=source['source_id'], dataset=source['dataset'],
        source_sha256=source['source_sha256'], source_frames=[str(p) for p in files], source_hashes=hashes,
        source_start=start, source_total_frames=source['context']['video']['frames'],
        context=crop_context(source['context'], start, count), budget_per_frame=budget,
        caption_provenance=source['caption_provenance'])


def expanded_plan(selection, c, profile):
    sources, evidence, excluded = source_catalog(selection, c)
    examples, quality = [], []
    lengths = [17+(n-1)*(17-profile['window_overlap']) for n in c['sequence_window_counts']]
    for source in sources:
        frames = source['context']['video']['frames']
        limit = c['train_sequences_per_source' if source['role'] == 'train' else 'validation_sequences_per_source']
        for j, position in enumerate(sequence_starts(frames, min(lengths), limit, source['context']['indices'])):
            count = min(lengths[j % len(lengths)], frames)
            start = min(position, frames-count)
            for budget in c['budgets_per_frame']:
                examples.append(materialize_example(source, start, count, budget,
                    f'{source["source_id"]}_{start:06d}_{count}f_b{budget}'))
    examples = list({e['name']: e for e in examples}.values())
    # Fixed validation-only sources, selected before any model scores exist.
    datasets = sorted({s['dataset'] for s in sources})
    for dataset in datasets:
        available = [s for s in sources if s['dataset'] == dataset and s['role'] == 'validation']
        for i in range(min(len(available), c['quality_sources_per_dataset'])):
            requested = c['quality_durations_seconds'][i % len(c['quality_durations_seconds'])]
            source = next((s for s in available if s['context']['video']['frames']/s['context']['video']['fps'] >= requested), available[0])
            available.remove(source)
            frames = source['context']['video']['frames']
            count = min(frames, round(requested*source['context']['video']['fps']))
            # Alternate the beginning and tail; long-video validation must see
            # the end as well as the first frames of a source.
            start = 0 if i % 2 == 0 else frames-count
            quality.append(materialize_example(source, start, count, c['quality_budget_per_frame'],
                f'{source["source_id"]}_{start:06d}_{count}f'))
    if not any(e['context']['video']['frames']/e['context']['video']['fps'] >= 60 for e in quality):
        raise ValueError('ETRI policy requires at least one >=60-second validation sequence')
    summary = dict(sources={r: sum(s['role'] == r for s in sources) for r in ('train', 'validation')},
        datasets={d: {r: sum(s['dataset'] == d and s['role'] == r for s in sources)
                      for r in ('train', 'validation')} for d in datasets},
        sequences={r: sum(e['role'] == r for e in examples) for r in ('train', 'validation')},
        window_examples=sum(len(core.windows(len(e['source_frames']), overlap=profile['window_overlap'])) for e in examples),
        sequence_frames=lengths, source_duration_seconds=[s['context']['video']['frames']/s['context']['video']['fps'] for s in sources],
        quality_seconds=[len(e['source_frames'])/e['context']['video']['fps'] for e in quality],
        excluded_originals=len(excluded), long_term_generator_memory=False,
        training_completed=False, sufficient_training_verified=False)
    return examples, quality, summary, evidence, excluded


def balanced_order(examples, rng):
    """Cover every sequence; balance datasets, sources, and each source's rates."""
    groups = defaultdict(lambda: defaultdict(list))
    for i, e in enumerate(examples):
        groups[e['dataset']][e['source_sha256']].append(i)
    max_sources = max(map(len, groups.values()))
    max_examples = max(len(v) for g in groups.values() for v in g.values())
    shuffled = {d: {s: rng.permutation(v).tolist() for s, v in g.items()} for d, g in groups.items()}
    source_orders = {d: rng.permutation(sorted(g)).tolist() for d, g in groups.items()}
    order = []
    for j in range(max_examples):
        for k in range(max_sources):
            for d in rng.permutation(sorted(groups)):
                s = source_orders[d][k % len(source_orders[d])]
                indices = shuffled[d][s]
                order.append(indices[j % len(indices)])
    return order


def overlap_error_loss(previous, current, previous_start, current_start, frames=17):
    """Proxy consistency of prediction errors on shared frames, not RGB truth.

    VAE endpoints have different context. Subtract each window's own target
    before interpolating the error to the shared frame timeline. The preceding
    window is detached to keep sequential backpropagation memory bounded.
    """
    import torch.nn.functional as F
    overlap = previous_start+frames-current_start
    if not 0 < overlap < frames:
        raise ValueError('invalid overlapping training windows')
    shape = (frames, *current.shape[-2:])
    a = F.interpolate(previous.detach().float(), size=shape, mode='trilinear', align_corners=True)
    b = F.interpolate(current.float(), size=shape, mode='trilinear', align_corners=True)
    return F.l1_loss(a[:, :, -overlap:], b[:, :, :overlap])


def update_monitor(state, value, step, c):
    from .adapter_training_state import update_plateau
    monitor, improved = update_plateau(state, value, c['relative_min_delta'])
    if improved:
        monitor['step'] = step
    return monitor, improved


def plateau_ready(state, c, warmup):
    completed_epochs = state['epoch']-(state['offset'] < len(state['order']))
    return (state['step']-warmup >= c['min_joint_steps'] and completed_epochs >= c['min_coverage_epochs']
            and state.get('loss_monitor', {}).get('bad', 0) >= c['loss_patience'])
