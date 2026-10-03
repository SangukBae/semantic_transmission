"""Frozen diagnostic subset, six precision/frame-count profiles, no automatic judge."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import qwen35_advanced_caption as advanced
from qwen35_advanced_caption import base


def indices_for(row, count):
    if count == 4:
        return row['source_indices']
    a, b = row['start'], row['end_exclusive']
    indices = set(row['source_indices'])
    target = min(count, b-a)
    while len(indices) < target:
        candidates = [i for i in range(a,b) if i not in indices]
        indices.add(max(candidates, key=lambda i: (min(abs(i-j) for j in indices), -i)))
    return sorted(indices)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--quantization', choices=['nf4','int8'])
    args = p.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'plan.json').read_text())
    assert plan['runner_sha256'] == base.sha256(__file__)
    assert plan['advanced_sha256'] == base.sha256(advanced.__file__)
    dataset = json.loads(Path(plan['inputs_file']).read_text())
    assert base.sha256(plan['inputs_file']) == plan['inputs_sha256']
    if args.quantization is None:
        outcomes = []
        for quantization in ['int8','nf4']:
            with (root/f'{quantization}.log').open('w') as log:
                code = subprocess.call([sys.executable, __file__, '--root', str(root),
                                        '--quantization', quantization], stdout=log, stderr=log)
            outcomes.append(dict(quantization=quantization, exit_code=code))
            base.write_json(root/'suite_progress.json', dict(status='RUNNING', outcomes=outcomes))
        base.write_json(root/'suite_progress.json', dict(status='INFERENCE_DONE_REVIEW_PENDING', outcomes=outcomes))
        print(json.dumps(outcomes), flush=True)
        return
    import torch
    from PIL import Image
    quantization = args.quantization
    source = Path(dataset['source_frames'])
    dest = root/quantization
    dest.mkdir(exist_ok=True)
    try:
        model, processor, runtime = advanced.load(quantization)
    except Exception as exc:
        base.write_json(dest/'LOAD_FAILED.json', dict(status='LOAD_FAILED', error_type=type(exc).__name__, error=str(exc), created_utc=base.now()))
        raise
    base.write_json(dest/'runtime.json', runtime)
    rows = {x['segment']: x for x in dataset['records']}
    for count in plan['frame_counts']:
        for i in plan['segments']:
            path = dest/f'frames{count:02d}'/f'{i:03d}.json'
            if path.exists():
                raise FileExistsError(path)
            row = rows[i]
            indices = indices_for(row, count)
            images = []
            for j in indices:
                with Image.open(source/f'{j}.png') as im:
                    images.append(im.convert('RGB'))
            labels = [f'Source frame {j}, timestamp {j/dataset["fps"]:.3f} seconds' for j in indices]
            try:
                result = advanced.generate(model, processor, images, labels, plan['prompt'],
                                           plan['max_pixels'], plan['max_new_tokens'])
            except torch.cuda.OutOfMemoryError as exc:
                result = dict(status='OUT_OF_MEMORY_WITH_RESERVE', error=str(exc))
                gc.collect()
                torch.cuda.empty_cache()
            result.update(segment=i, quantization=quantization, target_frame_count=count,
                          source_indices=indices, source_sha256={str(j):base.sha256(source/f'{j}.png') for j in indices},
                          start=row['start'], end_exclusive=row['end_exclusive'],
                          created_utc=base.now(), plan_sha256=base.sha256(root/'plan.json'))
            base.write_json(path, result)
            base.write_json(root/'progress.json', dict(status='RUNNING', quantization=quantization,
                target_frame_count=count, segment=i, result_status=result['status'], updated_utc=base.now()))
            print(json.dumps(dict(quantization=quantization, frames=count, segment=i,
                status=result['status'], seconds=result.get('generation_seconds'), caption=result.get('caption')),ensure_ascii=False), flush=True)
            if result['status'] == 'OUT_OF_MEMORY_WITH_RESERVE':
                break
    base.write_json(dest/'DONE.json', dict(status='PROFILE_PROBES_COMPLETE', created_utc=base.now()))


if __name__ == '__main__':
    main()
