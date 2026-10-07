"""Matched 2x2 selector/caption ablation using one frozen visual encoding.

All four cells are generated anew with identical precision, prompts and visual
tokens. References are used only after inference to count evidence coverage.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

from .qwen35_training import (CONFIG, RUN, engine, images_for, labels_for,
    rows_for, checkpoint_path, release_gpu)
from .qwen35_training_data import DEFAULT_DATA, read_json, write_json, sha256, verify
from .qwen35_training_eval import verify_baseline


def encode(model, processor, row, config):
    import torch
    eng = engine()
    inputs = eng.base.preprocess(processor, images_for(row, range(8)),
        labels_for(row, range(8)), 'Describe the visible frames.', config['max_pixels']).to('cuda')
    base = model.base_model.model if hasattr(model, 'peft_config') else model
    with torch.inference_mode():
        visual = base.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
        tokens = torch.stack(visual.pooler_output).detach()
    return {'tokens': tokens, 'pooled': tokens.float().mean(1), 'grid': inputs.image_grid_thw}


def generate(model, processor, row, keys, features, config, adapter=True):
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    start = time.monotonic()
    inputs = engine().base.preprocess(processor, images_for(row, keys), labels_for(row, keys),
                                     config['caption_prompt'], config['max_pixels']).to('cuda')
    assert torch.equal(inputs.image_grid_thw, features['grid'][keys])
    inputs.pop('pixel_values')
    context = nullcontext() if adapter else model.disable_adapter()
    with context, torch.inference_mode():
        output = model.generate(**inputs,
            mm_encoder_outputs={'image': BaseModelOutputWithPooling(pooler_output=tuple(features['tokens'][keys].unbind(0)))},
            max_new_tokens=config['max_caption_tokens'], do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    tokens = output[0, inputs.input_ids.shape[1]:]
    caption = processor.decode(tokens, skip_special_tokens=True).strip()
    return {'caption': caption, 'generated_tokens': len(tokens),
            'seconds': time.monotonic()-start, 'input_tokens': inputs.input_ids.shape[1],
            'grid': inputs.image_grid_thw.tolist(),
            'status': 'COMPLETE' if caption and len(tokens) < config['max_caption_tokens'] else 'EMPTY_OR_TRUNCATED'}


def load_inference(config, checkpoint):
    import torch
    from peft import PeftModel
    from .qwen35_multitask import TemporalSelector
    model, processor, runtime = engine().load('nf4')
    for name, p in model.named_parameters():
        if 'norm' in name and 'visual' not in name and p.dtype in (torch.bfloat16, torch.float16):
            p.data = p.data.float()
    model = PeftModel.from_pretrained(model, checkpoint/'adapter', is_trainable=False).eval()
    head = TemporalSelector(hidden=config['head_hidden'], candidates=8, keys=4).cuda().eval()
    head.load_state_dict(torch.load(checkpoint/'head.pt', weights_only=True, map_location='cuda'))
    return model, processor, head, runtime


def run(args):
    import torch
    from .qwen35_multitask import coverage
    verify(args.data)
    verify_baseline(args.data)
    config = read_json(CONFIG)
    ckpt = checkpoint_path(RUN/'caption')
    out = args.run
    protocol = {'data_sha256': sha256(args.data/'freeze.json'), 'checkpoint': str(ckpt),
        'checkpoint_manifest_sha256': sha256(ckpt/'complete.json'), 'config': config,
        'implementation_sha256': sha256(Path(__file__)),
        'cells': ['baseline_keys_base_caption', 'baseline_keys_lora_caption', 'learned_keys_base_caption', 'learned_keys_lora_caption'],
        'scope': 'New matched generation for every cell; fixed choices from frozen baseline or learned selector; no labels enter generation',
        'precision': 'NF4, BF16, nonvisual norms FP32 for every cell'}
    if (out/'protocol.json').exists() and read_json(out/'protocol.json') != protocol:
        raise ValueError('Ablation conditions changed; use a new run')
    write_json(out/'protocol.json', protocol)
    rows = [r for s in ('validation','test') for r in rows_for(args.data,s)]
    pending = [r for r in rows if not (out/'records'/f'{r["id"]}.json').exists()]
    if pending:
        model, processor, head, runtime = load_inference(config, ckpt)
        for row in pending:
            start = time.monotonic()
            features = encode(model, processor, row, config)
            with torch.no_grad():
                pred = head(features['pooled'], config['temperature'])
            baseline_path = RUN/'baseline/records'/f'{row["id"]}.json'
            choices = {'baseline': read_json(baseline_path)['keys'], 'learned': pred['indices'].tolist()}
            cells = {}
            # Counterbalance cell order across windows; no selective retry.
            order = [(k,a) for k in choices for a in ('base','lora')]
            if row['window'] % 2:
                order.reverse()
            generated = {}
            for selector, captioner in order:
                keys = choices[selector]
                identity = (tuple(keys), captioner)
                reused = identity in generated
                if not reused:
                    generated[identity] = generate(model, processor, row, keys, features, config, captioner=='lora')
                name = f'{selector}_keys_{captioner}_caption'
                cells[name] = dict(generated[identity], keys=keys, coverage=coverage(row,keys),
                                  identical_input_reuse=reused,
                                  source_hashes=[row['frames'][i]['sha256'] for i in keys])
                print(row['id'], name, cells[name]['status'], flush=True)
            write_json(out/'records'/f'{row["id"]}.json', {'id':row['id'], 'split':row['split'], 'cells':cells,
                'seconds':time.monotonic()-start, 'baseline_record_sha256':sha256(baseline_path),
                'protocol_sha256':sha256(out/'protocol.json'), 'runtime':runtime})
            del features,pred
        del model,processor,head
        release_gpu()
    files = {f'records/{r["id"]}.json':sha256(out/'records'/f'{r["id"]}.json') for r in rows}
    for name in files:
        assert read_json(out/name)['protocol_sha256']==sha256(out/'protocol.json')
    write_json(out/'generation_freeze.json', {'protocol_sha256':sha256(out/'protocol.json'),'files':files})
    write_json(out/'RESULT.json', {'status':'GENERATED_REVIEW_PENDING','windows':len(rows),'condition_windows':4*len(rows)})


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--run',type=Path,default=Path('outputs/qwen35_factorial_20261005'))
    args=p.parse_args()
    import fcntl
    with open('/tmp/qwen35_multitask_gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
