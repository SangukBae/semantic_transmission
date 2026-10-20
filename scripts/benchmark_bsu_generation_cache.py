"""Bounded received-input replay; never writes into an experiment candidate."""
import argparse
import json
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'cached'], required=True)
    parser.add_argument('--source', default='892')
    parser.add_argument('--arms', nargs='+', default=['baseline', 'unequal_rates'])
    parser.add_argument('--gpu-cache-gib', type=float, default=4.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from semantic_transmission import ltx_receiver as ltx
    from semantic_transmission.artifacts import write_json, sha256
    from semantic_transmission.webvid_ablation import snapshot
    protocol = json.loads((args.root/'protocol.json').read_text())
    assets = dict(protocol['runtime'], execution_offload=protocol['policy']['execution_offload'])
    optimization = None
    if args.mode == 'cached':
        from semantic_transmission.ltx_execution_cache import install
        optimization = install(gpu_cache_bytes=int(args.gpu_cache_gib*2**30))
    generator = None
    def factory(current):
        nonlocal generator
        if generator is None:
            generator = ltx.Generator(current)
        return generator
    results = []
    for arm in args.arms:
        source = args.root/'labels'/args.source/arm
        request = json.loads((source/'request.json').read_text())
        output = args.output/arm
        torch.cuda.reset_peak_memory_stats()
        before_io = dict(line.split(': ') for line in Path('/proc/self/io').read_text().splitlines())
        started = time.monotonic()
        ltx.reconstruct(request, assets, output, factory=factory)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        after_io = dict(line.split(': ') for line in Path('/proc/self/io').read_text().splitlines())
        record = dict(arm=arm, seconds=elapsed, request_sha256=sha256(source/'request.json'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            disk_read_bytes=int(after_io['read_bytes'])-int(before_io['read_bytes']),
            mp4_sha256=sha256(output/'reconstruction.mp4'),
            frames=snapshot(output, ['frames']),
            cache=optimization.stats if optimization else None)
        results.append(record)
        write_json(args.output/'measurements.json', dict(mode=args.mode, results=results))
        print('BENCHMARK '+json.dumps({k:v for k,v in record.items() if k != 'frames'}), flush=True)
    write_json(args.output/'RESULT.json', dict(status='COMPLETE', mode=args.mode,
        source=args.source, results=results, gpu_cache_gib=args.gpu_cache_gib,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    main()
