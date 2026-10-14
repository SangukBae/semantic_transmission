"""BSU default receiver: paid trajectory/cadence, then received-key context.

The historical sequence implementation and packet policy remain frozen. The
new receiver calls the validated key-context operator, without evaluation data.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time

from . import bsu_sequence_refinement as legacy
from .artifacts import write_json
from .wire import unpack

DEFAULT_MODE = 'combined_context'
MODES = (DEFAULT_MODE, *legacy.MODES)
MODULE = 'semantic_transmission.bsu_receiver'
POLICY = 'bsu_received_key_context_v1'


def sender_mode(mode):
    if mode not in MODES:
        raise ValueError('unsupported BSU refinement mode')
    return 'combined' if mode == DEFAULT_MODE else mode


def prepare(job, output):
    # Identical sender extraction, packet bytes and budget to legacy combined.
    return legacy.prepare({**job, 'mode': sender_mode(job['mode'])}, output)


def apply_key_context(frame_paths, keys, key_paths, destination, callback=None):
    """Receiver-only boundary; inputs are current RGB and actual received keys."""
    import cv2
    import numpy as np
    from PIL import Image
    from .receiver_key_context import preserve_context
    legacy.validate_keys(keys, len(frame_paths))
    if len(key_paths) != len(keys):
        raise ValueError('received key count differs from timeline')
    cv2.setNumThreads(2); cv2.setRNGSeed(2025)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    audits = []
    for index, (a, b) in enumerate(zip(keys, keys[1:])):
        def rgb(path):
            with Image.open(path) as image:
                return np.asarray(image.convert('RGB'))
        frames = np.stack([rgb(path) for path in frame_paths[a:b+1]])
        received = [rgb(path) for path in key_paths[index:index+2]]
        started = time.monotonic()
        output, audit = preserve_context(frames, received,
            lambda n,total: legacy.progress(f'WINDOW {index+1}/{len(keys)-1} key context',n,total,started))
        for offset, pixels in enumerate(output):
            target = destination/f'{a+offset:05d}.png'
            if index and offset == 0:
                if not np.array_equal(rgb(target), pixels):
                    raise ValueError('shared received key boundary differs')
                continue
            temporary = target.with_suffix('.tmp.png'); temporary.unlink(missing_ok=True)
            if np.array_equal(pixels, frames[offset]):
                os.link(frame_paths[a+offset], temporary)
            else:
                Image.fromarray(pixels).save(temporary)
            temporary.replace(target)
        audits.append(dict(start=a,end=b,anchor=audit))
        if callback: callback(b+1, len(frame_paths))
    return audits


def refine_windows(frame_paths, keys, key_directory, trajectories, destination,
                   mode=DEFAULT_MODE, callback=None):
    sender_mode(mode)
    if mode != DEFAULT_MODE:
        return legacy.refine_windows(frame_paths, keys, key_directory, trajectories, destination, mode, callback)
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    # Separate passes preserve the validated ordering and the RNG state used by
    # trajectory RANSAC. The intermediate PNGs are lossless and locally owned.
    with tempfile.TemporaryDirectory(prefix='bsu-combined-', dir=destination.parent) as temporary:
        current = Path(temporary)/'frames'
        print('START BSU receiver pass 1/2: trajectory + cadence', flush=True)
        baseline = legacy.refine_windows(frame_paths, keys, key_directory, trajectories, current, 'combined', callback)
        print('START BSU receiver pass 2/2: received-key context', flush=True)
        contexts = apply_key_context([current/f'{i:05d}.png' for i in range(len(frame_paths))], keys,
            [key_directory/f'{k}.png' for k in keys], destination, callback)
    return [{**before,'anchor':after['anchor']} for before,after in zip(baseline,contexts)]


def encode_frames(frames, movie, count, fps):
    """The same delivery encoder/settings used by the validated context arm."""
    temporary = movie.with_name('encoding.tmp.mp4')
    subprocess.run(['ffmpeg','-y','-v','error','-nostdin','-threads','2','-framerate',str(fps),
        '-i',str(frames/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p',
        '-threads','2',str(temporary)],check=True)
    from .etri_60s_check import pts_audit
    timeline = pts_audit(temporary,count,fps)
    temporary.replace(movie)
    return timeline


def refine(job, received, output):
    mode = job.get('mode', DEFAULT_MODE); sender_mode(mode)
    if mode != DEFAULT_MODE:
        return legacy.refine(job, received, output)
    run, _, frames = legacy.paths_for_run(job['base'])
    packet = (run/'received/metadata.bin').read_bytes(); header,_ = unpack(packet)
    paths = sorted(frames.glob('*.png')); count = header['video']['frames']
    if len(paths) != count or [p.name for p in paths] != [f'{i:05d}.png' for i in range(count)]:
        raise ValueError('base generated frames do not cover the complete timeline')
    keys = [k['index'] for k in header['keyframes']]
    trajectory = received/'trajectory.bin'
    tracks = legacy.decode_packet(trajectory.read_bytes(), packet) if trajectory.exists() else {}
    started = time.monotonic()
    audits = refine_windows(paths, keys, run/'receiver/frames/sample/key_frames_received', tracks, output/'frames', mode,
        lambda n,total: legacy.progress('REFINEMENT pass frames',n,total,started))
    print('EXPORT full-length MP4; encoding ETA unknown', flush=True)
    timeline = encode_frames(output/'frames',output/'reconstruction.mp4',count,header['video']['fps'])
    write_json(output/'refinement.json',dict(policy=POLICY,mode=mode,windows=audits,
        receiver_reads_source=False,received_key_endpoints_exact=True,key_endpoint_reference='received_rgb',
        additional_context_transmission=0,frames=count,fps=header['video']['fps'],timeline=timeline,
        seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','channel','refine','evaluate'))
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args();job=legacy.read(args.job)
    from .bsu_webvid import verify_plan
    plan=legacy.read(args.job.parent.parent.parent/'protocol.json');verify_plan(plan)
    if job['benchmark_signature']!=plan['signature'] or job['mode']!=plan['mode']:
        raise ValueError('worker job belongs to another benchmark or receiver mode')
    sender_mode(job['mode']);root=args.job.parent
    if args.stage=='prepare': prepare(job,root/'sender')
    elif args.stage=='channel': legacy.channel(job,root/'sender',root/'received')
    elif args.stage=='refine': refine(job,root/'received',root/'refined')
    else: legacy.evaluate(job,root/'refined')


if __name__=='__main__': main()
