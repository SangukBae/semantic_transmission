"""Experimental received low-rate video conditions; no learned weights."""
from pathlib import Path
import subprocess

from .receiver_temporal_alignment import temporal_layout
from .wire import pack, unpack

FIELD = 'structure_video_v1'


def add_video(packet, bitstream, width, height, timestamps, frames):
    header, payload = unpack(packet)
    if FIELD in header or not bitstream:
        raise ValueError('duplicate/empty structure video')
    if (width < 2 or height < 2 or width % 2 or height % 2 or not timestamps
            or any(type(t) is not int or not 0 <= t < frames for t in timestamps)
            or list(timestamps) != sorted(set(timestamps))):
        raise ValueError('invalid structure video geometry/timestamps')
    header[FIELD] = dict(codec='h264_annexb_yuv420p', width=width, height=height,
        timestamps=list(timestamps), frames=frames, offset=len(payload), bytes=len(bitstream),
        spatial_upsample='opencv_linear', temporal_upsample='linear_with_endpoint_hold')
    return pack(header, payload + bitstream)


def video_payload(packet):
    header, payload = unpack(packet)
    item = header[FIELD]
    offset, size = item['offset'], item['bytes']
    timestamps = item['timestamps']
    rate_end = max(k['rate_offset'] + k['rate_bytes'] for k in header['keyframes'])
    if (offset != rate_end or size < 1 or offset + size != len(payload)
            or item['frames'] != header['video']['frames']
            or not timestamps or timestamps != sorted(set(timestamps))
            or any(type(t) is not int or not 0 <= t < item['frames'] for t in timestamps)
            or item['codec'] != 'h264_annexb_yuv420p'):
        raise ValueError('invalid received structure payload')
    return item, payload[offset:offset + size]


def compress(frames, path, fps, crf):
    h, w = frames.shape[1:3]
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
        '-g', str(len(frames)), '-f', 'h264', str(path)], input=frames.tobytes(), check=True)
    return Path(path).read_bytes()


def decode_video(bitstream, item):
    import numpy as np
    data = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-f', 'h264', '-i',
        'pipe:0', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '2', 'pipe:1'],
        input=bitstream, stdout=subprocess.PIPE, check=True).stdout
    expected = len(item['timestamps']) * item['height'] * item['width'] * 3
    if len(data) != expected:
        raise ValueError('decoded structure video length differs from packet')
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, item['height'], item['width'], 3).copy()


def expand_video(decoded, item, height, width):
    import cv2
    import numpy as np
    if len(decoded) != len(item['timestamps']):
        raise ValueError('structure timestamp/frame mismatch')
    times = np.asarray(item['timestamps'])
    output = []
    for frame in range(item['frames']):
        right = min(int(np.searchsorted(times, frame, side='right')), len(times)-1)
        left = max(0, right-1) if frame < times[-1] else right
        weight = float(np.clip((frame-times[left]) / max(1, times[right]-times[left]), 0, 1))
        low = decoded[left].astype('float32')*(1-weight) + decoded[right].astype('float32')*weight
        output.append(cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR).round().clip(0,255).astype('uint8'))
    return np.stack(output)


def anchor_slots(frames, timestamps, protected):
    """Map sample times to matching slots in a full, phase-aligned video VAE.

    Representative times use the final decoded frame in each causal group,
    accounting for each 17-frame block's temporal left padding. The condition
    itself comes from encoding the full upsampled received sequence, not from
    misplacing independent single-image latents into a compressed timeline.
    """
    points = []
    for block in temporal_layout(frames):
        for j in range(block['latents']):
            time = block['frame_start'] + min(block['frames']-1, max(0, 4*(j+1)-1-block['left_padding']))
            points.append((block['latent_start']+j, time))
    requested = [min(points, key=lambda p: (abs(p[1]-t), p[0]))[0] for t in timestamps[1:]]
    selected = sorted(set(requested) - set(protected))
    return dict(slots=selected, requested=requested, protected=sorted(protected),
                latent_representative_frames=[t for _, t in points],
                removed_duplicates=len(requested)-len(set(requested)),
                removed_protected=sorted(set(requested) & set(protected)))
