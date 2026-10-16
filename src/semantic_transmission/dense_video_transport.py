"""BSU dense appearance/motion transport, replacing independent RGB key packets.

The sender pays for every frame in a temporally predictive HEVC guide. The
receiver gets no original images, optical flow, tracking cache, or source path.
This is a hybrid video/semantic codec, not a reproduction of CGVC or DiSCo.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

from .packets import accounting, validate_rows
from .wire import pack, unpack

POLICY = 'bsu_dense_appearance_v1'


def validate_context(context):
    video = context['video']
    for k in ('frames', 'width', 'height'):
        if type(video[k]) is not int or video[k] < 1:
            raise ValueError('invalid video geometry')
    if not 0 < video['fps'] <= 240:
        raise ValueError('invalid video fps')
    keys = context['indices']
    if (not keys or any(type(i) is not int for i in keys) or
            keys != sorted(set(keys)) or keys[0] != 0 or keys[-1] != video['frames']-1):
        raise ValueError('invalid virtual key timeline')
    validate_rows(context['segments'])
    if len(context['segments']) != len(keys)-1:
        raise ValueError('caption/interval count mismatch')


def encode_packet(context, stream, width, height):
    validate_context(context)
    if (type(width) is not int or type(height) is not int or min(width, height) < 2
            or width % 2 or height % 2):
        raise ValueError('HEVC guide requires positive even dimensions')
    metadata = zlib.compress(json.dumps(context, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode(), 9)
    return pack(dict(schema=POLICY, context_codec='zlib', context_bytes=len(metadata),
        guide_width=width, guide_height=height, guide_codec='hevc', container='matroska',
        guide_sha256=hashlib.sha256(stream).hexdigest(), independent_rgb_keys=0), metadata+stream)


def decode_packet(packet):
    header, payload = unpack(packet)
    if header.get('schema') != POLICY or header.get('context_codec') != 'zlib':
        raise ValueError('unsupported dense appearance packet')
    n = header['context_bytes']
    if type(n) is not int or not 0 < n < len(payload):
        raise ValueError('invalid context length')
    # Bound decompression and reject trailing/concatenated context streams.
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload[:n], 4*1024*1024+1)
    if len(raw) > 4*1024*1024 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('invalid or oversized compressed context')
    context = json.loads(raw)
    validate_context(context)
    stream = payload[n:]
    if hashlib.sha256(stream).hexdigest() != header['guide_sha256']:
        raise ValueError('guide checksum mismatch')
    if (header.get('guide_codec') != 'hevc' or header.get('container') != 'matroska'
            or any(type(header[k]) is not int or not 2 <= header[k] <= 8192 or header[k] % 2
                   for k in ('guide_width', 'guide_height'))):
        raise ValueError('invalid guide decoder contract')
    return context, stream, header


def encode_stream(source_frames, frames, fps, width, height, crf, target):
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '2',
        '-framerate', str(fps), '-start_number', '0', '-i', str(Path(source_frames)/'%d.png'),
        '-frames:v', str(frames), '-an', '-vf',
        f'scale={width}:{height}:flags=bicubic:in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
        '-c:v', 'libx265', '-preset', 'slow', '-crf', str(crf),
        '-x265-params', 'pools=none:frame-threads=2:log-level=error:range=limited:colormatrix=bt709',
        '-color_range', 'tv', '-colorspace', 'bt709', '-map_metadata', '-1', '-y', str(target)]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return command


def send(source_frames, context, directory, *, uses_per_frame=512, width=288, height=160,
         allow_budget_miss=False):
    """Source-only rate search. No BSU outputs, target labels or metrics select it."""
    from .artifacts import write_json, sha256
    validate_context(context)
    if type(uses_per_frame) is not int or uses_per_frame <= 0:
        raise ValueError('positive integer channel budget required')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    capacity = (context['video']['frames']*uses_per_frame // 2304)*2304
    if capacity < 2304:
        raise ValueError('budget smaller than one LDPC block')
    lo, hi, candidates = 0, 51, []
    while lo <= hi:
        crf = (lo+hi)//2
        target = directory/f'guide_crf_{crf}.mkv'
        command = encode_stream(source_frames, context['video']['frames'], context['video']['fps'],
                                width, height, crf, target)
        packet = encode_packet(context, target.read_bytes(), width, height)
        uses = accounting(len(packet))['complex_channel_uses']
        candidates.append(dict(crf=crf, bytes=len(packet), channel_uses=uses,
                               stream=str(target), command=command, stream_sha256=sha256(target)))
        if uses <= capacity:
            hi = crf-1
        else:
            lo = crf+1
    feasible = [r for r in candidates if r['channel_uses'] <= capacity]
    if not feasible and not allow_budget_miss:
        raise ValueError('guide cannot fit budget even at CRF 51')
    selected = min(feasible, key=lambda r:r['crf']) if feasible else min(candidates,key=lambda r:r['channel_uses'])
    packet = encode_packet(context, Path(selected['stream']).read_bytes(), width, height)
    (directory/'packet.bin').write_bytes(packet)
    record = dict(policy=POLICY, capacity_channel_uses=capacity, uses_per_frame=uses_per_frame,
        channel_uses=accounting(len(packet))['complex_channel_uses'], packet_sha256=sha256(directory/'packet.bin'),
        selected=selected, candidates=candidates, independent_rgb_keys=0,
        budget_met=selected['channel_uses']<=capacity,
        source_used_for_encoding_only=True, selection_uses_quality_metrics=False)
    write_json(directory/'sender.json', record)
    return record


def decode_video(stream, directory, context, header):
    import numpy as np
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    movie = directory/'guide.mkv'; movie.write_bytes(stream)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=codec_name,width,height,nb_read_frames,r_frame_rate',
        '-of', 'json', str(movie)]))['streams'][0]
    w, h = header['guide_width'], header['guide_height']
    if (info['codec_name'] != 'hevc' or (info['width'], info['height']) != (w, h)
            or int(info['nb_read_frames']) != context['video']['frames']):
        raise ValueError('received guide geometry differs from paid context')
    from fractions import Fraction
    if float(Fraction(info['r_frame_rate'])) != context['video']['fps']:
        raise ValueError('received guide frame rate differs')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(movie),
        '-vf', 'scale=in_range=tv:out_range=pc:in_color_matrix=bt709,format=rgb24',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    raw = subprocess.check_output(command)
    if len(raw) != context['video']['frames']*w*h*3:
        raise ValueError('incomplete guide decode')
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
