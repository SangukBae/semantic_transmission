"""CPU screening of predictive key-image coding, not an end-to-end BSU result."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .artifacts import sha256, write_json
from .bsu_lossless_validation import audit_case, event
from .packets import accounting
from .wire import pack

REPO = Path(__file__).resolve().parents[2]
CRFS = (18, 23, 28)
CODECS = ('libx264rgb', 'libx265')


def quality(source, restored):
    if source.shape != restored.shape:
        raise ValueError('decoded key dimensions changed')
    psnr, ssim = [], []
    for a, b in zip(source, restored):
        mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        psnr.append(10 * np.log10(255 ** 2 / max(mse, 1e-12)))
        ssim.append(float(structural_similarity(a, b, data_range=255, channel_axis=2,
                                               gaussian_weights=True, sigma=1.5, use_sample_covariance=False)))
    return dict(psnr=psnr, ssim=ssim, mean_psnr=statistics.mean(psnr), mean_ssim=statistics.mean(ssim))


def load_images(directory, keys):
    images = []
    for key in keys:
        with Image.open(directory / f'{key}.png') as image:
            images.append(np.asarray(image.convert('RGB')))
    return np.stack(images)


def encode_decode(source, codec, crf, destination):
    n, h, w, _ = source.shape
    video = destination / 'keys.mkv'
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '24', '-i', 'pipe:0',
           '-frames:v', str(n), '-an', '-c:v', codec, '-preset', 'medium', '-crf', str(crf), '-threads', '2']
    if codec == 'libx265':
        # 4:4:4 retains chroma detail; declare both conversion directions.
        cmd += ['-vf', 'scale=in_range=pc:out_range=pc:out_color_matrix=bt709,format=yuv444p',
                '-x265-params', 'pools=none:frame-threads=2:log-level=error:range=full:colormatrix=bt709',
                '-color_range', 'pc', '-colorspace', 'bt709']
    cmd += [str(video)]
    completed = subprocess.run(cmd, input=source.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (destination / 'encode.log').write_bytes(completed.stderr)
    completed.check_returncode()
    decode = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '2', '-i', str(video)]
    if codec == 'libx265':
        decode += ['-vf', 'scale=in_range=pc:out_range=pc:in_color_matrix=bt709,format=rgb24']
    decode += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    completed = subprocess.run(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if len(completed.stdout) != source.nbytes:
        raise ValueError('codec did not preserve the number/shape of key images')
    write_json(destination / 'commands.json', dict(encode=cmd, decode=decode))
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(source.shape), video


def run(baseline, output):
    if output.exists():
        raise ValueError('output already exists; choose a new directory')
    result = json.loads((baseline / 'RESULT.json').read_text())
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', dict(code_sha256=sha256(Path(__file__)),
        baseline_result_sha256=sha256(baseline / 'RESULT.json'), codecs=CODECS, crfs=CRFS,
        scope='key-image codec screening only; no channel simulation or BSU regeneration',
        key_gate='every key PSNR >= original received key PSNR and RGB SSIM >= original received key SSIM',
        accounting='new self-contained key-video packet through same LDPC/16QAM, plus ALL unchanged legacy digital costs',
        source_only=True, no_receiver_noise_or_generated_output_used_by_encoder=True,
        ffmpeg_version=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]))
    rows, done, start = [], 0, time.monotonic()
    for item in result['videos']:
        case = baseline / 'benchmark/snr_10' / item['id']
        run_dir = case / 'base/reconstruction/event_pairs/run'
        _, _, audit = audit_case(case, item)
        keys = json.loads((run_dir / 'keyframes.json').read_text())['indices']
        source = load_images(baseline / 'inputs/kinetics' / item['id'] / 'source_frames', keys)
        old = quality(source, load_images(run_dir / 'receiver/frames/sample/key_frames_received', keys))
        for codec in CODECS:
            for crf in CRFS:
                dest = output / item['id'] / f'{codec}_{crf}'
                dest.mkdir(parents=True)
                decoded, video = encode_decode(source, codec, crf, dest)
                current = quality(source, decoded)
                # Timestamp map, container headers and decoder identity are all
                # paid. Retaining old NTSCC metadata overcounts, conservatively.
                wire = pack(dict(schema='bsu_key_video_probe_v1', indices=keys, codec=codec,
                    container='matroska', width=source.shape[2], height=source.shape[1],
                    color='rgb24' if codec == 'libx264rgb' else 'bt709_yuv444p_full'), video.read_bytes())
                (dest / 'key_video.bin').write_bytes(wire)
                uses = accounting(len(wire))['complex_channel_uses'] + audit['legacy_digital_uses']
                key_gate = all(a >= b for a, b in zip(current['psnr'], old['psnr'])) and all(
                    a >= b for a, b in zip(current['ssim'], old['ssim']))
                row = dict(id=item['id'], codec=codec, crf=crf, keys=len(keys),
                    key_video_packet_bytes=len(wire), total_uses_if_digital_channel_succeeds=uses,
                    baseline_total_uses=audit['legacy_total_uses'],
                    saving_percent=100 * (1 - uses / audit['legacy_total_uses']),
                    key_psnr_delta=current['mean_psnr'] - old['mean_psnr'],
                    key_ssim_delta=current['mean_ssim'] - old['mean_ssim'],
                    every_key_psnr_ssim_noninferior=key_gate, key_gate_and_lower_rate=key_gate and uses < audit['legacy_total_uses'])
                write_json(dest / 'result.json', dict(**row, original_key_quality=old, candidate_key_quality=current,
                    source_key_hashes={str(k): sha256(baseline / 'inputs/kinetics' / item['id'] / 'source_frames' / f'{k}.png') for k in keys},
                    video_sha256=sha256(video), packet_sha256=sha256(dest / 'key_video.bin')))
                rows.append(row); done += 1
                event('key_codec_screen', done, len(result['videos']) * len(CODECS) * len(CRFS))
    summaries = []
    for codec in CODECS:
        for crf in CRFS:
            selected = [r for r in rows if r['codec'] == codec and r['crf'] == crf]
            old_uses = sum(r['baseline_total_uses'] for r in selected)
            new_uses = sum(r['total_uses_if_digital_channel_succeeds'] for r in selected)
            summaries.append(dict(codec=codec, crf=crf, saving_percent=100 * (1 - new_uses / old_uses),
                key_gate_pass_videos=sum(r['every_key_psnr_ssim_noninferior'] for r in selected),
                key_gate_and_lower_rate_videos=sum(r['key_gate_and_lower_rate'] for r in selected),
                mean_key_psnr_delta=statistics.mean(r['key_psnr_delta'] for r in selected),
                mean_key_ssim_delta=statistics.mean(r['key_ssim_delta'] for r in selected)))
    with (output / 'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output / 'RESULT.json', dict(status='COMPLETE_KEY_IMAGE_SCREEN_ONLY', summary=summaries,
        videos=len(result['videos']), elapsed_seconds=time.monotonic() - start,
        final_bsu_quality_preservation_proven=False, channel_transmission_tested=False,
        limits=['key RGB PSNR/SSIM do not guarantee final BSU perceptual, motion or action quality',
                'reported uses include LDPC padding/parity but assume successful digital decoding',
                'codecs change key pixels; final matched multi-seed BSU evaluation still required',
                'no production promotion; no thresholds relaxed; source videos reused for screening']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=REPO / 'outputs/bsu_kinetics_context')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
