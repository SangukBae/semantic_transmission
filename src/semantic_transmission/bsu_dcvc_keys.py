"""Actual DCVC-RT key-image bitstreams; opt-in screening, not final BSU quality."""
import argparse
import io
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image

from .artifacts import sha256, write_json
from .bsu_key_codec_probe import load_images, quality
from .bsu_lossless_validation import audit_case, event
from .lossless_transport import choose, cost
from .packets import accounting
from .wire import pack, unpack

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = {
    'intra': '555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9',
    'inter': 'b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6',
}


class Codec:
    def __init__(self, vendor, checkpoints):
        sys.path.insert(0, str(vendor))
        import torch
        from src.models.image_model import DMCI
        from src.models.video_model import DMC
        from src.utils.common import get_state_dict, set_torch_env
        from src.utils import stream_helper
        from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
        from src.layers.cuda_inference import replicate_pad
        set_torch_env()
        self.torch = torch
        self.stream = stream_helper
        self.rgb2ycbcr, self.ycbcr2rgb, self.pad = rgb2ycbcr, ycbcr2rgb, replicate_pad
        self.i, self.p = DMCI(), DMC()
        for name, model in [('intra', self.i), ('inter', self.p)]:
            path = checkpoints / f'dcvc_rt_{name}.tar'
            if sha256(path) != WEIGHTS[name]:
                raise ValueError('unpinned DCVC checkpoint: ' + str(path))
            model.load_state_dict(get_state_dict(path), strict=True)
            model.cuda().eval()
            model.update(force_zero_thres=0.12)
            model.half()
            model.set_use_two_entropy_coders(False)

    def encode(self, images, indices, qp, intra_only=False):
        """TX reads original selected images only; no received or generated images."""
        t, s = self.torch, self.stream
        n, h, w, _ = images.shape
        if len(indices) != n or not 0 <= qp <= 63:
            raise ValueError('invalid key count or QP')
        output, helper = io.BytesIO(), s.SPSHelper()
        padding_r, padding_b = self.i.get_padding_size(h, w, 16)
        self.p.clear_dpb(); self.p.set_curr_poc(0)
        last_qp = 0
        with t.no_grad():
            for index, image in enumerate(images):
                x = t.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
                x = self.pad(self.rgb2ycbcr(x).half(), padding_b, padding_r)
                is_i = index == 0 or intra_only
                ada = not is_i and index % 32 == 1
                sps = dict(sps_id=-1, height=h, width=w, ec_part=0, use_ada_i=int(ada))
                if is_i:
                    curr_qp = qp
                    encoded = self.i.compress(x, curr_qp)
                    self.p.clear_dpb(); self.p.add_ref_frame(None, encoded['x_hat'])
                else:
                    if ada:
                        self.p.prepare_feature_adaptor_i(last_qp)
                    curr_qp = self.p.shift_qp(qp, [0, 1, 0, 2, 0, 2, 0, 2][index % 8])
                    encoded = self.p.compress(x, curr_qp)
                    last_qp = curr_qp
                sps_id, new = helper.get_sps_id(sps)
                sps['sps_id'] = sps_id
                if new:
                    s.write_sps(output, sps)
                s.write_ip(output, is_i, sps_id, curr_qp, encoded['bit_stream'])
        # Pay all NAL headers, timestamp indices, dimensions and model identity.
        return pack(dict(schema='bsu_dcvc_rt_keys_v1', indices=indices, height=h, width=w,
                         weights=WEIGHTS, color='full_range_ycbcr444_bt709',
                         upstream='cbdae87a5445114cdc7f48816da63ea80bdeac40'), output.getvalue())

    def decode(self, packet):
        """RX has only packet + fixed weights; reset all encoder reference state."""
        t, s = self.torch, self.stream
        header, payload = unpack(packet)
        if header['schema'] != 'bsu_dcvc_rt_keys_v1' or header['weights'] != WEIGHTS:
            raise ValueError('unsupported key packet')
        self.p.clear_dpb(); self.p.set_curr_poc(0)
        source, helper, result = io.BytesIO(payload), s.SPSHelper(), []
        with t.no_grad():
            for index in range(len(header['indices'])):
                nal = s.read_header(source)
                while nal['nal_type'] == s.NalType.NAL_SPS:
                    helper.add_sps_by_id(s.read_sps_remaining(source, nal['sps_id']))
                    nal = s.read_header(source)
                sps = helper.get_sps_by_id(nal['sps_id'])
                if sps is None or (sps['height'], sps['width']) != (header['height'], header['width']):
                    raise ValueError('missing/inconsistent SPS')
                qp, bits = s.read_ip_remaining(source)
                if nal['nal_type'] == s.NalType.NAL_I:
                    decoded = self.i.decompress(bits, sps, qp)
                    self.p.clear_dpb(); self.p.add_ref_frame(None, decoded['x_hat'])
                elif nal['nal_type'] == s.NalType.NAL_P and index:
                    if sps['use_ada_i']:
                        self.p.reset_ref_feature()
                    decoded = self.p.decompress(bits, sps, qp)
                else:
                    raise ValueError('invalid first frame / NAL type')
                x = self.ycbcr2rgb(decoded['x_hat'][:, :, :header['height'], :header['width']])
                result.append((x[0].permute(1, 2, 0).float().clamp(0, 1) * 255).round().byte().cpu().numpy())
        if source.read():
            raise ValueError('unconsumed key bitstream')
        return header['indices'], np.stack(result)


def run(args):
    baseline, output = args.baseline.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('output already exists; use a new directory')
    original = json.loads((baseline / 'RESULT.json').read_text())
    videos = original['videos'][:args.limit] if args.limit else original['videos']
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', dict(
        code_sha256=sha256(Path(__file__)), baseline_sha256=sha256(baseline / 'RESULT.json'),
        baseline=str(baseline), weights=WEIGHTS, qps=args.qps, intra_only=args.intra_only,
        vendor_code={str(p.relative_to(args.vendor)): sha256(p) for p in sorted(args.vendor.rglob('*.py'))},
        scope='real bitstream key screening; not channel success or final video quality',
        accounting='actual framed entropy stream plus original losslessly bundled digital side information; LDPC 6144/9216 16QAM',
        unchanged='all original key indices, captions, flow, trajectories, video timeline',
        key_gate='every key PSNR and RGB SSIM >= original received key; screening only',
        installed_weights_are_shared_model=True, physical_link_overhead_included=False))
    start = time.monotonic()
    codec = Codec(args.vendor.resolve(), args.checkpoints.resolve())
    rows = []
    for video in videos:
        case = baseline / 'benchmark/snr_10' / video['id']
        run_dir = case / 'base/reconstruction/event_pairs/run'
        metadata, trajectory, audit = audit_case(case, video)
        _, messages = choose(metadata, trajectory, method='xz', bundle=True)
        digital = cost(messages)
        keys = json.loads((run_dir / 'keyframes.json').read_text())['indices']
        source = load_images(baseline / 'inputs/kinetics' / video['id'] / 'source_frames', keys)
        old = quality(source, load_images(run_dir / 'receiver/frames/sample/key_frames_received', keys))
        for qp in args.qps:
            dest = output / video['id'] / f'qp_{qp}'
            dest.mkdir(parents=True)
            begin = time.monotonic()
            packet = codec.encode(source, keys, qp, args.intra_only)
            (dest / 'keys.bin').write_bytes(packet)
            returned_keys, decoded = codec.decode((dest / 'keys.bin').read_bytes())
            if keys != returned_keys or source.shape != decoded.shape:
                raise ValueError('key timeline changed')
            (dest / 'decoded').mkdir()
            for key, image in zip(keys, decoded):
                Image.fromarray(image).save(dest / 'decoded' / f'{key}.png')
            current = quality(source, decoded)
            uses = accounting(len(packet))['complex_channel_uses'] + digital
            gate = all(a >= b for a, b in zip(current['psnr'], old['psnr'])) and all(
                a >= b for a, b in zip(current['ssim'], old['ssim']))
            row = dict(id=video['id'], qp=qp, keys=len(keys), packet_bytes=len(packet),
                       key_uses=accounting(len(packet))['complex_channel_uses'], digital_uses=digital,
                       total_uses=uses, baseline_total_uses=audit['legacy_total_uses'],
                       saving_percent=100 * (1 - uses / audit['legacy_total_uses']),
                       key_psnr_delta=current['mean_psnr'] - old['mean_psnr'],
                       key_ssim_delta=current['mean_ssim'] - old['mean_ssim'],
                       every_key_noninferior=gate, lower_rate=uses < audit['legacy_total_uses'])
            write_json(dest / 'result.json', dict(**row, elapsed_seconds=time.monotonic() - begin,
                original_key_quality=old, candidate_key_quality=current, packet_sha256=sha256(dest/'keys.bin'),
                decoded_hashes={str(k): sha256(dest/'decoded'/f'{k}.png') for k in keys},
                source_hashes={str(k): sha256(baseline/'inputs/kinetics'/video['id']/'source_frames'/f'{k}.png') for k in keys}))
            rows.append(row)
            event('dcvc_actual_bitstream', len(rows), len(videos) * len(args.qps))
    summaries = []
    for qp in args.qps:
        subset = [r for r in rows if r['qp'] == qp]
        uses = sum(r['total_uses'] for r in subset)
        old_uses = sum(r['baseline_total_uses'] for r in subset)
        summaries.append(dict(qp=qp, total_uses=uses, baseline_total_uses=old_uses,
            saving_percent=100*(1-uses/old_uses),
            every_key_noninferior_videos=sum(r['every_key_noninferior'] for r in subset),
            noninferior_lower_rate_videos=sum(r['every_key_noninferior'] and r['lower_rate'] for r in subset),
            mean_key_psnr_delta=statistics.mean(r['key_psnr_delta'] for r in subset),
            mean_key_ssim_delta=statistics.mean(r['key_ssim_delta'] for r in subset)))
    write_json(output/'RESULT.json', dict(status='COMPLETE_ACTUAL_BITSTREAM_KEY_SCREEN',
        summary=summaries, rows=rows, elapsed_seconds=time.monotonic()-start,
        channel_transmission_tested=False, final_bsu_quality_preservation_proven=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=REPO/'outputs/bsu_kinetics_context')
    p.add_argument('--vendor', type=Path, default=REPO/'.local/vendor/DCVC-RT')
    p.add_argument('--checkpoints', type=Path, default=REPO/'.local/checkpoints/dcvc_rt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qps', nargs='+', type=int, default=[0, 16, 32, 48, 63])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--intra-only', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
