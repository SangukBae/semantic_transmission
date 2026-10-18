"""DCVC-RT inference wrapper extracted from the pre-unification BSU codec.

The pinned codec operations are unchanged from bsu_dcvc_keys at fd37e28.
Keep experiment/quality imports out of packet and channel-only processes.
"""
import io
import sys

from .artifacts import sha256
from .wire import pack, unpack

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
        import numpy as np
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
