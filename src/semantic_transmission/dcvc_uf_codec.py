"""Pinned DCVC-UF HT-S bitstream adapter, with an independent receiver."""
import io
import sys

from .artifacts import sha256

FILES = ('cvpr2026_image.pth.tar', 'cvpr2026_video_hts.pth.tar')


def qp_schedule(count, qp):
    """One QP for the I frame and each subsequent eight-frame chunk."""
    if type(count) is not int or count < 1:
        raise ValueError('invalid frame count')
    size = 1 + (count - 1 + 7) // 8
    values = [qp] * size if type(qp) is int else qp
    if (not isinstance(values, list) or len(values) != size or
            any(type(q) is not int or not 0 <= q <= 63 for q in values)):
        raise ValueError('invalid UF QP schedule')
    return values


class Codec:
    def __init__(self, vendor, weights, identities):
        for name in FILES:
            if sha256(weights/name) != identities[name]:
                raise ValueError('UF checkpoint changed: '+name)
        sys.path.insert(0, str(vendor))
        import torch
        from src.models.image_model import DMCI
        from src.models.video_model_ht import DMC
        from src.utils.common import ModelStructure, get_state_dict, set_torch_env
        set_torch_env()
        self.torch = torch
        self.stream = torch.cuda.Stream(device='cuda', priority=0)
        torch.cuda.set_stream(self.stream)
        self.i, self.p = DMCI(), DMC(ModelStructure.HTS)
        for model, name in zip((self.i, self.p), FILES):
            model.eval()
            model.load_state_dict(get_state_dict(str(weights/name)))
            model.update(0)
            model.half().cuda().to(memory_format=torch.channels_last)
        # Upstream initializes DMCI's proxy only inside compress(). Initialize
        # from weights/CDFs directly, so a fresh receiver never encodes a source.
        from inference_extensions_cuda import DMCIProxy
        self.i.proxy = DMCIProxy()
        self.i.proxy.set_param(self.i.add_cdf_to_state_dict(self.i.state_dict()),
                              self.i.gaussian_encoder.skip_thres)

    def encode(self, images, qp):
        from src.utils.transforms import rgb2ycbcr
        from src.utils.stream_helper import SPSHelper, write_sps, write_ip
        t = self.torch
        height, width = images.shape[1:3]
        right, bottom = self.i.get_padding_size(height, width, 16)
        out, helper = io.BytesIO(), SPSHelper()
        pos, records = 0, []
        qps = qp_schedule(len(images), qp)
        with t.inference_mode():
            while pos < len(images):
                intra = pos == 0
                current_qp = qps[len(records)]
                valid = min(1 if intra else 8, len(images)-pos)
                batch = []
                for image in images[pos:pos+valid]:
                    x = t.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).cuda()
                    batch.append(rgb2ycbcr(x.float()/255))
                while not intra and len(batch) < 8:
                    batch.append(batch[-1])
                x = (t.cat(batch, 1).half()-0.5).to(memory_format=t.channels_last)
                reset = int(not intra and (pos+8) % 32 == 1)
                if intra:
                    encoded = self.i.compress(x, current_qp, bottom, right)
                    self.p.clear_dpb()
                    self.p.add_ref_feature_from_frame(encoded['x_hat'])
                else:
                    encoded = self.p.compress(x, current_qp, reset, bottom, right)
                sps = dict(sps_id=-1, height=height, width=width)
                sid, new = helper.get_sps_id(sps)
                sps['sps_id'] = sid
                if new:
                    write_sps(out, sps)
                count = write_ip(out, intra, sid, current_qp, encoded['ec_parallel'], reset, encoded['bit_stream'])
                records.append(dict(position=pos, valid_frames=valid, intra=intra,
                                    qp=current_qp, reset_feature_memory=reset,
                                    padded_frames=len(batch)-valid, entropy_bytes=len(encoded['bit_stream']),
                                    framed_bytes=count))
                pos += valid
        t.cuda.synchronize()
        return out.getvalue(), records

    def decode(self, stream, count, video, qp_expected):
        import numpy as np
        from src.utils.transforms import ycbcr2rgb
        from src.utils.stream_helper import (SPSHelper, NalType, read_header,
                                            read_sps_remaining, read_ip_remaining)
        t, source, helper, frames = self.torch, io.BytesIO(stream), SPSHelper(), []
        qps = qp_schedule(count, qp_expected)
        chunk_number = 0
        with t.inference_mode():
            while len(frames) < count:
                header = read_header(source)
                while header['nal_type'] == NalType.NAL_SPS:
                    helper.add_sps_by_id(read_sps_remaining(source, header['sps_id']))
                    header = read_header(source)
                sps = helper.get_sps_by_id(header['sps_id'])
                qp, ec, reset, bits = read_ip_remaining(source)
                intra = header['nal_type'] == NalType.NAL_I
                if (sps['height'] != video['height'] or sps['width'] != video['width'] or
                        qp != qps[chunk_number] or intra != (len(frames) == 0) or
                        header['nal_type'] not in (NalType.NAL_I, NalType.NAL_P)):
                    raise ValueError('UF elementary stream disagrees with packet')
                if intra:
                    decoded = self.i.decompress(bits, sps, qp, ec)
                    self.p.clear_dpb()
                    self.p.add_ref_feature_from_frame(decoded['x_hat'], apply_feature_adaptor=False)
                else:
                    decoded = self.p.decompress(bits, sps, qp, ec, reset)
                batch = [decoded['x_hat']] if intra else decoded['x_hat']
                if len(batch) != (1 if intra else 8):
                    raise ValueError('unexpected UF decoded chunk size')
                for x in batch[:count-len(frames)]:
                    rgb = ycbcr2rgb(x[:, :, :video['height'], :video['width']]+0.5)
                    frames.append((rgb*255).clamp(0, 255).round().byte()[0].permute(1, 2, 0).cpu().numpy())
                chunk_number += 1
        if source.tell() != len(stream):
            raise ValueError('unconsumed UF stream bytes')
        return np.stack(frames)
