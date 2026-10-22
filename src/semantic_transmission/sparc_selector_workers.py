"""Resident, separately initialized semantic/UF/channel/RIFE worker processes."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .sparc_selector_data import read, save, sha

REPO = Path(__file__).resolve().parents[2]


class Client:
    def __init__(self, role, runtime, root):
        self.role = role
        self.log = (root/(role+'.log')).open('a')
        env = dict(os.environ, PYTHONPATH=str(REPO/'src'), PYTHONNOUSERSITE='1',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env.pop('LD_LIBRARY_PATH', None)
        if role == 'channel':
            env['CUDA_VISIBLE_DEVICES'] = '-1'
        self.proc = subprocess.Popen([runtime, '-m', __name__, role], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, env=env)

    def call(self, job):
        self.proc.stdin.write(json.dumps(job)+'\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f'{self.role} worker exited {self.proc.poll()}; see {self.log.name}')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(f'{self.role}: {result["error"]}; see {self.log.name}')
        return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=20)
        self.log.close()


def semantic_worker():
    from . import sparc_semantics as semantics
    from .qwen35_backend import LocalQwen35
    model = []
    def factory(plan):
        if not model:
            model.append(LocalQwen35(plan))
        return model[0]
    original = semantics.Session
    class SharedSession(original):
        def __init__(self, root, plan, **kwargs):
            super().__init__(root, plan, factory=kwargs.get('factory', factory))
    semantics.Session = SharedSession
    def run(job):
        result = semantics.select(job['source'], read(REPO/'configs/sparc.json'), Path(job['output']))
        return dict(keys=result['keys'], checksum=result['checksum'])
    return run


def codec_worker():
    import numpy as np
    from .dcvc_uf_codec import Codec
    from .sparc_uf import WEIGHTS
    from . import sparc_uf_rate_transport as transport
    from .sparc_selection import scene_groups
    from .packets import accounting
    model = Codec(REPO/'.local/dcvc_uf_vendor', REPO/'.local/dcvc_uf_weights', WEIGHTS)
    def run(job):
        if job['action'] == 'encode':
            frames = np.load(job['frames'], mmap_mode='r')
            coded = []
            for ids in scene_groups(job['indices'], job['cuts']):
                qps = [30]*(1+(len(ids)-1+7)//8)
                stream, _ = model.encode(frames[ids], qps)
                coded.append((ids, qps, stream))
            video = dict(frames=len(frames), width=frames.shape[2], height=frames.shape[1], fps=24.)
            packet = transport.make(video, job['keys'], job['cuts'], coded, WEIGHTS, max_gap=None)
            Path(job['packet']).write_bytes(packet)
            return dict(packet_sha256=sha(job['packet']), **accounting(len(packet)))
        if job['action'] != 'decode':
            raise ValueError('invalid codec action')
        header, payload = transport.parse(Path(job['packet']).read_bytes(), WEIGHTS)
        decoded, indices = [], []
        for c in header['chunks']:
            decoded.extend(model.decode(payload[c['offset']:c['offset']+c['bytes']],
                                        len(c['indices']), header['video'], c['qps']))
            indices.extend(c['indices'])
        np.save(job['anchors'], np.stack(decoded))
        return dict(indices=indices, video=header['video'], cuts=header['cuts'],
                    packet_sha256=sha(job['packet']), source_pixels_at_receiver=False)
    return run


def channel_worker():
    from .metadata_channel import transmit
    def run(job):
        packet = Path(job['input']).read_bytes()
        received, report = transmit(packet, 10., 42, bits_per_symbol=4)
        if received != packet or report['bit_errors']:
            raise ValueError('channel integrity failure; no oracle replacement permitted')
        Path(job['output']).write_bytes(received)
        return dict(report, received_sha256=sha(job['output']),
                    physical_link_overhead_included=False)
    return run


def receiver_worker():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from . import dense_temporal_receiver as temporal
    from .sparc import configure_numerics
    configure_numerics()
    weights = REPO/'.local/checkpoints/rife/v4.25'
    vendor = REPO/'.local/vendor/Practical-RIFE'
    for p, h in [(weights/'flownet.pkl', temporal.WEIGHT_SHA),
                 (weights/'IFNet_HDv3.py', temporal.NETWORK_SHA),
                 (vendor/'model/warplayer.py', temporal.WARP_SHA)]:
        if sha(p) != h:
            raise ValueError('RIFE identity changed')
    sys.path[:0] = [str(weights), str(vendor)]
    from IFNet_HDv3 import IFNet
    net = IFNet().cuda().eval()
    state = torch.load(weights/'flownet.pkl', map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'):v for k, v in state.items()}
    extras = set(state)-set(net.state_dict())
    if any(not k.startswith(('teacher.', 'caltime.')) for k in extras):
        raise ValueError('unexpected RIFE weights')
    net.load_state_dict({k:v for k,v in state.items() if k not in extras}, strict=True)
    def run(job):
        anchors = np.load(job['anchors'])
        info = job['info']
        v, indices = info['video'], info['indices']
        temporal.validate_indices(indices, v['frames'], None)
        from .sparc_selection import validate_anchors
        validate_anchors(indices, v['frames'], None, info['cuts'])
        h, w = anchors.shape[1:3]
        out = np.empty((v['frames'], h, w, 3), np.uint8)
        def tensor(a):
            x = torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/255
            return F.pad(x, (0, (-w)%64, 0, (-h)%64))
        with torch.inference_mode():
            left = tensor(anchors[0])
            out[0] = anchors[0]
            for k, (a,b) in enumerate(zip(indices, indices[1:])):
                right = tensor(anchors[k+1])
                out[b] = anchors[k+1]
                for i in range(a+1,b):
                    _,_,merged = net(torch.cat((left,right),1), (i-a)/(b-a), [16,8,4,2,1])
                    out[i] = (merged[-1][0,:,:h,:w].permute(1,2,0).clamp(0,1)*255).round().byte().cpu().numpy()
                left = right
        if not np.array_equal(out[indices], anchors):
            raise ValueError('received anchors changed')
        movie = job['movie']
        subprocess.run(['ffmpeg','-v','error','-y','-threads','2','-f','rawvideo','-pix_fmt','rgb24',
                        '-s',f'{w}x{h}','-r','24','-i','-','-an','-c:v','libx264','-crf','18',
                        '-pix_fmt','yuv420p','-threads','2',movie], input=out.tobytes(), check=True)
        return dict(movie_sha256=sha(movie), source_pixels_at_receiver=False,
                    received_anchors_preserved_before_mp4=True, scene_cut_interpolation=False)
    return run


def quality_worker():
    import numpy as np
    import torch
    import cv2
    import lpips
    cv2.setNumThreads(2)
    net = lpips.LPIPS(net='vgg').cuda().eval()
    def run(job):
        originals = np.load(job['frames'], mmap_mode='r')
        h, w = originals.shape[1:3]
        raw = subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',job['movie'],
                                      '-f','rawvideo','-pix_fmt','rgb24','-threads','2','-'])
        restored = np.frombuffer(raw, np.uint8).reshape(-1,h,w,3)
        if restored.shape != originals.shape:
            raise ValueError('quality timeline mismatch')
        mse, perceptual, temporal, region = [], [], [], []
        previous = None
        with torch.inference_mode():
            for a,b in zip(originals,restored):
                error = (a.astype(np.float32)-b)/255
                mse.append(float((error**2).mean()))
                region.append(float(np.quantile(np.abs(error).mean(-1), .95)))
                if previous is not None:
                    temporal.append(float(((error-previous)**2).mean()))
                previous = error
                perceptual.append(float(net(lpips.im2tensor(a.copy()).cuda(),
                                            lpips.im2tensor(b.copy()).cuda()).item()))
        return dict(mse=float(np.mean(mse)), lpips_vgg=float(np.mean(perceptual)),
                    temporal_mse=float(np.mean(temporal)), local_mae95=float(np.mean(region)),
                    psnr_db=float(np.mean(-10*np.log10(np.maximum(mse,1e-12)))),
                    per_frame=dict(mse=mse, lpips_vgg=perceptual, temporal_mse=temporal,
                                   local_mae95=region), final_mp4_crf=18)
    return run


def main():
    import traceback
    role = sys.argv[1]
    functions = dict(semantic=semantic_worker, encoder=codec_worker, decoder=codec_worker,
                     channel=channel_worker, receiver=receiver_worker, quality=quality_worker)
    # Library banners go to the retained stderr log, never the RPC response pipe.
    with contextlib.redirect_stdout(sys.stderr):
        handler = functions[role]()
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = handler(json.loads(line))
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(error=repr(error))), flush=True)
            return
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
