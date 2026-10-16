"""Paid, learned 1D innovation tokens for unified BSU.

Inspired by compact latent/context and region-adaptive coding research, not a
reproduction of GVC1D/UniGVC. DCVC remains frozen. Learned queries aggregate
source-minus-received VAE features; only quantized tokens and a coarse importance
map cross the channel. Temporal prediction is integer differencing within each
window, with actual Golomb-Rice coding (not a learned hyperprior).
"""
import math
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POLICY = 'bsu_innovation_tokens_v1'
STEPS = (.125, .25, .5, 1.)
HEADER = struct.Struct('!4sBBBBHHI')


class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values.round()

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def importance(target, prior):
    """Sender motion/error heuristic; not a semantic hand detector."""
    if target.shape != prior.shape or target.ndim != 5 or target.shape[1] != 4:
        raise ValueError('invalid source/received latent pair')
    error = (target-prior).abs().mean(1, keepdim=True)
    motion = torch.zeros_like(error)
    motion[:, :, 1:] = (target[:, :, 1:]-target[:, :, :-1]).abs().mean(1, keepdim=True)
    score = F.adaptive_avg_pool3d(error+motion, (1, 5, 9)).detach()
    flat = score.flatten(1)
    order = torch.argsort(flat, dim=1, descending=True, stable=True)
    mask = torch.zeros_like(flat).scatter_(1, order[:, :9], 1.)
    return mask.reshape(-1, 1, 1, 5, 9)


def expanded_roi(roi, prior):
    return F.interpolate(roi.to(prior), size=prior.shape[2:], mode='nearest')


def coordinates(prior):
    h, w = prior.shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=prior.device, dtype=prior.dtype),
                            torch.linspace(-1, 1, w, device=prior.device, dtype=prior.dtype), indexing='ij')
    return torch.stack((yy, xx))[None, :, None].expand(prior.shape[0], -1, prior.shape[2], -1, -1)


def rice_parameters(delta):
    values = np.asarray(delta, dtype=np.int64)
    unsigned = np.where(values >= 0, values*2, -values*2-1)
    return [min(range(8), key=lambda k: int(((unsigned[..., c] >> k)+1+k).sum()))
            for c in range(values.shape[-1])]


def encode_symbols(symbols, roi, step_index):
    """Exact integer prediction and prefix coding, including paid ROI/parameters."""
    q = np.asarray(symbols)
    mask = np.asarray(roi)
    if (q.ndim != 3 or q.shape[0] != 5 or not 1 <= q.shape[1] <= 64 or q.shape[2] != 8
            or q.dtype.kind not in 'iu' or np.abs(q.astype(np.int64)).max() > 512
            or mask.shape != (5, 9) or not np.isin(mask, [0, 1]).all()
            or type(step_index) is not int or not 0 <= step_index < len(STEPS)):
        raise ValueError('invalid innovation symbols/ROI')
    q = q.astype(np.int64)
    delta = np.diff(q, axis=0, prepend=np.zeros_like(q[:1]))
    ks = rice_parameters(delta); bits = []
    for row in delta.reshape(-1, 8):
        for value, k in zip(row, ks):
            value = int(value); unsigned = 2*value if value >= 0 else -2*value-1
            bits.extend([0]*(unsigned >> k)); bits.append(1)
            bits.extend((unsigned >> shift) & 1 for shift in range(k-1, -1, -1))
    body = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
    roi_bytes = np.packbits(mask.astype(np.uint8).reshape(-1)).tobytes()
    return HEADER.pack(b'UIT1', step_index, q.shape[0], q.shape[1], q.shape[2],
                       len(roi_bytes), len(ks), len(bits))+bytes(ks)+roi_bytes+body


def decode_symbols(data):
    if len(data) < HEADER.size:
        raise ValueError('truncated token header')
    magic, step, t, n, c, nr, nk, nb = HEADER.unpack(data[:HEADER.size])
    if (magic != b'UIT1' or step >= len(STEPS) or t != 5 or not 1 <= n <= 64
            or c != 8 or nr != 6 or nk != 8 or not 0 < nb <= 1_000_000
            or len(data) != HEADER.size+nk+nr+(nb+7)//8):
        raise ValueError('invalid token dimensions/framing')
    ks = list(data[HEADER.size:HEADER.size+nk])
    if max(ks) > 7:
        raise ValueError('invalid Rice parameter')
    start = HEADER.size+nk
    mask = np.unpackbits(np.frombuffer(data[start:start+nr], dtype=np.uint8))
    bits = np.unpackbits(np.frombuffer(data[start+nr:], dtype=np.uint8))
    if mask[45:].any() or bits[nb:].any():
        raise ValueError('nonzero token padding')
    offset = 0; values = []
    for i in range(t*n*c):
        quotient = 0
        while offset < nb and bits[offset] == 0:
            quotient += 1; offset += 1
            if quotient > 2048:
                raise ValueError('invalid token magnitude')
        k = ks[i % c]
        if offset >= nb or offset+1+k > nb:
            raise ValueError('truncated Rice symbol')
        offset += 1; remainder = 0
        for bit in bits[offset:offset+k]:
            remainder = (remainder << 1) | int(bit)
        offset += k; unsigned = (quotient << k)+remainder
        values.append(unsigned//2 if unsigned % 2 == 0 else -(unsigned+1)//2)
    if offset != nb:
        raise ValueError('unconsumed token bits')
    q = np.asarray(values, dtype=np.int64).reshape(t, n, c).cumsum(axis=0)
    if np.abs(q).max() > 512:
        raise ValueError('token magnitude exceeds bound')
    return q.astype(np.int16), mask[:45].reshape(5, 9), step


class InnovationCodec(nn.Module):
    def __init__(self, tokens=16, width=32):
        super().__init__()
        if not 1 <= tokens <= 64 or width < 8:
            raise ValueError('invalid token architecture')
        self.tokens, self.width = tokens, width
        self.analysis = nn.Sequential(nn.Conv3d(11, width, 3, padding=1), nn.SiLU(),
                                      nn.Conv3d(width, width, 3, padding=1), nn.SiLU())
        self.queries = nn.Parameter(torch.randn(tokens, width)*.1)
        self.to_token = nn.Linear(width, 8)
        self.spatial_query = nn.Conv3d(7, width, 1)
        self.token_key = nn.Linear(8, width)
        self.token_value = nn.Linear(8, width)
        self.synthesis = nn.Sequential(nn.Conv3d(width+7, width, 3, padding=1), nn.SiLU(),
                                       nn.Conv3d(width, 4, 3, padding=1))
        nn.init.normal_(self.synthesis[-1].weight, std=.001)
        nn.init.zeros_(self.synthesis[-1].bias)

    def encode(self, target, prior, roi):
        mask = expanded_roi(roi, prior)
        x = torch.cat((target-prior, prior, mask, coordinates(prior)), 1)
        features = F.avg_pool3d(self.analysis(x), (1, 2, 2))
        features = features.permute(0, 2, 3, 4, 1).flatten(2, 3)
        logits = torch.einsum('nd,btpd->btnp', self.queries, features)/math.sqrt(self.width)
        weights = F.avg_pool3d(mask, (1, 2, 2)).flatten(3).squeeze(1)
        logits = logits+torch.log1p(3*weights[:, :, None])
        pooled = torch.einsum('btnp,btpd->btnd', logits.softmax(-1), features)
        return 8*torch.tanh(self.to_token(pooled)/8)

    def decode(self, tokens, prior, roi):
        side = torch.cat((prior, expanded_roi(roi, prior), coordinates(prior)), 1)
        query = F.avg_pool3d(self.spatial_query(side), (1, 2, 2))
        b, _, t, h, w = query.shape
        query = query.permute(0, 2, 3, 4, 1).flatten(2, 3)
        keys = self.token_key(tokens)+self.queries[None, None]
        attention = torch.einsum('btpd,btnd->btpn', query, keys)/math.sqrt(self.width)
        features = torch.einsum('btpn,btnd->btpd', attention.softmax(-1), self.token_value(tokens))
        features = features.reshape(b, t, h, w, self.width).permute(0, 4, 1, 2, 3)
        features = F.interpolate(features, size=prior.shape[2:], mode='trilinear', align_corners=False)
        return prior+.5*torch.tanh(self.synthesis(torch.cat((features, side), 1)))

    def forward(self, target, prior, step_index=1):
        roi = importance(target, prior)
        tokens = self.encode(target, prior, roi); scaled = tokens/STEPS[step_index]
        symbols = _Quantize.apply(scaled)
        decoded = self.decode(symbols*STEPS[step_index], prior, roi)
        delta = torch.diff(symbols, dim=1, prepend=torch.zeros_like(symbols[:, :1]))
        # Differentiable approximation to actual Rice lengths, NOT measured rate.
        ks = rice_parameters(delta.detach().cpu().numpy().reshape(-1, self.tokens, 8))
        powers = torch.tensor([2**k for k in ks], device=prior.device, dtype=prior.dtype)
        rate = (2*delta.abs()/powers+1+torch.tensor(ks, device=prior.device)).sum()
        rate = rate+(HEADER.size+8+6)*8
        weights = 1+3*expanded_roi(roi, prior)
        distortion = ((decoded-target).square()*weights).mean()
        temporal = F.l1_loss(torch.diff(decoded, dim=2), torch.diff(target, dim=2))
        return decoded, dict(rate_bits_proxy=rate, distortion=distortion, temporal=temporal), roi

    @torch.no_grad()
    def compress(self, target, prior, step_index=1):
        if target.shape[0] != 1:
            raise ValueError('one video window per token packet')
        roi = importance(target, prior)
        q = (self.encode(target, prior, roi)/STEPS[step_index]).round().to(torch.int16)
        return encode_symbols(q[0].cpu().numpy(), roi[0, 0, 0].cpu().numpy(), step_index)

    def decompress(self, data, prior):
        q, mask, step = decode_symbols(data)
        if q.shape[1] != self.tokens or prior.shape[0] != 1 or prior.shape[2] != 5:
            raise ValueError('token checkpoint/timeline mismatch')
        tokens = torch.from_numpy(q.copy()).to(prior)[None]*STEPS[step]
        roi = torch.from_numpy(mask.copy()).to(prior)[None, None, None]
        return self.decode(tokens, prior, roi), roi
