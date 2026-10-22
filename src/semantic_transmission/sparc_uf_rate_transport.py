"""Paid per-chunk QP schedules with scene boundaries and strict receiver checks."""
import zlib

from . import sparc_uf_transport as old
from .bsu_predict import validate_video
from .dcvc_uf_codec import qp_schedule
from .sparc_selection import scene_groups, validate_anchors
from .sparc_transport import temporal_identity
from .wire import pack, unpack

SCHEMA = 'sparc_uf_chunk_qp_v1'


def descriptors(indices, cuts):
    rows = []
    for scene, group in enumerate(scene_groups(indices, cuts)):
        for pos in [0] + list(range(1, len(group), 8)):
            ids = group[pos:pos+(1 if pos == 0 else 8)]
            rows.append(dict(scene=scene, position=pos, indices=ids, intra=pos == 0))
    return rows


def make(video, keys, cuts, coded, weights, max_gap=4):
    payload, chunks = bytearray(), []
    for indices, qps, stream in coded:
        qp_schedule(len(indices), qps)
        chunks.append(dict(indices=indices, qps=qps, offset=len(payload), bytes=len(stream),
                           padded_frames=old.padding(len(indices))))
        payload.extend(stream)
    header = dict(schema=SCHEMA, upstream=old.UPSTREAM, weights=weights, model='hts',
                  video=video, keys=keys, cuts=cuts, max_gap=max_gap,
                  temporal=temporal_identity(max_gap), chunks=chunks)
    original = pack(header, payload)
    prefix = original[:-len(payload)]
    body = zlib.compress(prefix, 9)
    envelope = old.ENVELOPE.pack(b'SUQP',len(original),len(prefix),len(body))+body+payload
    packet = envelope if len(envelope) < len(original) else original
    parse(packet, weights)
    return packet


def parse(packet, weights):
    if not isinstance(packet, bytes) or not 16 < len(packet) <= old.LIMIT:
        raise ValueError('invalid chunk-QP packet')
    prefix_size = None
    if packet[:4] == b'SUQP':
        _, total, prefix_size, body_size = old.ENVELOPE.unpack(packet[:16])
        if (not 16 < prefix_size <= old.PREFIX_LIMIT or not prefix_size < total <= old.LIMIT or
                body_size < 1 or len(packet) != 16+body_size+total-prefix_size):
            raise ValueError('invalid chunk-QP envelope')
        inflater = zlib.decompressobj()
        try:
            prefix = inflater.decompress(packet[16:16+body_size],prefix_size+1)
        except zlib.error as error:
            raise ValueError('invalid chunk-QP compressed header') from error
        if (len(prefix) != prefix_size or not inflater.eof or inflater.unused_data or
                inflater.unconsumed_tail):
            raise ValueError('invalid chunk-QP metadata boundary')
        packet = prefix + packet[16+body_size:]
    elif packet[:4] != b'LGV1':
        raise ValueError('unknown chunk-QP envelope')
    header, payload = unpack(packet)
    fields = {'schema','upstream','weights','model','video','keys','cuts','max_gap','temporal','chunks'}
    if (set(header) != fields or header['schema'] != SCHEMA or
            header['upstream'] != old.UPSTREAM or header['weights'] != weights or
            header['model'] != 'hts' or
            header['temporal'] != temporal_identity(header['max_gap'])):
        raise ValueError('unsupported chunk-QP model')
    validate_video(header['video'])
    if not isinstance(header['chunks'],list) or not header['chunks']:
        raise ValueError('missing scenes')
    offset, indices, groups = 0, [], []
    for chunk in header['chunks']:
        if (not isinstance(chunk,dict) or set(chunk) != {'indices','qps','offset','bytes','padded_frames'} or
                type(chunk['offset']) is not int or chunk['offset'] != offset or
                type(chunk['bytes']) is not int or chunk['bytes'] < 1 or
                not isinstance(chunk['indices'],list) or not chunk['indices'] or
                type(chunk['padded_frames']) is not int or
                chunk['padded_frames'] != old.padding(len(chunk['indices']))):
            raise ValueError('invalid scene descriptor')
        qp_schedule(len(chunk['indices']),chunk['qps'])
        indices.extend(chunk['indices']); groups.append(chunk['indices']); offset += chunk['bytes']
    validate_anchors(indices,header['video']['frames'],header['max_gap'],header['cuts'])
    keys = header['keys']
    if (not isinstance(keys,list) or not keys or keys != sorted(set(keys)) or
            any(type(k) is not int for k in keys) or not set(keys) <= set(indices) or
            keys[0] != 0 or keys[-1] != header['video']['frames']-1):
        raise ValueError('invalid semantic keys')
    if offset != len(payload) or groups != scene_groups(indices,header['cuts']):
        raise ValueError('scene boundaries or payload changed')
    actual = len(packet)-len(payload)
    if actual > old.PREFIX_LIMIT or prefix_size is not None and actual != prefix_size:
        raise ValueError('invalid prefix boundary')
    return header, payload
