"""Paired keyframe ablations: retain existing conditions and account for additions."""
import bisect
import math

from .artifacts import sha256, write_json
from .webvid5 import read_json
from .wire import pack, unpack


def densify(indices, gap):
    if gap < 1 or len(indices) < 2 or indices[0] != 0 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("invalid keyframe endpoints/gap")
    result = [0]
    for a, b in zip(indices, indices[1:]):
        count = math.ceil((b - a) / gap)
        result.extend(a + round((b - a) * i / count) for i in range(1, count + 1))
    return result


def subdivide_metadata(indices, dense, rows):
    if len(rows) != len(indices) - 1 or not set(indices).issubset(dense):
        raise ValueError("metadata must cover the original intervals")
    result = []
    for i, (a, b) in enumerate(zip(dense, dense[1:])):
        segment = bisect.bisect_right(indices, a) - 1
        if segment < 0 or segment >= len(rows) or b > indices[segment + 1]:
            raise ValueError("new interval crosses an original semantic boundary")
        result.append(dict(rows[segment], path=f"clips/sample/{i:05d}.mp4"))
    return result


def preserve_transport(run, baseline, *, received=False):
    """Replay baseline symbol blocks, with separately seeded AWGN on added keys.

    Digital metadata is still transmitted in full by the channel worker. Its
    accounting remains valid because replacing visual values changes no lengths.
    """
    import numpy as np
    from .transmission_accounting import packet_breakdown
    header, payload = unpack((run / "transmitter/metadata.bin").read_bytes())
    old, old_payload = unpack((baseline / "transmitter/metadata.bin").read_bytes())
    old_items = {item["index"]: item for item in old["keyframes"]}
    if not set(old_items).issubset(item["index"] for item in header["keyframes"]):
        raise ValueError("dense ablation dropped original keyframes")
    values = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    if received:
        original = np.fromfile(baseline / "received/visual.c64", dtype="<c8")
        cfg = read_json(run / "run_config.json")
        sigma = math.sqrt(1 / (2 * 10 ** (cfg["snr_db"] / 10)))
        seed = cfg.get("channel_seed", cfg["seed"]) + 100000
        for item in header["keyframes"]:
            start, count = item["complex_offset"], item["complex_count"]
            if item["index"] in old_items:
                previous = old_items[item["index"]]
                if count != previous["complex_count"]:
                    raise ValueError("paired symbol block length changed")
                s = previous["complex_offset"]
                values[start:start + count] = original[s:s + count]
            else:
                rng = np.random.default_rng(seed + item["index"])
                noise = (rng.normal(0, sigma, count) + 1j * rng.normal(0, sigma, count)).astype("<c8")
                values[start:start + count] += noise
        values.tofile(run / "received/visual.c64")
        report = read_json(run / "channel_accounting.json")
        report["received_files"]["visual.c64"]["sha256"] = sha256(run / "received/visual.c64")
        report["visual_awgn_rng"] = f"replayed baseline blocks; added keys numpy_PCG64 seed={seed}+frame_index"
        write_json(run / "channel_accounting.json", report)
        return
    original = np.fromfile(baseline / "transmitter/visual.c64", dtype="<c8")
    new_payload, chunks, items = bytearray(), [], []
    offset = 0
    for current in header["keyframes"]:
        paired = current["index"] in old_items
        item = dict(old_items[current["index"]] if paired else current)
        stream, rates = (original, old_payload) if paired else (values, payload)
        s, n, r, rn = (item[k] for k in ("complex_offset", "complex_count", "rate_offset", "rate_bytes"))
        chunks.append(stream[s:s + n])
        item.update(complex_offset=offset, rate_offset=len(new_payload))
        new_payload.extend(rates[r:r + rn])
        items.append(item)
        offset += n
    header["keyframes"] = items
    packet = pack(header, new_payload)
    (run / "transmitter/metadata.bin").write_bytes(packet)
    np.concatenate(chunks).tofile(run / "transmitter/visual.c64")
    report = read_json(run / "sender_accounting.json")
    report.update(visual_complex_channel_uses=offset, visual_real_values=offset * 2,
                  visual_fp32_iq_serialization_bytes=offset * 8, metadata_packet_bytes=len(packet),
                  packed_rate_index_bytes=len(new_payload), header_json_and_framing_bytes=len(packet) - len(new_payload),
                  serialized_model_input_bytes=offset * 8 + len(packet), metadata_breakdown=packet_breakdown(packet),
                  original_keyframe_transport_replayed=True)
    report["transmitter_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                   for p in (run / "transmitter").iterdir()}
    write_json(run / "sender_accounting.json", report)
