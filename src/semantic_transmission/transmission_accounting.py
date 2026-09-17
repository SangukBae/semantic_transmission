"""Disjoint on-wire byte and LDPC ledgers; no RF-bit claim for continuous JSCC."""
import argparse
import json
from pathlib import Path

from .artifacts import sha256, write_json
from .packets import accounting
from .wire import HEADER, pack, unpack

VERSION = "lgvsc_transport_ledger_v1"


def json_size(value):
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8"))


def packet_breakdown(packet):
    header, payload = unpack(packet)
    if pack(header, payload) != packet:
        raise ValueError("byte attribution requires canonical LGV1 serialization")
    keyframes, segments = header["keyframes"], header["segments"]
    if sum(k["rate_bytes"] for k in keyframes) != len(payload):
        raise ValueError("rate-index bytes do not cover the wire payload")
    parts = {
        "caption_json_values": sum(json_size(s["text"]) for s in segments),
        "flow_json_values": sum(json_size(s["flow"]) for s in segments),
        "normalization_json_values": sum(json_size(k["average_power"]) for k in keyframes),
        "keyframe_index_json_values": sum(json_size(k["index"]) for k in keyframes),
        "rate_and_stream_descriptor_json_values": sum(json_size(v) for k in keyframes
            for name, v in k.items() if name not in {"average_power", "index"}),
        "video_configuration_json": json_size(header["video"]),
        "decoder_configuration_json": json_size(header["decoder"]),
    }
    parts["other_header_json"] = json_size(header) - sum(parts.values())
    parts.update(rate_index_payload=len(payload), framing_without_crc=HEADER.size - 4, crc32=4)
    if min(parts.values()) < 0 or sum(parts.values()) != len(packet):
        raise ValueError("packet byte conservation failed")
    return {"version": VERSION, "packet_bytes": len(packet), "components_bytes": parts,
            "caption_utf8_bytes": sum(len(s["text"].encode("utf-8")) for s in segments),
            "note": "Disjoint serialized components; JSON values include quotes/escapes. "
                    "Other header JSON includes keys, paths, schema and checkpoint identifiers."}


def channel_breakdown(packet, visual_uses, video, channel_report):
    ledger = packet_breakdown(packet)
    expected = accounting(len(packet), k=channel_report["ldpc_k"], n=channel_report["ldpc_n"],
                          bits_per_symbol=channel_report["bits_per_symbol"])
    for key, value in expected.items():
        if channel_report[key] != value:
            raise ValueError(f"recorded digital accounting disagrees: {key}")
    header, _ = unpack(packet)
    if visual_uses != sum(k["complex_count"] for k in header["keyframes"]):
        raise ValueError("visual channel uses disagree with packet descriptors")
    digital = expected["complex_channel_uses"]
    denominator = 3 * video["width"] * video["height"] * video["frames"]
    if denominator <= 0:
        raise ValueError("CBR requires positive source dimensions")
    ledger.update(
        ldpc_bits={"information": expected["information_bits"], "padding": expected["padding_bits"],
                   "parity": expected["coded_bits"] - expected["blocks"] * expected["ldpc_k"],
                   "coded_total": expected["coded_bits"]},
        complex_channel_uses={"visual": visual_uses, "digital": digital, "total": visual_uses + digital},
        cbr={"denominator_source_scalars": denominator, "visual": visual_uses / denominator,
             "digital": digital / denominator, "total": (visual_uses + digital) / denominator},
        physical_link_overhead_included=False, paper_cbr_equivalence_claimed=False,
        attribution_note="Digital components share LDPC blocks; no independent per-field symbol rounding. "
                         "Total CBR retains all metadata, padding and parity. complex64 file bytes are storage.")
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new report path")
    tx = args.run_dir / "transmitter"
    packet = (tx / "metadata.bin").read_bytes()
    size = (tx / "visual.c64").stat().st_size
    if size % 8:
        raise ValueError("invalid complex64 stream size")
    header, _ = unpack(packet)
    report = json.loads((args.run_dir / "channel_accounting.json").read_text())
    result = channel_breakdown(packet, size // 8, header["video"], report)
    result["source_files"] = {n: sha256(tx / n) for n in ("metadata.bin", "visual.c64")}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
