"""Actual Sionna LDPC + QAM + AWGN transport for semantic packets."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

from .artifacts import write_json
from .packets import accounting, decode, encode


def transmit(packet, snr_db, seed, *, k=6144, n=9216, bits_per_symbol=4):
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    # A separate CPU environment avoids competing with the video model for VRAM.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf
    import sionna
    from sionna.channel import AWGN
    from sionna.fec.ldpc.encoding import LDPC5GEncoder
    from sionna.fec.ldpc.decoding import LDPC5GDecoder
    from sionna.mapping import Constellation, Mapper, Demapper

    sionna.config.seed = seed
    budget = accounting(len(packet), k=k, n=n, bits_per_symbol=bits_per_symbol)
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    padded = np.pad(bits, (0, budget["padding_bits"])).reshape(-1, k)
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, num_iter=20, return_infobits=True)
    constellation = Constellation("qam", num_bits_per_symbol=bits_per_symbol)
    mapper, demapper = Mapper(constellation=constellation), Demapper("app", constellation=constellation)
    noise = tf.constant(10.0 ** (-snr_db / 10.0), tf.float32)  # Es/N0, E|x|^2 = 1
    restored = []
    # Bound memory independently of packet length.
    for start in range(0, len(padded), 32):
        coded = encoder(tf.convert_to_tensor(padded[start:start + 32], dtype=tf.float32))
        symbols = mapper(coded)
        received = AWGN()([symbols, noise])
        restored.append(decoder(demapper([received, noise])).numpy().astype(np.uint8))
    recovered = np.concatenate(restored).reshape(-1)[:len(bits)]
    budget.update({"snr_db": snr_db, "snr_definition": "Es/N0 per complex symbol",
                   "seed": seed, "bit_errors": int(np.count_nonzero(bits != recovered)),
                   "backend": "sionna_ldpc5g_qam_awgn"})
    return np.packbits(recovered).tobytes(), budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON segment rows")
    parser.add_argument("--output", type=Path, required=True, help="receiver CSV")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("output/report already exists; use a new run directory")
    packet = encode(json.loads(args.input.read_text()))
    received, report = transmit(packet, args.snr, args.seed)
    try:
        rows = decode(received)
    except (ValueError, UnicodeError) as error:
        report.update({"status": "FAILED", "error": str(error)})
        write_json(args.report, report)
        raise SystemExit("Metadata channel failed integrity validation; decoder input was not written")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
        writer.writeheader()
        writer.writerows(rows)
    report["status"] = "PASSED"
    write_json(args.report, report)


if __name__ == "__main__":
    main()
