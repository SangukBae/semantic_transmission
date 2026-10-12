"""Run the frozen BSU video pipeline with an explicit benchmark digital link.

Only the metadata modulation changes: QPSK below 10 dB, 16-QAM otherwise.
The historical runner and protocols remain byte-identical. This adapter is
included in every new benchmark protocol's code hashes and worker commands.
"""
from functools import partial

from . import qwen35_pairs_video as current
from . import qwen35_pairs_comparison as comparison
from . import metadata_channel
from .bsu_sequence_refinement import read

MODULE = "semantic_transmission.bsu_webvid_base"


def digital_bits(snr_db):
    return 2 if snr_db < 10 else 4


def paired_channel(run):
    config = read(run / "run_config.json")
    bits = config["benchmark_digital_bits_per_symbol"]
    if bits != digital_bits(config["snr_db"]):
        raise ValueError("digital modulation differs from the declared SNR policy")
    original = metadata_channel.transmit
    # comparison.paired_channel imports the transport at call time. Injection
    # keeps its frame-seeded visual noise and complete accounting unchanged.
    metadata_channel.transmit = partial(original, bits_per_symbol=bits)
    try:
        return comparison.paired_channel(run)
    finally:
        metadata_channel.transmit = original


def main():
    original_worker = current.worker
    def worker(root, stage):
        if stage == "channel":
            return paired_channel(root / "reconstruction/event_pairs/run")
        return original_worker(root, stage)
    current.worker = worker
    current.MODULE = MODULE  # Extraction and subsequent workers use this adapter.
    current.main()


if __name__ == "__main__":
    main()
