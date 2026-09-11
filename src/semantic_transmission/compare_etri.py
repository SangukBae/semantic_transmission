"""Compare completed LGVSC runs with preserved SGD-JSCC outputs at common boundaries."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from .artifacts import sha256, write_json
from .packets import accounting
from .research_quality import Metrics, read_frames, read_video


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    manifest = json.loads((root / "batch_manifest.json").read_text())
    if manifest["status"] != "PASSED" or manifest["completed_videos"] != 10:
        raise ValueError("comparison requires all ten successful LGVSC reconstructions")
    inventory = json.loads(args.baseline_inventory.read_text())
    if not inventory["source_video_pixels_match"]:
        raise ValueError("historical source frames were not verified")
    output.mkdir(parents=True, exist_ok=False)
    metrics = Metrics()
    rows = []
    inputs = {source["id"]: source for source in manifest["inputs"]}
    for name, source in inputs.items():
        run = root / name
        quality = json.loads((run / "quality.json").read_text())
        sender = json.loads((run / "sender_accounting.json").read_text())
        channel = json.loads((run / "channel_accounting.json").read_text())
        if sha256(source["path"]) != source["sha256"] or quality["source_sha256"] != source["sha256"]:
            raise ValueError("LGVSC source provenance changed")
        for file, info in sender["transmitter_files"].items():
            if sha256(run / "transmitter" / file) != info["sha256"]:
                raise ValueError("transmitter payload changed after accounting")
        record = next(r for r in manifest["runs"] if r["id"] == name)
        row = {"video": name, "model": "LGVSC_SKEM_DSA_50",
               "digital_packet_bytes": sender["metadata_packet_bytes"],
               "continuous_complex_symbols": sender["visual_complex_channel_uses"],
               "float32_iq_serialization_bytes": sender["visual_fp32_iq_serialization_bytes"],
               "serialized_model_input_bytes": sender["serialized_model_input_bytes"],
               "measured_simulated_channel_uses": channel["total_complex_channel_uses"],
               "common_phy_equivalent_uses": channel["total_complex_channel_uses"],
               "common_phy_basis": "measured_NTSCC_plus_LDPC_16QAM_AWGN",
               "keyframes_or_transmitted_frame_packets": len(json.loads((run / "keyframes.json").read_text())["indices"]),
               "elapsed_seconds_historical_hardware": sum(s["seconds"] for s in record["stages"]),
               "hardware": "RTX4080_plus_i7_14700F",
               "source_sha256": source["sha256"]}
        for boundary in ("lossless_frames", "delivered_mp4"):
            row.update({boundary + "_" + key: quality[boundary][key] for key in ("psnr_db", "ssim", "lpips_alex")})
        rows.append(row)
        original = read_video(source["path"])
        for baseline in [b for b in inventory["baselines"] if b["video"] == name]:
            if baseline["source_sha256"] != source["sha256"]:
                raise ValueError("SGD-JSCC and LGVSC source hash mismatch")
            actual_bytes = 0
            for file in baseline["packet_files"]:
                path = Path(file["path"])
                if sha256(path) != file["sha256"] or path.stat().st_size != file["bytes"]:
                    raise ValueError("historical SGD-JSCC packet changed")
                actual_bytes += path.stat().st_size
            if actual_bytes != baseline["verified_bundle_bytes"]:
                raise ValueError("SGD-JSCC byte accounting mismatch")
            label = f"SGD_{baseline['decoder_policy']}_{baseline['guide_profile']}"
            entry = {"video": name, "model": label, "digital_packet_bytes": actual_bytes,
               "continuous_complex_symbols": 0, "float32_iq_serialization_bytes": 0,
               "serialized_model_input_bytes": actual_bytes, "measured_simulated_channel_uses": None,
               "common_phy_equivalent_uses": accounting(actual_bytes)["complex_channel_uses"],
               "common_phy_basis": "counterfactual_video_packed_LDPC_6144_9216_16QAM_no_channel_replay",
               "keyframes_or_transmitted_frame_packets": baseline["bundle_count"],
               "elapsed_seconds_historical_hardware": float(baseline["historical_metrics"]["total_elapsed_s"]),
               "hardware": "historical_RTX4090", "source_sha256": source["sha256"]}
            for boundary, values in (("lossless_frames", read_frames(baseline["frames_directory"])),
                                     ("delivered_mp4", read_video(baseline["video_path"]))):
                summary, frame_rows = metrics.evaluate(original, values)
                entry.update({boundary + "_" + key: summary[key] for key in ("psnr_db", "ssim", "lpips_alex")})
                csv_write(output / f"{name}_{label}_{boundary}.csv", frame_rows)
            rows.append(entry)
            print(f"Compared {name}: {label}", flush=True)
        csv_write(output / "per_video.csv", rows)
    aggregate = []
    quantities = ("digital_packet_bytes", "continuous_complex_symbols", "serialized_model_input_bytes",
                  "common_phy_equivalent_uses", "lossless_frames_psnr_db", "lossless_frames_ssim",
                  "lossless_frames_lpips_alex", "delivered_mp4_psnr_db", "delivered_mp4_ssim",
                  "delivered_mp4_lpips_alex")
    for model in dict.fromkeys(r["model"] for r in rows):
        subset = [r for r in rows if r["model"] == model]
        if len(subset) != 10:
            raise ValueError("incomplete matched-video comparison")
        aggregate.append({"model": model, "videos": 10,
                          **{key: statistics.mean(r[key] for r in subset) for key in quantities}})
    csv_write(output / "aggregate.csv", aggregate)
    write_json(output / "comparison.json", {"status": "PASSED", "videos": 10,
        "source_identity_verified": True, "metrics_recomputed_with_common_evaluator": True,
        "baseline_inventory_sha256": sha256(args.baseline_inventory),
        "lgvsc_manifest_sha256": sha256(root / "batch_manifest.json"), "aggregate": aggregate,
        "scope": "historical_development_set_comparison_not_matched_rate_or_multiseed",
        "physical_comparison": "LGVSC simulated symbols measured; SGD radio coding is explicitly counterfactual",
        "serialization_note": "complex64 file bytes describe I/Q replay storage, not an RF bitstream"})
    lines = ["# ETRI 10영상 LGVSC–SGD-JSCC 비교", "", "모든 수치는 영상 10개의 평균입니다. 원본은 512×256, 10fps, 100프레임이며 원본 SHA-256과 픽셀 동일성을 확인했습니다.", "",
        "| 모델 | 디지털 패킷 B | 연속 복소 심벌 | 전체 입력 직렬화 B | PSNR | SSIM | LPIPS Alex |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for a in aggregate:
        lines.append(f"| {a['model']} | {a['digital_packet_bytes']:,.1f} | {a['continuous_complex_symbols']:,.1f} | {a['serialized_model_input_bytes']:,.1f} | {a['lossless_frames_psnr_db']:.3f} | {a['lossless_frames_ssim']:.4f} | {a['lossless_frames_lpips_alex']:.4f} |")
    lines += ["", "화질은 MP4 압축 전 복원 PNG를 동일 평가기로 측정했습니다. 최종 MP4의 별도 화질은 CSV에 있습니다.", "",
        "LGVSC는 NTSCC 연속 신호와 디지털 메타데이터를 함께 보냅니다. 전체 입력 직렬화 크기는 실제 송수신 프로세스 사이에서 사용하는 complex64 I/Q 파일과 메타데이터 파일의 합이며, RF 전송 bit 수가 아닙니다. SGD-JSCC는 기존 .sgbundle 디지털 패킷 파일을 직접 합산했습니다.", "",
        "공통 채널 사용량 환산은 SGD-JSCC의 영상별 패킷을 합쳐 LDPC(6144,9216), 16QAM으로 보낸다고 가정한 값입니다. 기존 SGD 결과의 실제 물리 채널 실측값이 아니므로 파일 byte 절감률과 혼동하면 안 됩니다.", "",
        "동일 원본의 기존 결과 비교이며 동일 전송률·다중 seed의 통제 실험이 아닙니다. 기존 실행 시간은 RTX 4090, 새 실행은 RTX 4080이므로 속도 우열을 주장하지 않습니다. 공개 NTSCC 10dB 체크포인트를 사용했고 원 논문의 저자 학습 가중치·데이터셋과 일치하는 성능 재현은 아닙니다."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
