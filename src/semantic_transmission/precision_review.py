"""Repair only precision-run reporting while retaining frozen inference hashes.

The executed precision_reconstruction module is part of both the execution and
noise contracts. Keep those bytes intact so completed inference can be reused
and future BF16 runs can validate the FP32 noise trace. The public launcher uses
this report adapter; its hash is recorded in RESULT.json, a stage artifact.
"""
import math
from numbers import Real
from pathlib import Path

from . import precision_reconstruction as runner
from .artifacts import sha256, write_json

METRICS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")


def metric_rows(before, after):
    """Render measured scores only; frames and shape are metadata."""
    rows = []
    for key in METRICS:
        values = before[key], after[key]
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
            raise ValueError(f"invalid numeric quality metric: {key}")
        rows.append(f'<tr><td>{key}</td><td>{values[0]:.4f}</td><td>{values[1]:.4f}</td></tr>')
    return ''.join(rows)


def comparison(output):
    run = output / "run"
    policy = runner.read_json(run / "receiver_policy.json")
    previous = Path(policy["noise_reference"]).parents[2] if policy["noise_reference"] else runner.PREVIOUS
    paired = runner.caption_revision.verify_pair(previous, output)
    if runner.snapshot(previous / "run", runner.tail.INPUTS) != runner.snapshot(run, runner.tail.INPUTS):
        raise ValueError("received inputs differ between comparisons")
    controlled = bool(policy["noise_reference"])
    if controlled:
        a, b = [runner.read_json(root / "run" / runner.text.REPORT) for root in (previous, output)]
        unchanged = lambda report: {k: v for k, v in report["contract"].items() if k not in {"compute_dtype", "device"}}
        if unchanged(a) != unchanged(b) or a["prompts"] != b["prompts"]:
            raise ValueError("paired T5 model, prompt, or non-precision configuration changed")
    paired.update(diffusion_noise_tensor_identity_verified=controlled, vae_noise_identity_verified=controlled,
        scope="Same tail17 and verified VAE/initial/RFLOW noise; T5 CPU FP32 vs GPU BF16 changes compute device as well."
        if controlled else "Historical comparison only: legacy noise was not recorded. No T5-only causal claim.")
    quality = [runner.read_json(root / "run/quality.json") for root in (previous, output)]
    rows = metric_rows(quality[0]["delivered_mp4"], quality[1]["delivered_mp4"])
    videos = [run / "data/normalized.mp4", previous / "run/receiver/reconstruction/sample_0000.mp4",
              run / "receiver/reconstruction/sample_0000.mp4"]
    if any(q["status"] != "PASSED" or q["source_sha256"] != sha256(videos[0])
           or q["video_sha256"] != sha256(video) for q, video in zip(quality, videos[1:])):
        raise ValueError("quality artifacts do not match comparison videos")
    target = output / "comparison.mp4"
    runner.subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *[x for v in videos for x in ("-i", str(v))],
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-an", "-c:v", "libx264",
        "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target)], check=True)
    cfg = runner.read_json(run / "run_config.json")
    runner.tail.hybrid.pts_audit(target, cfg["frames"], cfg["fps"])
    report = runner.read_json(run / runner.text.REPORT)
    result = dict(status="PASS_T5_PRECISION_RECONSTRUCTION_REVIEW_PENDING", pairing=paired,
        precision=policy["t5_precision"], t5_device=report["contract"]["device"], reference_policy=runner.tail.POLICY,
        frames=cfg["frames"], fps=cfg["fps"], previous=str(previous), quality_before=quality[0]["delivered_mp4"],
        quality_after=quality[1]["delivered_mp4"], video_sha256=sha256(videos[2]), comparison_sha256=sha256(target),
        noise_trace_sha256=sha256(run / runner.noise.TRACE), noise_reference_verified=controlled,
        channel_uses=runner.read_json(run / "channel_accounting.json")["total_complex_channel_uses"],
        additional_channel_uses=0, hallucination_review="PENDING", hallucination_mitigation_verified=False,
        report_builder=dict(path="src/semantic_transmission/precision_review.py", sha256=sha256(Path(__file__))))
    note = "두 실행의 VAE·초기 생성·반복 생성 잡음 해시 일치를 확인했습니다." if controlled else \
        "이전 결과에는 잡음 기록이 없어 잡음이 같은 비교가 아닙니다. 이 결과만으로 T5의 영향을 확정할 수 없습니다."
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>T5 정밀도 복원</title><style>body{font:16px system-ui;margin:24px}video{width:100%}'
        'td,th{padding:10px}</style><h1>원본 / 비교 기준 / 이번 복원</h1>'
        '<video controls preload="metadata" src="comparison.mp4"></video>'
        f'<p>T5 {policy["t5_precision"].upper()} · 마지막 17프레임 참조 · 생성 {policy["noise_contract"]["steps"]}단계.</p><p>{note}</p>'
        '<p>FP32는 CPU, BF16은 GPU에서 사전 계산합니다. 영상 생성기는 기존 BF16을 유지합니다.</p>'
        '<table><tr><th>지표</th><th>비교 기준</th><th>이번 복원</th></tr>'+rows+
        '</table><p>의미 오류 검수·할루시네이션 완화 입증은 미완료입니다.</p><a href="RESULT.json">검증 기록</a></html>')
    write_json(output / "RESULT.json", result)
    (output / "review.html").write_text(html)


def main(argv=None):
    original = runner.comparison
    runner.comparison = comparison
    try:
        return runner.main(argv)
    finally:
        runner.comparison = original


if __name__ == "__main__":
    main()
