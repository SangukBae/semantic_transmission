"""Freeze authored v2 captions and publish the ready-to-run input index only."""
import html
import os
from pathlib import Path
import re

from semantic_transmission import assisted_captions as captions
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.hybrid_selection import verify_selection
from semantic_transmission.short_video_batch import NAMES, REPO, ROOT, paths, preflight
from semantic_transmission.webvid5 import read_json


def main():
    rows = []
    for name in NAMES:
        root, bundle, output = paths(name)
        authored = REPO / f"configs/captions/short3_{name}_faithful_v2.json"
        texts = read_json(authored)
        for text in texts.values():
            if not text.strip() or len(text.split()) > 80 or len(re.findall(r"[.!?](?:\s|$)", text)) > 6:
                raise ValueError(f"{name}: caption exceeds the v2 limits")
        captions.freeze(root, authored)
        page = root / "assistant_captions/review.html"
        page.write_text(page.read_text().replace("혼합 키프레임 78구간 · AI 작성 캡션", f"{name} · {len(texts)}구간 · 캡션 v2"))
        _, selection = verify_selection(root)
        report = preflight(name)[-1]
        if output.exists():
            raise ValueError("preparation must not mark an already started reconstruction as unstarted")
        report.update(authored_file=str(authored), authored_sha256=sha256(authored),
            selection_sha256=sha256(root / "selection.json"), caption_bundle_sha256=sha256(bundle),
            skem_comparisons=selection["skem_comparisons"], skem_accepted=selection["skem_accepted"],
            skem_rejected=selection["skem_rejected"], source_sample_observations=len(texts)*4,
            max_caption_words=max(len(t.split()) for t in texts.values()),
            independent_ground_truth=False, hallucination_mitigation_verified=False)
        rows.append(report)
    write_json(ROOT / "PREPARED.json", dict(status="READY_FOR_USER_RECONSTRUCTION", videos=rows,
        command="bash scripts/reconstruct_short3.sh", reconstruction_started=False,
        caption_rules="Visible objects, framing, scale, motion, visibility/occlusion and observed image softness; no inferred identity, mood or unseen action. Up to six English sentences and 80 words per interval.",
        scope="Offline assistant source observations with exact sequential SKEM candidate scoring. No reconstruction, T5 preparation, flow or channel model was run during this preparation."))
    descriptions = read_json(REPO / "configs/captions/short3_source_observations_v2.json")["videos"]
    table = ''.join(f'<tr><td>{r["video"]}<br>{html.escape(descriptions[r["video"]]["description_ko"])}</td>'
        f'<td>{r["seconds"]:.2f}초</td><td>{r["keyframes"]}</td><td>{r["captions"]}</td>'
        f'<td><a href="{r["video"]}/selection.html">키프레임</a> · '
        f'<a href="{r["video"]}/assistant_captions/review.html">캡션</a></td></tr>' for r in rows)
    (ROOT / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>세 영상 복원 준비 완료</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto}td,th{padding:12px;border:1px solid #ddd}table{border-collapse:collapse}code{background:#eee;padding:5px}</style>'
        '<h1>키프레임·캡션 준비 완료</h1><p>복원은 아직 실행하지 않았습니다. 아래 명령으로 세 영상을 순서대로 복원합니다.</p>'
        '<p><code>bash scripts/reconstruct_short3.sh</code></p><p>준비 검사: <code>bash scripts/reconstruct_short3.sh --check</code></p>'
        '<table><tr><th>영상</th><th>입력 길이</th><th>키프레임</th><th>캡션</th><th>준비 자료</th></tr>'+table+'</table>'
        '<p>선택: 확인한 가림·등장·방향 변화 등은 유지하고, 나머지 후보는 SKEM 확률 차이 0.35로 판정. 최대 1초 간격으로 보강합니다.</p>'
        '<p>복원: 광류 추출·AWGN 10dB 전송 → 마지막 17프레임 참조·T5 CPU FP32 사전 계산과 저장 → 생성 30단계 → 화질 평가·기존 결과 비교.</p>'
        '<p>candle-flowers 파일의 실제 내용은 노란 등갓의 초점 변화입니다. 보행 영상의 원본에는 짧은 수평 구도 변동이 있어 입력을 그대로 유지했습니다.</p>'
        '<p>이 자료는 오프라인 AI 작성 입력이며 독립 정답이나 개선 효과 검증이 아닙니다. 생성 품질과 할루시네이션은 복원 후 평가해야 합니다.</p>'
        '<a href="PREPARED.json">입력·선택·캡션 검증 기록</a></html>')
    print(ROOT / "review.html")


if __name__ == "__main__":
    main()
