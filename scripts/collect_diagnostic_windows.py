#!/usr/bin/env python3
"""Collect high-discrepancy intervals, explicitly without semantic error labels."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_transmission.automatic_validation import write
from semantic_transmission.pair_inputs import load_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    root = args.run.resolve()
    rows = json.loads((root / "real_pairs.json").read_text())
    count = json.loads((root / "protocol.json").read_text())["frames_sampled"]
    from PIL import Image
    def png(f):
        buffer = io.BytesIO()
        Image.fromarray(f).save(buffer, "PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    records, views = [], []
    for row in rows:
        a, b, alignment = load_pair(row, count)
        ranked = sorted(((i, x) for i, x in enumerate(row["rte_transitions"]) if x is not None), key=lambda v: v[1], reverse=True)[:3]
        for i, value in ranked:
            record = {"source_id": row["source_id"], "model": row["model"], "rte": value,
                      "source_frames": alignment["source_indices"][i:i + 2],
                      "reconstruction_frames": alignment["reconstruction_indices"][i:i + 2],
                      "selection": "three largest sampled transition discrepancies per reconstruction",
                      "confirmed_semantic_error": None, "ground_truth": None}
            records.append(record)
            views.append(dict(record, images=[png(a[i]), png(a[i + 1]), png(b[i]), png(b[i + 1])]))
    write(root / "diagnostic_windows.json", {"status": "COLLECTED_UNLABELLED_DIAGNOSTICS", "windows": records})
    page = """<!doctype html><html lang="ko"><meta charset="utf-8"><title>실제 복원 차이 구간</title>
<style>body{font:16px/1.7 system-ui;max-width:980px;margin:32px auto;padding:0 20px;color:#243047}.notice{background:#fff1ce;padding:16px}select{max-width:100%;font:inherit;padding:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}small{color:#5a667a}</style>
<h1>실제 복원에서 차이가 큰 구간</h1><p class="notice">자동으로 차이가 큰 구간을 골랐습니다. 의미 오류가 확정된 구간은 아닙니다. 두 모델의 실행 조건도 달라 점수로 모델 우열을 정할 수 없습니다.</p>
<select id="case"></select><p id="details"></p><h2>원본: 앞 시점 → 뒤 시점</h2><div class="grid"><img id="a0"><img id="a1"></div>
<h2>복원: 같은 물리 시간의 두 시점</h2><div class="grid"><img id="b0"><img id="b1"></div>
<p><small>영상 전체를 같은 길이로 늘이거나, 내용을 맞춰 시간축을 이동하지 않았습니다. 표시 이미지는 공통 평가 해상도 224×128을 확대한 것입니다. 실제 보행 방향·객체 환각의 정답 주석은 없습니다.</small></p>
<script>const views=VIEW_DATA;const s=document.getElementById('case');views.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.source_id+' / '+v.model+' / 원본 프레임 '+v.source_frames.join(' → ');s.appendChild(o)});function show(){const v=views[+s.value];document.getElementById('details').textContent='전이 RTE: '+v.rte.toFixed(3)+' / 확정 오류 여부: 정답 없음';['a0','a1','b0','b1'].forEach((id,i)=>document.getElementById(id).src=v.images[i])}s.onchange=show;if(views.length)show();</script></html>"""
    (root / "real_diagnostics.html").write_text(page.replace("VIEW_DATA", json.dumps(views)))
    print(f"{len(records)} unlabelled diagnostic windows")


if __name__ == "__main__":
    main()
