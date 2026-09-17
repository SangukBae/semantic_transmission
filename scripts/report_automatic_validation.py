#!/usr/bin/env python3
"""Export a standalone Korean report and an interactive controlled-case viewer."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NAMES = {"reverse": "역순", "freeze": "정지", "lag": "지연", "swap": "순서 교환",
         "object_addition": "객체 추가", "object_omission": "객체 누락", "object_shape": "객체 모양 왜곡"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    root = args.run.resolve()
    s = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    extra_path = root / "additional_baselines/summary.json"
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else None
    real_sources = len(protocol["inventory"])
    synthetic_sources = sum(len(v) for v in protocol["synthetic_seeds"].values())
    failed = [name.upper() for name, gate in s["candidate_gates"].items() if gate["status"] != "PASSED_CONTROLLED_GATE"]
    conclusion = ("채택 기준 미달: " + ", ".join(failed)) if failed else "두 후보가 이번 통제 실험 기준을 통과했다. 일반 의미 오류의 유효성 입증과는 구분한다."
    cases = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
    partial = []
    for metric, domain, kinds in (("rte", "real_temporal", ("reverse", "freeze", "swap")),
                                  ("lssd", "rendered", ("object_addition", "object_omission", "object_shape"))):
        stat = s["statistics"][domain].get(metric)
        if stat is None:
            continue
        for kind in kinds:
            group = [c for c in cases if c["domain"] == domain and c["split"] == "heldout" and c["kind"] == kind and c["severity"] == 0.25]
            detected = sum(c[metric] is not None and c[metric] > stat["threshold_oriented"] for c in group)
            partial.append(f"{metric.upper()} {NAMES[kind]} {detected}/{len(group)}개")
    partial_text = "평가 시점 25%에 오류를 넣었을 때의 검출 수: " + "; ".join(partial) + "."
    diagnostic_link = '<a href="real_diagnostics.html">실제 복원에서 자동 선별한 차이 구간 보기</a>' if (root / "real_diagnostics.html").exists() else ""
    graph_link = '<a href="detection_rates.svg">비교 그래프</a>' if extra else ""
    integration_text = ""
    new_pairs_path = root / "new_input_pair_scores.json"
    if new_pairs_path.exists():
        new_pairs = json.loads(new_pairs_path.read_text())
        source_count = len({r["source_sha256"] for r in new_pairs["results"]})
        integration_text = (f"새 입력 {source_count}개에서 실제 복원 결과 {len(new_pairs['results'])}개를 공통 평가했다. "
                            "이번 연결 확인은 짧은 영상에서 수행했다. 전체 67개 영상을 두 모델로 다시 복원한 결과가 아니다. "
                            "SGD-JSCC는 프레임별 AWGN 경로이고 기존 int4 비교 실험과 구분한다.")
    import numpy as np
    from PIL import Image
    examples = []
    for path in sorted(root.glob("example_*.npz")):
        data = np.load(path)
        sides = {}
        for key in ("source", "reconstructed"):
            frames = []
            for f in data[key]:
                buffer = io.BytesIO()
                Image.fromarray(f).save(buffer, format="PNG")
                frames.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
            sides[key] = frames
        examples.append(dict(name=path.stem.removeprefix("example_"), **sides))
    table = []
    md = ["# 사람 검수 없는 1차 지표 실험", "", "작성일: 2026-09-12", "",
          f"자동 검증 체계 구현과 실험을 완료했다. **{conclusion}**. "
          "이 통제 실험을 실제 복원 영상의 의미 정확도 입증으로 해석하지 않는다.", "",
          f"- 실제 영상 {real_sources}개(ETRI는 조정용, WebVid·Kinetics는 평가용), 렌더링 장면 {synthetic_sources}개.",
          f"- 총 {s['cases']:,}개 원본–변형 쌍, 영상당 {protocol['frames_sampled']}개 시점, 224×128 해상도.",
          f"- 등록된 실제 복원 {s['real_pairs']}쌍을 공통 평가기로 진단.",
          "- 사람이 만든 오류 주석이나 VLM 판정을 정답으로 사용하지 않았다.", "",
          "## 숫자를 읽는 방법", "",
          "검출률은 코드로 넣은 오류에 경보를 낸 비율이다. 오탐률은 밝기·약한 흐림·JPEG 압축처럼 "
          "오류를 넣지 않은 대조군에 경보를 낸 비율이다. 임계값은 조정용 자료의 대조군에서만 정했다. "
          "서로 다른 지표의 오탐률이 같지 않으므로 검출률만으로 순위를 정하면 안 된다.", "",
          "| 검증 대상 | 지표 | 검출률 | 대조군 오탐률 | AUC |", "|---|---|---:|---:|---:|"]
    for domain, kinds in (("real_temporal", ("reverse", "freeze", "lag", "swap")),
                          ("rendered", ("object_addition", "object_omission", "object_shape"))):
        for kind in kinds:
            names = ["rte" if domain == "real_temporal" else "lssd", "ssim", "lpips_alex", "clip_cosine"]
            if extra:
                names += ["tlp_alex", "tof_farneback", "psnr_global_db"]
            for metric in names:
                stat = (extra if extra else s)["statistics"][domain].get(metric)
                if stat is None:
                    continue
                e = stat["errors"][kind]
                line = [NAMES[kind], metric, f"{e['tpr']:.1%}" if e['tpr'] is not None else "측정 불가",
                        f"{stat['heldout_fpr']:.1%}" if stat['heldout_fpr'] is not None else "측정 불가",
                        f"{e['auc']:.3f}" if e['auc'] is not None else "측정 불가"]
                table.append(line)
                md.append("| " + " | ".join(line) + " |")
    md += ["", "## 채택 판단과 부분 구간 오류", "", conclusion, "", partial_text, "",
           *[f"- {name.upper()}: " + "; ".join(gate["failures"]) for name, gate in s["candidate_gates"].items() if gate["failures"]], "",
           "통제 실험의 실패 사례와 점수를 보존한다. 단독 의미 오류 판정기로 사용할 수 있다는 결론은 내리지 않는다.", "",
           "## 이번 실험이 증명하지 않는 것", "",
           "실제 영상의 시간 변형은 프레임 조작 정답이며, 사건 의미를 사람이 판정한 정답이 아니다. 객체 실험은 "
           "단순 도형 렌더링 장면에 한정된다. 실제 사람·자동차의 환각 판정, 가려짐, 객체 식별, 미세한 동작 방향, "
           "자유로운 장면의 사건 순서 정확도는 입증하지 않았다. 표본 시점 사이의 짧은 오류도 놓칠 수 있다.", "",
           "CLIP은 영상 시간축 학습 모델이 아니고, 패치 토큰은 객체 마스크가 아니다. RTE·LSSD는 연구 후보이며 "
           "신규성도 확정하지 않았다. 기존 PSNR은 프레임별 dB 평균(동일 프레임은 168.13 dB 상한)이므로 "
           "일부 프레임만 바꾸는 실험에서 해석에 주의해야 한다.", "",
           "실제 LGVSC·SGD-JSCC 진단에는 의미 오류 정답이 없다. 두 실행의 채널·전송량·해상도 조건도 달라 "
           "모델 우열이나 환각률 비교로 사용할 수 없다. 원본 물리 시간에 맞춰 평가하고 실제 지연을 지우는 정렬은 하지 않았다.", "",
           "## 재현 자료", "", "- [동결 프로토콜](protocol.json), [전체 통계](summary.json), [사례별 점수](scores.csv)",
           "- [프레임 조작 및 렌더링 정답](cases.jsonl), [실제 복원 진단](real_pairs.json)",
           "- [오류 사례 슬라이더](report.html)", "",
           "## 선행 연구와 해석", "",
           "시간축 검증을 별도로 둔 근거는 [Ge et al., CVPR 2024](https://arxiv.org/abs/2404.12391)의 "
           "FVD 시간 왜곡 민감도 분석이다. 해당 연구가 이번 후보를 검증한 것은 아니다. "
           "[ARGUS](https://arxiv.org/abs/2506.07371)는 사람의 정답 캡션을 이용한 Video-LLM 평가이므로 "
           "이번 사람 검수 없는 픽셀 복원 평가의 정답으로 사용하지 않았다.", ""]
    if extra:
        md += ["## 추가 시간축 기준선", "",
               "첫 실험 이후 tLP·tOF·전체 MSE의 PSNR을 같은 사례에 추가했다. 원본·변형 픽셀 SHA256이 "
               "모두 최초 실험과 일치하는지 확인했다. 후보 수식과 임계값은 다시 조정하지 않았다. "
               "기존 프레임별 dB 평균 PSNR과 전체 MSE의 PSNR이 다른 결론을 줄 수 있어 두 값을 구분한다.", "",
               "[추가 비교 통계](additional_baselines/summary.json), "
               "[tLP·tOF 원전 코드](https://github.com/thunil/TecoGAN/blob/master/metrics.py). "
               "이번 계산은 공간 가장자리 crop 없이 모든 표본 시점을 쓴 명시적 변형이다.", ""]
    if diagnostic_link:
        md += ["[실제 복원 차이 구간](real_diagnostics.html): 자동 선별한 진단 구간이며 확정된 의미 오류가 아니다.", ""]
    if integration_text:
        md += ["## 새 입력의 실제 실행 확인", "", integration_text, "",
               "[새 입력 공통 점수](new_input_pair_scores.json), [고정 입력에서의 점수 반복 검사](metric_repeatability.json).", ""]
    (root / "REPORT.md").write_text("\n".join(md))
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in table)
    document = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>사람 검수 없는 지표 실험</title><style>body{font:16px/1.7 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#243047}h1{font-size:28px}.notice{padding:16px;background:#fff1ce;border-radius:8px}.cards{display:flex;gap:18px;flex-wrap:wrap}.cards figure{margin:0}img{width:448px;max-width:100%;image-rendering:auto}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #dde3ed;text-align:left}select,input{padding:8px;font:inherit}small{color:#58667c}</style>
<h1>사람 검수 없는 1차 지표 실험</h1><p class="notice"><b>CONCLUSION</b><br>2,432개 사례에서 기존 지표와 새 후보의 오탐·미탐을 비교했습니다.</p>
<p>사람 대신 코드가 오류를 넣고 정답을 기록했습니다. 실제 영상은 77개, 별도로 만든 도형 장면은 48개입니다. 원본별로 조정용·평가용을 분리했습니다.</p>
<h2>어떤 오류인지 직접 보기</h2><select id="example"></select><p>시점 <span id="frame">1</span> / <span id="frame_count"></span> <input id="slider" type="range" min="0" value="0"></p>
<div class="cards"><figure><figcaption>원본</figcaption><img id="source"></figure><figure><figcaption>코드로 바꾼 영상</figcaption><img id="reconstructed"></figure></div>
<p><small>reverse: 시간 순서 반전 / object_addition: 객체 추가 / object_omission: 객체 누락 / identity: 원본 그대로. real_temporal은 실제 영상의 표본 시점이고 rendered는 도형 렌더링 장면입니다. 영상 전체를 재생하는 화면은 아닙니다.</small></p>
<h2>오류를 얼마나 찾았는가</h2><p>검출률은 넣은 오류를 찾은 비율, 오탐률은 대조군에 잘못 경보를 낸 비율입니다. 각 지표의 임계값은 조정용 자료에서만 정했습니다.</p>
<table><thead><tr><th>대상</th><th>지표</th><th>검출률</th><th>오탐률</th><th>AUC</th></tr></thead><tbody>TABLE_ROWS</tbody></table>
<h2>현재 결론</h2><p>RTE는 시간 변화, LSSD는 화면 일부의 내용 차이를 측정하는 후보입니다. 최종 의미 오류 판정기로 쓸 수 있다는 결론은 내리지 않습니다. 객체 실험은 도형 장면에 한정되고, 실제 복원에서 사람이 잘못 생성되었는지 자동 확정한 결과가 아닙니다. 실제 LGVSC·SGD-JSCC 비교도 조건이 달라 모델 우열을 뜻하지 않습니다.</p>
<p>PARTIAL_RESULTS</p><p>표본 시점 사이의 짧은 오류도 놓칠 수 있습니다. 프레임 대신 원본 영상 단위로 신뢰구간을 계산했습니다. 상세 수치·한계·재현 방법은 같은 폴더의 REPORT.md와 protocol.json에 있습니다.</p>
<p>DIAGNOSTICS_LINK GRAPH_LINK</p>
<p>INTEGRATION_TEXT</p>
<script>const examples=EXAMPLE_DATA;const select=document.getElementById('example');examples.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=e.name;select.appendChild(o)});function show(){const e=examples[+select.value];const slider=document.getElementById('slider');slider.max=e.source.length-1;const i=Math.min(+slider.value,e.source.length-1);document.getElementById('source').src=e.source[i];document.getElementById('reconstructed').src=e.reconstructed[i];document.getElementById('frame').textContent=i+1;document.getElementById('frame_count').textContent=e.source.length}select.onchange=show;document.getElementById('slider').oninput=show;if(examples.length)show();</script></html>"""
    document = document.replace("2,432개", f"{s['cases']:,}개").replace("77개", f"{real_sources}개").replace("48개", f"{synthetic_sources}개")
    document = document.replace("CONCLUSION", html.escape(conclusion)).replace("PARTIAL_RESULTS", html.escape(partial_text))
    document = document.replace("DIAGNOSTICS_LINK", diagnostic_link).replace("GRAPH_LINK", graph_link)
    document = document.replace("INTEGRATION_TEXT", html.escape(integration_text))
    (root / "report.html").write_text(document.replace("TABLE_ROWS", rows).replace("EXAMPLE_DATA", json.dumps(examples)))
    # Standalone research figure, with each metric's different false-alarm rate shown.
    if extra:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        groups = [("real_temporal", ["reverse", "freeze", "lag", "swap"], ["rte", "tof_farneback", "tlp_alex", "ssim"]),
                  ("rendered", ["object_addition", "object_omission", "object_shape"], ["lssd", "ssim", "lpips_alex", "clip_cosine"])]
        for ax, (domain, kinds, metrics) in zip(axes, groups):
            x = np.arange(len(kinds))
            for i, metric in enumerate(metrics):
                stat = extra["statistics"][domain].get(metric)
                if stat is None or stat['heldout_fpr'] is None:
                    continue
                ax.bar(x + (i - 1.5) * 0.2, [100 * stat["errors"][k]["tpr"] for k in kinds], 0.19,
                       label=f"{metric} (false alarms {100 * stat['heldout_fpr']:.1f}%)")
            ax.set_xticks(x, [k.replace("object_", "") for k in kinds])
            ax.set_ylim(0, 105); ax.set_ylabel("Detection rate (%)")
            ax.set_title("Temporal interventions on real videos" if domain == "real_temporal" else "Object interventions in rendered scenes")
            ax.legend(fontsize=8, ncol=2, loc="lower left"); ax.grid(axis="y", alpha=0.15)
        fig.suptitle("Controlled evaluation: " + ("candidate metrics did not all pass acceptance criteria" if failed else "controlled acceptance criteria passed"))
        fig.savefig(root / "detection_rates.png", dpi=150)
        fig.savefig(root / "detection_rates.svg")
        plt.close(fig)
    print(root / "REPORT.md")
    print(root / "report.html")


if __name__ == "__main__":
    main()
