"""Build shareable ETRI01 comparison videos, plots, and an offline review page."""
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from evaluate_etri01_ablation import BASE, ROOT, CASES, read, load_case

LABELS = {"historical": "이전 공식 복원", "replay": "현재 PC 기존 설정", "aligned": "정렬 보정",
          "clean_keys": "정렬 + 원본 키프레임(진단)", "action_caption": "정렬 + 행동 캡션(수동)",
          "dense_2s": "정렬 + 키프레임 7장", "dense_1s": "정렬 + 키프레임 12장",
          "combined": "정렬 + 12장 + 행동 캡션(수동)"}


def video_command(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str,args)], check=True)


def main():
    results = {name: read(ROOT / f"evaluation_{name}.json") for name in CASES}
    if not all(r["all_five_metrics"] for r in results.values()):
        raise ValueError("Full evaluation is required before final report")
    assets = ROOT / "review_assets"
    assets.mkdir(exist_ok=True)
    shutil.copyfile(BASE / "data/normalized.mp4", assets / "source.mp4")
    for name in CASES:
        run = BASE if name == "historical" else ROOT / name
        video = run / "receiver/reconstruction/sample_0000.mp4"
        if name in {"historical", "replay"}:
            video_command(["-i", video, "-vf", r"select=not(eq(n\,180)),setpts=N/(24*TB)",
                           "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", assets / f"{name}.mp4"])
        else:
            shutil.copyfile(video, assets / f"{name}.mp4")
    # Common timeline, with panel labels outside the video pixels.
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    inputs = [assets / "source.mp4", assets / "historical.mp4", assets / "combined.mp4"]
    filters = []
    for i, label in enumerate(["SOURCE", "PREVIOUS", "ALIGNED + 12 KEYS + ACTION CAPTION"]):
        filters.append(f"[{i}:v]pad=iw:ih+32:0:32:color=white,drawtext=fontfile={font}:text='{label}':x=8:y=8:fontsize=16:fontcolor=black[v{i}]")
    filters.append("[v0][v1][v2]hstack=inputs=3[out]")
    args = []
    for path in inputs:
        args.extend(["-i",path])
    video_command([*args,"-filter_complex",";".join(filters),"-map","[out]","-an","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",ROOT / "comparison_source_previous_combined.mp4"])
    panel_inputs = [assets / f"{name}.mp4" for name in ["source", "historical", "dense_1s", "combined"]]
    panel_labels = ["SOURCE", "PREVIOUS: 3 KEYS", "ALIGNED: 12 KEYS", "ALIGNED: 12 KEYS + ACTION CAPTION"]
    panel_args, panel_filters = [], []
    for i,(path,label) in enumerate(zip(panel_inputs,panel_labels)):
        panel_args.extend(["-i",path])
        panel_filters.append(f"[{i}:v]pad=iw:ih+32:0:32:color=white,drawtext=fontfile={font}:text='{label}':x=8:y=8:fontsize=16:fontcolor=black[p{i}]")
    # drawtext treats colons as option separators even inside text quotes.
    panel_filters = [f.replace("PREVIOUS: ","PREVIOUS - ").replace("ALIGNED: ","ALIGNED - ") for f in panel_filters]
    panel_filters.append("[p0][p1][p2][p3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]")
    video_command([*panel_args,"-filter_complex",";".join(panel_filters),"-map","[out]","-an","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",ROOT / "comparison_four_panel.mp4"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(11,8), constrained_layout=True)
    source = read(ROOT / "source_tracking.json")
    time = np.arange(240)/24
    axes[0].plot(time,[r["center_x"] if r else np.nan for r in source],"k--",lw=2,label="source")
    for name in ["historical","aligned","dense_2s","dense_1s","combined"]:
        tracks = read(ROOT / f"tracking_{name}.json")["reconstruction"]
        axes[0].plot(time,[r["center_x"] if r else np.nan for r in tracks],label=name,alpha=.85)
    axes[0].set(xlabel="Time (s)",ylabel="Coat centroid x (pixels)",title="Horizontal position proxy (fixed color mask; visually reviewed)")
    axes[0].legend(ncol=3)
    for name in ["historical","aligned","action_caption","dense_2s","dense_1s","combined"]:
        r = results[name]
        channel = r["channel"]["total_complex_channel_uses"]
        axes[1].scatter(channel,r["lossless_frames"]["psnr_db"],s=45)
        offset = {"historical":(8,8),"aligned":(8,-15),"action_caption":(8,24),
                  "dense_2s":(8,6),"dense_1s":(-65,14),"combined":(-65,-17)}[name]
        axes[1].annotate(name,(channel,r["lossless_frames"]["psnr_db"]),xytext=offset,textcoords="offset points",fontsize=9)
    axes[1].set(xlabel="Total complex channel uses (visual + LDPC metadata)",ylabel="PSNR (dB)",title="One video / one seed: increased information vs quality")
    axes[1].set_ylim(19.0, 24.4)
    fig.savefig(ROOT / "trajectory_and_rate.png",dpi=160)
    plt.close(fig)
    base_uses = results["historical"]["channel"]["total_complex_channel_uses"]
    table = []
    for name,r in results.items():
        q = r["lossless_frames"]
        uses = r["channel"]["total_complex_channel_uses"]
        table.append({"name":name,"label":LABELS[name],"psnr":q["psnr_db"],"ssim":q["ssim"],"lpips":q["lpips_vgg"],"dists":q["dists"],"clip":q["clip"],"roi_psnr":r["source_person_roi"]["psnr_db"],"position_mae":r["motion_proxy"]["horizontal_mae_pixels"],"channel_uses":uses,"rate_multiple":uses/base_uses if uses is not None else None})
    (ROOT / "comparison_summary.json").write_text(json.dumps(table,ensure_ascii=False,indent=2)+"\n")
    lines = ["# ETRI 첫 번째 영상: 단계 1·2 복원 비교", "",
        "실행: 2026-09-17~18, 현재 WSL/RTX 4080. 240프레임·24fps·576×320·seed 42·30 steps. 모델 가중치는 학습하지 않았다.", "",
        "[동기 재생 비교 페이지](comparison.html) · [원본/이전/12장/통합 4분할 영상](comparison_four_panel.mp4) · [원본/이전/통합 3분할 영상](comparison_source_previous_combined.mp4) · [측정 JSON](comparison_summary.json) · [입력/전송/시간축 검증](AUDIT.json)", "",
        "이번 영상에서는 정렬 보정+키프레임 12장 조건이 전체 PSNR·SSIM·LPIPS·CLIP·DISTS 모두 가장 좋았다. 이전 대비 PSNR은 19.71→23.83dB이고, 총 채널 사용량은 4.09배다. 통합 조건은 사람 영역 PSNR/위치 proxy에서 조금 더 좋았으나 전체 화질 5개 지표는 12장-only보다 나빴다. 수동 행동 캡션의 추가 효과는 일관되지 않았다.", "",
        "## 동일한 240개 시점의 비교", "",
        "MP4 인코딩 전 PNG 기준이다. PSNR/SSIM/CLIP은 높을수록, LPIPS-VGG/DISTS/위치 오차는 낮을수록 좋다. 과거 영상의 중복 경계 1장을 제거했다.", "",
        "| 조건 | PSNR dB | SSIM | LPIPS VGG | DISTS | CLIP | 사람 ROI PSNR | 위치 오차 px | 전송량 배수 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        rate = f"{r['rate_multiple']:.3f}" if r['rate_multiple'] is not None else "진단 전용"
        lines.append(f"| {r['label']} | {r['psnr']:.4f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['dists']:.4f} | {r['clip']:.4f} | {r['roi_psnr']:.4f} | {r['position_mae']:.2f} | {rate} |")
    lines += ["", "## 무엇을 바꿨는가", "",
        "정렬 보정은 끝 잠재 프레임의 조건을 5칸 단위로 반올림하지 않고, 이전 출력의 실제 마지막 17프레임을 overlap으로 인코딩하며, 중복 경계를 제거한다. VAE가 손실을 주므로 원본/수신 픽셀을 그대로 붙이는 처리는 아니다.", "",
        "원본 키프레임 조건은 NTSCC/채널을 우회한 진단용 oracle이다. 이 조건의 화질을 실제 전송 성능으로 해석하지 않는다. 행동 캡션은 원본을 확인해 수동 작성했으며 자동 캡션 모델이나 딥러닝 학습 개선을 입증하지 않는다.", "",
        "7장/12장 조건은 기존 0/179/239 키프레임을 보존하고 간격을 각각 최대 48/24프레임으로 제한한다. 기존 3장 유지로 총 개수가 6/11장이 아닌 7/12장이 되었다. 광류는 기존 두 구간의 scalar를 반복하고, 캡션만 바꾸는 조건 외에는 기존 캡션을 반복했다.", "",
        "12장 조건과 통합 조건의 수신 시각 심벌과 PNG는 바이트 단위로 동일하다. 원본 키프레임 oracle을 제외하면 모든 기존 3장 수신 PNG도 보존했다. 추가 키프레임과 메타데이터의 전송량을 포함한다.", "",
        "## 끝점 확인", "", "| 조건 | 프레임 0 PSNR | 프레임 179 PSNR | 프레임 239 PSNR |", "|---|---:|---:|---:|"]
    for name,r in results.items():
        scores = {x["index"]: x["psnr_db"] for x in r["keyframe_scores"]}
        lines.append(f"| {LABELS[name]} | {scores[0]:.3f} | {scores[179]:.3f} | {scores[239]:.3f} |")
    lines += ["", "정렬 보정은 두 구간 끝점의 PSNR을 올렸지만 전체 평균 화질을 일관되게 올리지는 않았다. 원본 키프레임을 넣어도 중간 동작/외형 차이가 남았다. 이번 한 영상에서는 키프레임 간격 축소의 픽셀 화질 개선이 더 컸다. 통합 조건의 캡션 추가 효과는 위 표의 12장 조건과 직접 비교한다.", "",
        "## 총 채널 사용량", "", "직렬화 파일 byte나 RF bit 수가 아닌 복소 채널 사용 횟수이다. NTSCC 시각 심벌과 프레이밍/rate index/캡션/LDPC/16QAM 메타데이터를 포함한다. 물리 링크 헤더 등은 포함하지 않는다.", "",
        "| 조건 | 시각 심벌 | 디지털 심벌 | 합계 |", "|---|---:|---:|---:|"]
    for name,r in results.items():
        if name == "clean_keys":
            continue
        c = r["channel"]
        lines.append(f"| {LABELS[name]} | {c['visual_complex_channel_uses']:,} | {c['digital_complex_channel_uses']:,} | {c['total_complex_channel_uses']:,} |")
    lines += ["", "## 최종 MP4 화질", "", "| 조건 | PSNR dB | SSIM | LPIPS VGG | DISTS |", "|---|---:|---:|---:|---:|"]
    for name,r in results.items():
        q = r["delivered_mp4"]
        lines.append(f"| {LABELS[name]} | {q['psnr_db']:.4f} | {q['ssim']:.4f} | {q['lpips_vgg']:.4f} | {q['dists']:.4f} |")
    lines += ["", "## 해석 범위와 검증", "",
        "현재 PC에서 기존 설정을 재실행한 PNG 241장과 MP4는 과거 결과와 바이트 단위로 일치했다. 과거 원본/복원 파일 hash도 유지됐다. 기준선 지표는 source/video hash가 일치하는 기존 공식 5개 지표 CSV를 같은 240행으로 정렬해 재사용했다. 새 조건은 동일 평가기로 계산했다.", "",
        "관련 검사 24 passed, 1 skipped. 단위 검사의 명시적 GPU parity 항목은 별도 GPU 테스트 플래그가 없어 skip됐으며, 실제 이번 복원/5개 지표 계산은 GPU에서 실행했다. 검증 로그는 regression.log에 있다.", "",
        "이것은 개발 영상 한 개와 seed 42의 비교다. VAE가 확률적으로 샘플링하므로 구간 길이/개수/overlap 길이가 달라지는 조건은 난수 소비 순서도 달라진다. seed가 같아도 서로 다른 분할 사이에서 latent noise를 일대일 공유하지 않으며, 일반화나 동일 전송량 우월성을 주장하지 않는다.", "",
        "사람 ROI는 원본 남색 코트의 색상/연결성 bbox에 고정 여백을 더했고 모든 후보에서 같은 영역을 사용했다. 위치 오차는 각 영상에서 독립적으로 구한 코트 중심점의 보조 proxy이며, 검증된 객체 추적/포즈 지표가 아니다. 오른쪽 끝 도달 시점은 몸의 회전 시작 시점과 다르다.", "",
        "영상 생성과 평가를 동시에 실행했을 때 host RAM/swap 여유가 줄어 평가만 중단했고, 완료된 평가를 보존한 채 나머지는 생성 종료 후 순차 실행했다. 마지막 생성에는 임시 swap 8GiB를 추가했으며, 작업 후 swapoff와 파일 제거를 완료했다(runtime_resources.json). 모델 정밀도/가중치는 바꾸지 않았다. 모든 최종 평가는 완료 상태를 확인한 뒤 보고했다.", "",
        "[실험 설계](EXPERIMENT_PLAN.md) · [시각 검수 기록](visual_review_notes.md) · [회귀 검사](regression.log)", "",
        "재현 코드는 scripts/run_etri01_ablation.py, evaluate_etri01_ablation.py, audit_etri01_ablation.py, report_etri01_ablation.py이다. 각 조건 폴더의 experiment.json에 입력 hash와 설정이 있다. 기존 폴더를 덮어쓰지 않으며 --resume은 완료된 조건만 건너뛴다.", "",
        "![위치와 전송량 비교](trajectory_and_rate.png)", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    rows=[]
    for r in table:
        rate = f"{r['rate_multiple']:.2f}×" if r['rate_multiple'] is not None else "진단 전용"
        rows.append(f"<tr><td>{html.escape(r['label'])}</td><td>{r['psnr']:.2f}</td><td>{r['ssim']:.4f}</td><td>{r['lpips']:.4f}</td><td>{r['dists']:.4f}</td><td>{r['roi_psnr']:.2f}</td><td>{r['position_mae']:.1f}</td><td>{rate}</td></tr>")
    options="".join(f'<option value="{name}" {"selected" if name == "combined" else ""}>{html.escape(LABELS[name])}</option>' for name in CASES if name != "historical")
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ETRI 01 복원 비교</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#101620;color:#e5ebf3;margin:24px auto;max-width:1800px;padding:0 20px}h1{font-size:26px}select,button{padding:10px;background:#243249;color:inherit;border:1px solid #567;border-radius:6px} .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%;background:black}label{display:block;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #394455}td:first-child,th:first-child{text-align:left}.note{color:#b9c7d9}input{width:75%}img{max-width:100%}@media(max-width:800px){.grid{grid-template-columns:1fr}table{font-size:12px}}</style>
<h1>ETRI 첫 번째 영상 · 복원 비교</h1><p>240프레임 / 24fps / 576×320 · 생성 seed 42 · 30 steps</p>
<label>비교할 새 복원 <select id="variant">OPTIONS</select></label>
<div class="grid"><div>원본<video id="source" src="review_assets/source.mp4" muted playsinline preload="auto"></video></div><div>이전 공식 복원<video id="old" src="review_assets/historical.mp4" muted playsinline preload="auto"></video></div><div id="newlabel">선택한 복원<video id="new" src="review_assets/combined.mp4" muted playsinline preload="auto"></video></div></div>
<p><button id="play">동시 재생 / 정지</button> <button id="back">−1 프레임</button> <button id="next">+1 프레임</button></p><p><input id="seek" type="range" min="0" max="239" step="1" value="0"> <span id="time">0.000초</span> <label>재생 속도 <select id="speed"><option>1</option><option>0.5</option><option>0.25</option></select></label></p>
<p class="note">이전 복원의 중복 경계 1프레임을 제거해 시간을 맞췄습니다. 행동 캡션은 원본을 확인해 수동 작성한 진단 조건이며, 자동 캡션 개선이나 학습된 새 모델의 성능이 아닙니다. 키프레임 7장·12장은 추가 전송량을 사용합니다.</p>
<h2>같은 240개 시점의 화질</h2><p class="note">표는 MP4 인코딩 전 PNG 기준. PSNR·SSIM은 높을수록, LPIPS·DISTS·위치 오차는 낮을수록 좋습니다. 위치 오차는 남색 코트의 색상 기반 중심점 보조 지표입니다.</p><div style="overflow:auto"><table><thead><tr><th>조건</th><th>PSNR dB ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>DISTS ↓</th><th>사람 ROI PSNR ↑</th><th>위치 오차 px ↓</th><th>총 전송량</th></tr></thead><tbody>ROWS</tbody></table></div><p><a href="comparison_source_previous_combined.mp4">원본·이전·통합 조건 비교 영상</a> · <a href="REPORT.md">상세 보고서</a> · <a href="comparison_summary.json">측정 데이터</a></p><img src="trajectory_and_rate.png" alt="궤적과 전송량별 화질">
<script>const vids=['source','old','new'].map(id=>document.getElementById(id)), master=vids[0],seek=document.getElementById('seek');let playing=false;
function jump(t){for(const v of vids)v.currentTime=Math.min(t,9.958333);seek.value=Math.round(t*24);document.getElementById('time').textContent=t.toFixed(3)+'초'}
document.getElementById('play').onclick=()=>{playing=!playing;for(const v of vids){if(playing)v.play().catch(()=>{});else v.pause()}};
seek.oninput=()=>{playing=false;vids.forEach(v=>v.pause());jump(Number(seek.value)/24)};
document.getElementById('next').onclick=()=>{playing=false;vids.forEach(v=>v.pause());jump(Math.min(239,Math.round(master.currentTime*24)+1)/24)};
document.getElementById('back').onclick=()=>{playing=false;vids.forEach(v=>v.pause());jump(Math.max(0,Math.round(master.currentTime*24)-1)/24)};
master.ontimeupdate=()=>{if(playing){seek.value=Math.round(master.currentTime*24);document.getElementById('time').textContent=master.currentTime.toFixed(3)+'초';for(const v of vids.slice(1))if(Math.abs(v.currentTime-master.currentTime)>.10)v.currentTime=master.currentTime}};
master.onended=()=>{playing=false;vids.forEach(v=>v.pause())};document.getElementById('speed').onchange=e=>vids.forEach(v=>v.playbackRate=Number(e.target.value));
document.getElementById('variant').onchange=e=>{const t=master.currentTime;playing=false;vids.forEach(v=>v.pause());vids[2].src='review_assets/'+e.target.value+'.mp4';vids[2].onloadedmetadata=()=>jump(t)};
</script></html>'''.replace('OPTIONS',options).replace('ROWS',''.join(rows))
    (ROOT / "comparison.html").write_text(page)
    page = page.replace('value="combined" selected', 'value="combined" ').replace('value="dense_1s" ', 'value="dense_1s" selected ')
    page = page.replace('id="new" src="review_assets/combined.mp4"', 'id="new" src="review_assets/dense_1s.mp4"')
    page = page.replace('<a href="comparison_source_previous_combined.mp4">', '<a href="comparison_four_panel.mp4">원본·이전·12장·통합 4분할 영상</a> · <a href="comparison_source_previous_combined.mp4">')
    page = page.replace('input{width:75%}', 'a{color:#8bd3ff}input{width:75%}')
    (ROOT / "comparison.html").write_text(page)
    print(json.dumps(table,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
