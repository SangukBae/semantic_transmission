"""Readable evidence for the bounded selector speed experiment."""
import argparse
import csv
import html
import json
import statistics
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO / "src"))
from semantic_transmission.artifacts import write_json
from semantic_transmission.webvid5 import read_json
from validate_skem_speed import selection_path


def main():
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=REPO / "outputs/skem_speed_20260929")
    args=p.parse_args()
    root=args.output.resolve()
    result=read_json(root / "RESULT.json")
    protocol=read_json(root / "protocol.json")
    modes=("baseline","int8","brief","sparse")
    labels={"source":"원본","baseline":"기존 BF16","int8":"8비트","brief":"선택용 설명 간략화","sparse":"초당 4장 후보 검사"}
    historical=result["baseline_source"]=="VERIFIED_HISTORICAL_SELECTION"
    table=[]
    for mode in modes:
        values=[w[mode] for w in result["windows"].values()]
        selections=[read_json(selection_path(root,mode,w)) for w in protocol["windows"]]
        records=[r for selected in selections for r in selected["records"]]
        seconds=sum(r["selection_seconds"] for r in values)
        base_seconds=sum(w["baseline"]["selection_seconds"] for w in result["windows"].values())
        runtime=root / f"selection/{mode}/runtime.json"
        load=read_json(runtime) if runtime.exists() else None
        prior_loads=[read_json(p) for p in (root / "failed_attempts").glob(f"select_{mode}_*/selection/{mode}/runtime.json")]
        attempts=[root / f"resources/select_{mode}.json",*(root / "failed_attempts").glob(f"select_{mode}_*/resources/select_{mode}.json")]
        common_count=sum(len(v["common_interior_frames"]) for v in values)
        common_quality={metric:sum(len(v["common_interior_frames"])*v["common_interior_quality"][metric]
            for v in values if v["common_interior_quality"])/common_count if common_count else None
            for metric in ("psnr_db","lpips_vgg")}
        old=historical and mode=="baseline"
        peaks=[v["peak_allocated_bytes"] for v in selections if "peak_allocated_bytes" in v]
        stage_seconds={stage:sum(read_json(root / f"resources/speed_{Path(v['run']).name}_{stage}.json")["seconds"]
            for v in values) for stage in ("flow","send","channel","receive","reconstruct","evaluate")}
        caption_seconds=sum(read_json(root / v["run"] / "caption_time.json")["seconds"] for v in values)
        table.append({"mode":mode,"comparisons":sum(v["comparisons"] for v in values),
            "selection_seconds":seconds,"selection_speedup":None if historical else base_seconds/seconds,
            "timing_source":"historical log" if old else "new completed comparisons",
            "model_load_seconds":None if old else sum(v["model_load_seconds"] for v in [load,*prior_loads]),
            "model_load_count":None if old else 1+len(prior_loads),"keys":sum(len(v["keys"]) for v in values),
            "process_seconds_all_attempts":None if old else sum(read_json(p)["seconds"] for p in attempts),
            "peak_cuda_allocated_gib":max(peaks)/2**30 if peaks else None,
            "median_description_tokens_reencoded":statistics.median(r["description_tokens_reencoded"] for r in records),
            "median_comparison_seconds":statistics.median(r["total_seconds"] for r in records),
            "max_comparison_seconds":max(r["total_seconds"] for r in records),
            "comparisons_over_180_seconds":sum(r["total_seconds"]>180 for r in records),
            "median_round1_seconds":statistics.median(r["round1_seconds"] for r in records),
            "median_round2_seconds":statistics.median(r["round2_seconds"] for r in records),
            "mean_lpips":sum(v["quality"]["lpips_vgg"] for v in values)/len(values),
            "mean_psnr_db":sum(v["quality"]["psnr_db"] for v in values)/len(values),
            "common_interior_frames":common_count,
            "common_interior_psnr_db":common_quality["psnr_db"],
            "common_interior_lpips":common_quality["lpips_vgg"],
            "caption_seconds_shared_model":caption_seconds,
            "reconstruction_seconds_including_model_load":stage_seconds["reconstruct"],
            "downstream_seconds_including_evaluation":caption_seconds+sum(stage_seconds.values()),
            "total_channel_uses":sum(v["channel_uses"] for v in values),
            "channel_use_ratio":sum(v["channel_uses"] for v in values)/sum(w["baseline"]["channel_uses"] for w in result["windows"].values())})
    coverage=read_json(root / "candidate_coverage.json") if (root / "candidate_coverage.json").exists() else None
    timing_recheck=(read_json(root / "timing_recheck/SUMMARY.json")
                    if (root / "timing_recheck/SUMMARY.json").exists() else None)
    ai_review=read_json(root / "AI_REVIEW.json") if (root / "AI_REVIEW.json").exists() else None
    if coverage and coverage["selection_protocol"] != protocol["signature"]:
        raise ValueError("candidate audit belongs to a different experiment")
    fresh_partial={}
    matches=[]
    for window in protocol["windows"]:
        fresh=root / f"selection/baseline/{window}.json"
        if not fresh.exists(): continue
        selected=read_json(fresh)
        fresh_partial[window]={k:selected[k] for k in ("status","indices","selection_seconds")}
        fresh_partial[window]["comparisons"]=len(selected["records"])
        for a in selected["records"]:
            for b in read_json(root / f"selection/sparse/{window}.json")["records"]:
                if (a["candidate"],a["reference"])==(b["candidate"],b["reference"]):
                    matches.append({"window":window,"candidate":a["candidate"],"reference":a["reference"],
                        "description_and_psss_exact":a["description"]==b["description"] and a["psss"]==b["psss"],
                        "baseline_seconds":a["total_seconds"],"sparse_seconds":b["total_seconds"]})
    write_json(root / "SUMMARY.json",{"conditions":table,"independent_semantic_review":"PENDING",
        "baseline_source":result["baseline_source"],"speedup_scope":result["timing_scope"],
        "fresh_baseline_measurements":fresh_partial,"identical_bf16_comparisons":matches,
        "timing_scope":"Timed candidate processing including image preparation; model load listed separately; shared input preparation and cut scanning excluded.",
        "downstream_timing_scope":"Serial isolated stages including model loads and evaluation; captions share one model, charged to the first run. Descriptive only, not full-pipeline speedup.",
        "scene_change_scope":"No detected cuts in reconstruction windows; separate CPU coverage audit does not validate downstream decisions or quality.",
        "candidate_coverage":coverage,
        "fixed_pair_timing_recheck":timing_recheck,
        "ai_sample_review":ai_review,
        "execution_events":read_json(root / "execution_events.json") if (root / "execution_events.json").exists() else [],
        "quality_scope":"Three 25-frame restarted windows from one development source. Same-key outputs reused exactly; no independent replicates."})
    with (root / "summary.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    sections=[]
    figure,axes=plt.subplots(1,3,figsize=(15,3.5),sharey=True,layout="constrained")
    (root / "sheets").mkdir(exist_ok=True)
    for axis,(window,conditions) in zip(axes,result["windows"].items()):
        sheet=Image.new("RGB",(1440,960),"white")
        draw=ImageDraw.Draw(sheet)
        start=protocol["windows"][window]["start"]
        videos=[f'inputs/{window}/source.mp4']+[f'{conditions[m]["run"]}/receiver/reconstruction/sample_0000.mp4' for m in modes]
        for col,label in enumerate(("source",*modes)):
            for row,index in enumerate((0,6,12,18,24)):
                folder=(root / f"inputs/{window}/frames" if label=="source" else
                        root / conditions[label]["run"] / "receiver/reconstruction/sample_0000_frames")
                files=sorted(folder.glob("*.png"),key=lambda p:int(p.stem))
                image=Image.open(files[index]).convert("RGB").resize((288,160))
                sheet.paste(image,(col*288,row*192+30))
                draw.text((col*288+3,row*192+5),f'{label} / frame {start+index}',fill="black",font=font)
        sheet.save(root / f"sheets/{window}.jpg",quality=94)
        players=''.join(f'<div><b>{html.escape(labels[label])}</b><video controls muted loop src="{path}"></video></div>'
                        for label,path in zip(("source",*modes),videos))
        rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            labels[mode],conditions[mode]['keys'],round(conditions[mode]['selection_seconds'],2),
            round(conditions[mode]['quality']['psnr_db'],3),round(conditions[mode]['quality']['lpips_vgg'],4),
            conditions[mode]['channel_uses']))+'</tr>' for mode in modes)
        for mode in modes:
            with (root / conditions[mode]["run"] / "quality_delivered_mp4.csv").open() as stream:
                per_frame=list(csv.DictReader(stream))
            x=[int(row["frame"]) for row in per_frame]
            y=[float(row["lpips_vgg"]) for row in per_frame]
            line,=axis.plot(x,y,label=mode,linewidth=1.4)
            keys=conditions[mode]["keys"]
            axis.scatter(keys,[y[i] for i in keys],color=line.get_color(),s=16)
        axis.set(title=window,xlabel="Local frame (24 fps)")
        axis.grid(alpha=.2)
        title={"people":"사람","car":"차량","door":"차량 문"}[window]
        sections.append(f'<h2>{title}</h2><div class="videos">{players}</div><table><tr><th>조건</th><th>선택 프레임</th><th>선택 시간(초)</th><th>PSNR ↑</th><th>LPIPS ↓</th><th>채널 사용량</th></tr>{rows}</table><img src="sheets/{window}.jpg">')
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("LPIPS VGG (lower is better)")
    axes[-1].legend(fontsize=9)
    figure.savefig(root / "frame_quality.svg")
    plt.close(figure)
    timing_note=("기준선 시간은 과거 로그입니다. 새 BF16 측정의 변동이 커 확정 가속률을 제시하지 않습니다."
                 if historical else "같은 입력의 단일 실행 측정이며 일반적인 가속률을 보장하지 않습니다.")
    recheck_html=""
    if timing_recheck:
        recheck_rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            row['phase'],row['timed_repeats'],round(row['median_seconds'],2),
            '–'.join(f'{v:.2f}' for v in row['range_seconds'])))+'</tr>'
            for row in timing_recheck['conditions'])
        recheck_html=('<h2>게임 종료 후 동일 프레임 쌍 재측정</h2>'
            '<p>원본 217·223번 프레임, 조건별 예열 1회 제외. 아래 세 구간 선택 시간과 별도 실험이며 '
            '60초 영상의 전체 처리 시간으로 환산하지 않습니다.</p>'
            '<table><tr><th>조건</th><th>반복 수</th><th>중앙값(초)</th><th>범위(초)</th></tr>'
            +recheck_rows+'</table>')
        if timing_recheck.get("recovered_failure"):
            recheck_html+='<p>마지막 기준선 재확인은 CUDA unknown error로 중단 후 재실행했습니다. 실패 로그와 이전 측정은 보존했습니다.</p>'
    review_html=""
    if ai_review and ai_review.get("summary_ko"):
        review_html=('<h2>검토 결과 — 기본값 채택 보류</h2><ul>'+
            ''.join(f'<li>{html.escape(line)}</li>' for line in ai_review['summary_ko'])+
            '</ul><p>AI 표본 검토이며 독립 의미 오류 정답 검수는 미완료입니다. <a href="AI_REVIEW.json">시각·영역·관찰 기록</a></p>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>키프레임 선택 속도 비교</title>
<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:16px}video{width:100%}.videos{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}table{border-collapse:collapse;margin:16px 0}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}</style>
<h1>키프레임 선택 속도 비교</h1><p>한 개발 영상의 짧은 세 구간, 각 25프레임. 출력은 24fps. 같은 키프레임 조합은 복원 결과를 재사용합니다. 60초 평가·독립 의미 오류 검수는 미완료입니다.</p>
<p>시간은 완료된 프레임 비교의 합계이며 모델 로딩·중단 작업은 별도 기록합니다.</p>
<button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})">처음부터 함께 재생</button>'''+f'<p>{timing_note}</p>'+review_html+recheck_html+''.join(sections)+(
        '<h2>프레임별 화질</h2><p>출력 MP4의 LPIPS. 낮을수록 양호하며, 점은 각 조건의 키프레임 위치입니다. '
        '의미 오류 검출 점수는 아닙니다.</p><img src="frame_quality.svg"></html>')
    (root / "review.html").write_text(page)
    print(json.dumps(table,indent=2))


if __name__=="__main__":main()
