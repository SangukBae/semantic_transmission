#!/usr/bin/env python3
"""Report the frozen MTE/OTF pilot, including paired uncertainty and examples."""
import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from semantic_transmission.automatic_validation import auc, write
from semantic_transmission.metric_v2_validation import METRICS, _verify

BASELINES = ("psnr_db", "ssim", "lpips_alex", "clip_cosine", "rte", "lssd", "tof_raft_small", "tof_farneback", "tlp_alex")


def comparisons(rows, summary):
    out = {}
    for group, metrics in summary["groups"].items():
        domain, target = group.split("/")
        candidate = "mte_tail" if target == "motion" else "otf_distortion_error"
        test = [r for r in rows if r["split"] == "heldout" and r["domain"] == domain and r["target"] in (target,"control") and r["observable"]]
        cmp = {}
        for baseline in BASELINES:
            deltas, identifiers = [], []
            for sid in sorted({r["source_id"] for r in test}):
                common = [r for r in test if r["source_id"] == sid and r.get(candidate) is not None and r.get(baseline) is not None]
                pos = [r for r in common if r["target"] == target]
                neg = [r for r in common if r["target"] == "control"]
                if not pos or not neg:
                    continue
                ca = auc([r[candidate] for r in pos], [r[candidate] for r in neg])
                ba = auc([METRICS[baseline]*r[baseline] for r in pos], [METRICS[baseline]*r[baseline] for r in neg])
                deltas.append(ca-ba)
                identifiers.append(sid)
            if deltas:
                rng = np.random.default_rng(20260914)
                draws = rng.integers(0,len(deltas),(500,len(deltas)))
                ci = np.quantile(np.asarray(deltas)[draws].mean(1), [.025,.975]).tolist()
                cmp[baseline] = {"mean_within_source_auc_difference": float(np.mean(deltas)),
                    "source_bootstrap_95ci": ci, "paired_sources": len(deltas), "source_ids": identifiers,
                    "positive_lower_bound": ci[0] > 0,
                    "scope": "within-source AUC on jointly measurable cases; coverage reported separately"}
        advantage = len(cmp) == len(BASELINES) and all(v["positive_lower_bound"] for v in cmp.values())
        out[group] = {"candidate": candidate, "comparisons": cmp,
            "superiority_criterion": bool(advantage), "candidate_gate": metrics[candidate]["gate"],
            "adopt_for_controlled_pilot": bool(advantage and metrics[candidate]["gate"] == "PASSED_CONTROLLED_PILOT")}
    return out


def image_data(frame):
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise ValueError("JPEG failed")
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); root=args.output
    _verify(root)
    summary=json.loads((root/"summary.json").read_text())
    merged={}
    for stage in ("motion","object"):
        for path in (root/"scores"/stage).glob("*.json"):
            row=json.loads(path.read_text()); merged.setdefault(row["case_id"],{}).update(row)
    rows=list(merged.values())
    compared=comparisons(rows,summary)
    write(root/"paired_comparisons.json",compared)
    selected=[]
    text=["# MTE·OTF 2차 자동 검증 결과", "", "새 공개 원본과 합성 장면에서 실시한 통제 오류 실험이다. 실제 복원의 의미 정답이나 논문 신규성을 입증한 결과가 아니다.", "",
          f"원본 {summary['sources']}개, 파생 사례 {summary['cases']}개. 개발/평가 분리는 원본 단위이며 평가 원본 수는 각 도메인 16개다.",""]
    for group,metrics in summary["groups"].items():
        domain,target=group.split("/"); candidate=compared[group]["candidate"]
        text.extend(["## "+group,"","| 지표 | 정상 오탐률 | 정상 측정률 | 오류 종류 평균 검출률 | 기준 |","|---|---:|---:|---:|---|"])
        show=[candidate,"mte_mean_ablation","mte_vector_ablation",*BASELINES] if target=="motion" else [candidate,"otf_error",*BASELINES]
        for m in show:
            if m not in metrics:continue
            r=metrics[m];tpr=np.mean([e["tpr_abstentions_as_misses"] for e in r["errors"].values()])
            fpr="측정 불가" if r["heldout_fpr"] is None else f"{r['heldout_fpr']:.1%}"
            text.append(f"| {m} | {fpr} | {r['control_coverage']:.1%} | {tpr:.1%} | {r['gate']} |")
        text.extend(["", "평균 검출률만으로 채택하지 않는다. 사전 기준은 오류 **각 종류**에 적용하며 AUC·오탐·측정률을 함께 본다.","",
                     "| 후보 오류 | AUC | 검출률 (원본 bootstrap 95% CI) | 짧은 12.5% 오류 검출률 |", "|---|---:|---|---:|"])
        for kind,e in metrics[candidate]["errors"].items():
            ci=e["tpr_source_bootstrap_95ci"]; area="null" if e["auc"] is None else f"{e['auc']:.3f}"
            text.append(f"| {kind} | {area} | {e['tpr_abstentions_as_misses']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] | {e['by_severity']['0.125']['tpr']:.1%} |")
        c=compared[group]
        text.extend(["", f"기존 비교 지표 전체에 대한 우위 기준 충족: **{c['superiority_criterion']}**. 후보 채택 기준: **{c['candidate_gate']}**.", "",
                     "| 비교 대상 | 원본 내 AUC 차이 평균 | 쌍별 원본 bootstrap 95% CI |", "|---|---:|---|"])
        for b,r in c["comparisons"].items():
            ci=r["source_bootstrap_95ci"]
            text.append(f"| {b} | {r['mean_within_source_auc_difference']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        threshold=metrics[candidate]["threshold"]
        group_rows=[r for r in rows if r["domain"]==domain and r["split"]=="heldout" and r["target"] in (target,"control")]
        false=[r for r in group_rows if r["target"]=="control" and r.get(candidate) is not None and r[candidate]>threshold]
        misses=[r for r in group_rows if r["target"]==target and r["observable"] and (r.get(candidate) is None or r[candidate]<=threshold)]
        # Deterministic failure examples, not a human-picked success gallery.
        examples=sorted(false,key=lambda r:-r[candidate])[:2]+sorted(misses,key=lambda r:(r["severity"],r["case_id"]))[:3]
        for r in examples:
            selected.append({"row":r,"candidate":candidate,"threshold":threshold,"group":group})
    text.extend(["", "## 자료의 범위", "", "- 합성 객체 마스크와 궤적은 코드 정답이다. 실제 공개 영상은 시간 변형 정답만 있다.",
                 "- DAVIS JPEG에는 촬영 FPS가 없어 매 3번째 프레임을 8 Hz로 재생한 시험 시간축을 사용했다. 원 촬영 FPS 평가가 아니다.",
                 "- 기존 LGVSC·SGD 복원 쌍 진단은 의미 정답 없이 점수·차이 구간만 제공한다.",
                 "- SAM2 검출 실패/과분할과 DINO 표현 차이가 OTF를 오염시킬 수 있다. 추출기 감사 결과는 extractor_audit.json을 확인한다.",
                 "- MTE는 방향 히스토그램 기반 국소 움직임 점수이며 실제 장거리 객체 궤적을 식별하는 모델은 아니다.",
                 "- TecoGAN, InterDyn, HOTA, VBench, PLACID와 목적·구성 요소가 겹친다. 신규성은 입증되지 않았다.",
                 "", "[실패 사례 비교 화면](report.html) · [모든 수치](scores.csv) · [프로토콜](protocol.json) · [쌍별 비교](paired_comparisons.json)"])
    (root/"REPORT.md").write_text("\n".join(text)+"\n")
    write(root/"failure_examples.json",[{k:v for k,v in x.items() if k!="row"}|{"case_id":x["row"]["case_id"]} for x in selected])
    examples=[]
    for x in selected:
        r=x["row"]
        with np.load(root/"inputs"/(r["source_stem"]+".npz")) as f:a=f["frames"]
        with np.load(root/"cases"/(r["case_id"]+".npz")) as f:b=f["frames"]
        examples.append({"id":r["case_id"],"kind":r["kind"],"metric":x["candidate"],"score":r.get(x["candidate"]),
            "threshold":x["threshold"],"changed":r["changed_frames"],"a":[image_data(f) for f in a],"b":[image_data(f) for f in b]})
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>MTE·OTF 실패 사례</title>
<style>body{font:17px system-ui;max-width:1000px;margin:30px auto;padding:15px;background:#f5f7fa;color:#132334}select,input{width:100%;margin:12px 0}.pair{display:flex;gap:18px}.pair div{flex:1}img{width:100%}pre{white-space:pre-wrap;background:white;padding:15px}</style>
<h1>MTE·OTF 자동 검증: 오탐·미검출 사례</h1><p>정답은 코드의 변형 기록입니다. 이 화면은 사람이 정답을 만드는 검수 절차가 아닙니다. 공개 영상과 합성 장면 결과를 구분해 보세요.</p>
<select id="pick"></select><input id="time" type="range" min="0" value="0"><div id="stamp"></div><div class="pair"><div>원본<img id="a"></div><div>비교 영상<img id="b"></div></div><pre id="info"></pre>
<script>const cases=DATA;const pick=document.getElementById('pick'),slider=document.getElementById('time');cases.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=x.id;pick.append(o)});function draw(){if(!cases.length)return;const x=cases[+pick.value];slider.max=x.a.length-1;let t=Math.min(+slider.value,x.a.length-1);document.getElementById('a').src=x.a[t];document.getElementById('b').src=x.b[t];document.getElementById('stamp').textContent=`${t}/ ${x.a.length-1} 프레임, ${(t/8).toFixed(3)}초, 변형 프레임: ${x.changed.includes(t)}`;document.getElementById('info').textContent=JSON.stringify({종류:x.kind,지표:x.metric,점수:x.score,검출문턱:x.threshold},null,2)}pick.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;draw();</script></html>'''
    (root/"report.html").write_text(page.replace("DATA",json.dumps(examples).replace("</","<\\/")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(summary["groups"]),figsize=(16,5),layout="constrained")
    for ax,(group,metrics) in zip(np.atleast_1d(axes),summary["groups"].items()):
        names=[compared[group]["candidate"],"ssim","lpips_alex","tof_raft_small","rte","lssd"]
        yy=np.arange(len(names));tp=[np.mean([e["tpr_abstentions_as_misses"] for e in metrics[m]["errors"].values()]) for m in names]
        fp=[metrics[m]["heldout_fpr"] for m in names]
        ax.barh(yy-.17,tp,height=.3,label="Detection (mean over error types)")
        ax.barh(yy+.17,fp,height=.3,label="False alarm",color="#d86842")
        ax.set(yticks=yy,yticklabels=names,xlim=(0,1),title=group);ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    axes[0].legend(loc="lower right",fontsize=8)
    fig.savefig(root/"comparison.png",dpi=180);fig.savefig(root/"comparison.svg");plt.close(fig)
    print("report written",root,flush=True)


if __name__=="__main__":main()
