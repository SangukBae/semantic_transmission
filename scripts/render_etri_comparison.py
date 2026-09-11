#!/usr/bin/env python3
"""Make a local synchronized video viewer from verified comparison artifacts."""
import argparse
import csv
import json
import os
from pathlib import Path
from urllib.parse import quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--baseline-inventory", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.comparison.resolve()
    summary = json.loads((output / "comparison.json").read_text())
    if summary["status"] != "PASSED" or summary["videos"] != 10:
        raise ValueError("viewer requires the completed, verified 10-video comparison")
    manifest = json.loads((root / "batch_manifest.json").read_text())
    inventory = json.loads(args.baseline_inventory.read_text())["baselines"]
    metrics = list(csv.DictReader((output / "per_video.csv").open()))
    clip_path = output / "clip/report.json"
    clip_rows = json.loads(clip_path.read_text())["rows"] if clip_path.exists() else []
    for row in metrics:
        extra = next((r for r in clip_rows if r["video"] == row["video"] and r["model"] == row["model"]), None)
        if extra:
            row["clip_paper_scaled_lossless"] = extra["clip_paper_scaled"]
    data = []
    for source in manifest["inputs"]:
        name = source["id"]
        sources = [("Source", Path(source["path"]))]
        sources.append(("LGVSC_SKEM_DSA_50", next((root / name / "receiver/reconstruction").glob("*.mp4"))))
        for baseline in [row for row in inventory if row["video"] == name]:
            sources.append((f"SGD_{baseline['decoder_policy']}_{baseline['guide_profile']}", Path(baseline["video_path"])))
        panes = []
        for label, path in sources:
            if not path.is_file():
                raise ValueError(f"missing video: {path}")
            row = next((r for r in metrics if r["video"] == name and r["model"] == label), None)
            panes.append({"label": label, "url": quote(os.path.relpath(path, output)), "metrics": row})
        data.append({"name": name, "panes": panes})
    html = r'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETRI 영상 복원 비교</title>
<style>
body{background:#10161f;color:#e8eef6;font:16px system-ui;margin:28px;max-width:1700px}
h1{font-size:25px;margin-bottom:8px}p{color:#b5c2d4;line-height:1.6}
select,button{font:inherit;padding:8px 12px;background:#253449;color:inherit;border:1px solid #536782;border-radius:6px}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:20px 0}
input{flex:1;min-width:200px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px}
article{background:#192432;border-radius:10px;overflow:hidden}h2{font-size:15px;margin:12px;overflow-wrap:anywhere}
video{display:block;width:100%;background:black}.metrics{padding:12px;font-size:14px;line-height:1.6;white-space:pre-line;color:#c2d2e7}
a{color:#9fc9ff}@media(max-width:450px){body{margin:12px}.grid{grid-template-columns:1fr}}
</style>
<h1>ETRI 영상 복원 비교</h1>
<p>512×256 · 10fps · 100프레임. 같은 원본 시점에서 사람의 이동, 객체, 글자와 장면 전환을 확인하세요.
화질 수치는 최종 MP4 기준입니다. 재생 중 브라우저에 따라 약간의 시차가 생길 수 있으므로 정밀 비교에는 프레임 이동을 사용하세요.</p>
<div class="controls"><select id="choice" aria-label="영상 선택"></select><button id="play">재생</button>
<button id="previous">이전 프레임</button><button id="next">다음 프레임</button>
<input type="range" id="seek" min="0" max="99" step="1" value="0" aria-label="프레임"><span id="time">0 / 99</span></div>
<div id="panes" class="grid"></div>
<p>LGVSC I/Q 파일의 byte 수는 연속 신호의 저장 크기입니다. 무선 전송 bit 수를 뜻하지 않습니다.
SGD-JSCC의 공통 채널 사용량은 기존 디지털 패킷을 LDPC/16QAM으로 보낸다고 가정한 환산값입니다.</p>
<p><a href="REPORT.md">비교 보고서</a> · <a href="per_video.csv">영상별 수치 CSV</a></p>
<script>
const dataset=__DATA__;
const choice=document.getElementById('choice'),panes=document.getElementById('panes');
const seek=document.getElementById('seek'),time=document.getElementById('time'),play=document.getElementById('play');
let videos=[],playing=false;
const format=x=>Number(x).toLocaleString('en-US',{maximumFractionDigits:0});
dataset.forEach((item,index)=>{const option=document.createElement('option');option.value=index;option.textContent=item.name;choice.append(option)});
function stop(){videos.forEach(v=>v.pause());playing=false;play.textContent='재생'}
function position(frame){stop();frame=Math.max(0,Math.min(99,frame));seek.value=frame;time.textContent=frame+' / 99';videos.forEach(v=>{v.currentTime=frame/10+0.001})}
function load(){stop();panes.replaceChildren();videos=[];
 for(const pane of dataset[Number(choice.value)].panes){
  const article=document.createElement('article'),title=document.createElement('h2'),video=document.createElement('video'),info=document.createElement('div');
  title.textContent=pane.label;video.src=pane.url;video.preload='auto';video.muted=true;video.playsInline=true;
  info.className='metrics';const m=pane.metrics;
  info.textContent=m?`PSNR ${Number(m.delivered_mp4_psnr_db).toFixed(2)} dB · SSIM ${Number(m.delivered_mp4_ssim).toFixed(3)} · LPIPS ${Number(m.delivered_mp4_lpips_alex).toFixed(3)}\n디지털 패킷 ${format(m.digital_packet_bytes)} B · 연속 복소 심벌 ${format(m.continuous_complex_symbols)}\n모델 입력 직렬화 ${format(m.serialized_model_input_bytes)} B`:'원본 영상';
  if(m&&m.clip_paper_scaled_lossless!==undefined)info.textContent+=`\nCLIP 장면 유사도 ${Number(m.clip_paper_scaled_lossless).toFixed(4)} (복원 PNG 기준)`;
  article.append(title,video,info);panes.append(article);videos.push(video);
 }
 videos[0].addEventListener('ended',stop);seek.value=0;time.textContent='0 / 99';
}
choice.addEventListener('change',load);seek.addEventListener('input',()=>position(Number(seek.value)));
document.getElementById('previous').onclick=()=>position(Number(seek.value)-1);
document.getElementById('next').onclick=()=>position(Number(seek.value)+1);
play.onclick=async()=>{if(playing){stop();return}if(Number(seek.value)>=99)position(0);
 try{await Promise.all(videos.map(v=>v.play()));playing=true;play.textContent='일시정지'}catch(error){stop();alert('브라우저가 영상을 열지 못했습니다. 파일 경로를 확인하세요.')}};
function tick(){if(playing){const master=videos[0];seek.value=Math.min(99,Math.floor(master.currentTime*10));time.textContent=seek.value+' / 99';for(const v of videos.slice(1)){if(Math.abs(v.currentTime-master.currentTime)>.15)v.currentTime=master.currentTime}}requestAnimationFrame(tick)}
load();tick();
</script></html>'''
    (output / "viewer.html").write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")))
    print(output / "viewer.html")


if __name__ == "__main__":
    main()
