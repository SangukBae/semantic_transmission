"""Five-view media and four-condition metrics; semantic review remains separate."""
import csv
import html
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.video_io import probe
from semantic_transmission.webvid5 import read_json
from etri_conditioning_evidence import frames, METRICS, SAMPLES

CASES = ("baseline", "no_rounding", "tail17", "combined")
LABELS = ("SOURCE", "BASELINE", "NO ROUNDING", "TAIL 17", "COMBINED")


def build(root):
    cards = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for name, samples in SAMPLES.items():
        records = [p for p in read_json(root/"pairs.json") if p["window"] == name]
        start, end = records[0]["original_frames_inclusive"]
        keys = records[0]["original_keyframes"]
        count = end-start+1
        runs = [root/name/c for c in CASES]
        clips = [root/"inputs"/name/"source.mp4", *[r/"receiver/reconstruction/sample_0000.mp4" for r in runs]]
        for path in clips:
            info = probe(path)
            if info["frames"] != count or info["fps"] != 24 or (info["width"], info["height"]) != (576, 320):
                raise ValueError(f"comparison video time/shape mismatch: {path}")
        dest = root/name
        command = ["ffmpeg", "-v", "error", "-y", "-nostdin"]
        for path in clips:
            command += ["-threads", "2", "-i", str(path)]
        filters = [f"[{i}:v]pad=iw:ih+32:0:32:black,drawtext=text='{label}':x=8:y=6:fontsize=18:fontcolor=white[v{i}]"
                   for i, label in enumerate(LABELS)]
        filters.append("[v0][v1][v2][v3][v4]xstack=inputs=5:layout=0_0|576_0|1152_0|576_352|1152_352:fill=black[out]")
        comparison = dest/"comparison.mp4"
        command += ["-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[out]",
                    "-an", "-threads", "2", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(comparison)]
        subprocess.run(command, check=True)
        if probe(comparison)["frames"] != count:
            raise ValueError("comparison output duration changed")
        decoded = [frames(p) for p in clips]
        if any(len(v) != count for v in decoded):
            raise ValueError("comparison decode frame count differs")
        sample_paths = []
        positions = ((0,0), (576,0), (1152,0), (576,352), (1152,352))
        for frame in samples:
            canvas = Image.new("RGB", (1728,704), "#121922")
            draw = ImageDraw.Draw(canvas)
            for label, values, (x,y) in zip(LABELS, decoded, positions):
                draw.text((x+8,y+6), label, font=font, fill="white")
                canvas.paste(values[frame-start], (x,y+32))
            draw.multiline_text((12,378), f"{name}\nOriginal frame {frame}\nTime {frame/24:.4f}s\nSame received data and replayed noise\nShort restarted context; AI review sample", font=font, fill="white", spacing=12)
            filename = f"sample_{frame:04d}.jpg"
            canvas.save(dest/filename, quality=95)
            sample_paths.append(filename)
        tables = []
        for run in runs:
            with (run/"quality_delivered_mp4.csv").open() as stream:
                tables.append(list(csv.DictReader(stream)))
        if any(len(t) != count for t in tables):
            raise ValueError("quality rows do not cover every output frame")
        rows = []
        for local in range(count):
            row = {"original_frame": local+start, "original_time_s": (local+start)/24}
            for case, table in zip(CASES, tables):
                if int(table[local]["frame"]) != local:
                    raise ValueError("quality rows out of order")
                row.update({f"{case}_{m}": float(table[local][m]) for m in METRICS})
            rows.append(row)
        with (dest/"four_condition_quality.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries = {}
        for scope, selected in {"whole_window": rows, "target_intermediate_frames":
                [r for r in rows if keys[2] < r["original_frame"] < keys[3]]}.items():
            means = {c: {m: sum(r[f"{c}_{m}"] for r in selected)/len(selected) for m in METRICS} for c in CASES}
            summaries[scope] = {"frames": len(selected), "metrics": means,
                "interaction_combined_minus_tail_minus_rounding_plus_baseline": {
                    m: means["combined"][m]-means["tail17"][m]-means["no_rounding"][m]+means["baseline"][m] for m in METRICS}}
        write_json(dest/"evidence.json", {"window": name, "quality": summaries,
            "original_frames_inclusive": [start,end], "sample_frames": samples, "sample_images": sample_paths,
            "comparison_sha256": sha256(comparison), "comparison_video": probe(comparison),
            "semantic_review": "Separate AI observations are not independent ground truth or full-video error rates."})
        links = " ".join(f'<a href="{name}/{p}">{f/24:.3f}s</a>' for f,p in zip(samples,sample_paths))
        buttons = " ".join(f'<button data-time="{(f-start)/24:.8f}">{f/24:.3f}s</button>' for f in samples)
        metric_rows = "".join(f'<tr><td>{c}</td><td>{summaries["whole_window"]["metrics"][c]["psnr_db"]:.3f}</td><td>{summaries["whole_window"]["metrics"][c]["lpips_vgg"]:.3f}</td></tr>' for c in CASES)
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>원본 프레임 {start}–{end}, {start/24:.3f}–{end/24:.3f}초</p>'
            f'<video controls loop preload="metadata" src="{name}/comparison.mp4"></video><p>{buttons}</p>'
            '<p><button data-step="-1">이전 프레임</button> <button data-step="1">다음 프레임</button> '
            '<select><option value="0.25">0.25배속</option><option value="1">1배속</option></select></p>'
            f'<table><tr><th>조건</th><th>PSNR ↑</th><th>LPIPS ↓</th></tr>{metric_rows}</table>'
            f'<p>비교 이미지: {links}</p><p><a href="{name}/four_condition_quality.csv">프레임별 지표</a></p></section>')
    page = '''<!doctype html><meta charset="utf-8"><title>ETRI 네 조건 비교</title>
<style>body{background:#101720;color:#e4eaf1;font:16px sans-serif;max-width:1600px;margin:30px auto;padding:20px}section{border-top:1px solid #465569;padding:20px 0}video{width:100%}a{color:#8dd1ff}button,select{padding:8px;margin:3px}td,th{padding:7px 20px;border-bottom:1px solid #465569}</style>
<h1>17프레임 참조 × 반올림 해제</h1><p>위: 원본 / 기존 방식 / 반올림 해제 · 아래: 마지막 17프레임 참조 / 두 변경 결합</p>
<p>개발 영상 한 편의 짧은 3개 창, 동일 수신 데이터·생성 잡음. 60초 전체 완화 결과가 아닙니다. 지표 개선과 의미 오류 감소는 별도로 판단합니다.</p>
<p><a href="RESULT.json">실행 결과</a> · <a href="../../docs/ETRI_COMBINED_REFERENCE_DIAGNOSIS.md">보고서</a></p>'''+"".join(cards)+'''
<script>document.querySelectorAll('section').forEach(s=>{const v=s.querySelector('video');v.playbackRate=.25;s.querySelectorAll('button').forEach(b=>b.onclick=()=>{v.pause();v.currentTime=b.dataset.time!==undefined?Number(b.dataset.time):Math.max(0,v.currentTime+Number(b.dataset.step)/24)});s.querySelector('select').onchange=e=>v.playbackRate=Number(e.target.value)})</script>'''
    (root/"review.html").write_text(page)
