"""Audit unique-frame comparisons and produce offline WebVid review artifacts."""
import html
import math
import shutil
import subprocess

from .ablation_transport import densify, subdivide_metadata
from .artifacts import sha256, write_json
from .official_quality import KEYS
from .temporal import output_source_indices, unique_output_positions
from .video_io import probe
from .webvid5 import read_json
from .wire import unpack

CASES = ("baseline", "aligned", "clean_keys", "dense_1s")
LABELS = {"baseline": "기존 LGVSC 설정", "aligned": "끝점·오버랩 정렬 보정",
          "clean_keys": "정렬 + 원본 키프레임 (진단)", "dense_1s": "정렬 + 최대 1초 간격"}


def collect(root):
    protocol = read_json(root / "protocol.json")
    cfg, selection = protocol["config"], protocol["selection"]
    base = root / "baseline"
    indices = read_json(base / "keyframes.json")["indices"]
    if indices[0] != 0 or indices[-1] != cfg["frames"] - 1:
        raise ValueError("baseline keyframes do not span source")
    source = base / "data/normalized.mp4"
    if sha256(source) != selection["processed_sha256"] or sha256(cfg["input"]) != sha256(source):
        raise ValueError("source changed or was not preserved")
    baseline_uses = read_json(base / "channel_accounting.json")["total_complex_channel_uses"]
    rows = []
    for case in CASES:
        run = root / case
        current_cfg = read_json(run / "run_config.json")
        if any(current_cfg[k] != cfg[k] for k in ("frames", "width", "height", "fps", "seed", "steps", "snr_db", "models")):
            raise ValueError(f"comparison settings changed: {case}")
        inputs = read_json(run / "receiver/decoder_inputs.json")
        expected_keys = densify(indices, round(cfg["fps"])) if case == "dense_1s" else indices
        if inputs["indices"] != expected_keys:
            raise ValueError(f"keyframe pairing changed: {case}")
        concat = inputs["decoder"]["concatenation_policy"]
        if concat != ("official_release" if case == "baseline" else "endpoint_exact"):
            raise ValueError(f"unexpected concatenation: {case}")
        positions = unique_output_positions(expected_keys, concat)
        mapping = output_source_indices(expected_keys, concat)
        if [mapping[p] for p in positions] != list(range(cfg["frames"])):
            raise ValueError("not a complete unique source timeline")
        video = run / "receiver/reconstruction/sample_0000.mp4"
        info = probe(video)
        if info["frames"] != len(mapping) or any(info[k] != cfg[k] for k in ("width", "height", "fps")):
            raise ValueError(f"output timeline mismatch: {case}")
        if len(list(video.with_name("sample_0000_frames").glob("*.png"))) != len(mapping):
            raise ValueError(f"lossless frame count mismatch: {case}")
        quality = read_json(root / "evaluations" / case / "quality.json")
        if (quality["status"] != "PASSED" or quality["evaluation_profile"] != "lgvsc_official_metrics_v1"
                or quality["video_sha256"] != sha256(video) or quality["reference_sha256"] != sha256(source)
                or quality["source_sha256"] != selection["processed_sha256"]):
            raise ValueError(f"evaluation provenance mismatch: {case}")
        view = quality["endpoint_exact_view"]
        if view["kept_generated_indices"] != positions or view["output_source_indices"] != list(range(cfg["frames"])):
            raise ValueError(f"evaluation positions mismatch: {case}")
        for boundary in ("lossless_frames", "delivered_mp4"):
            if view[boundary]["frames"] != cfg["frames"] or any(not math.isfinite(view[boundary][k]) for k in KEYS):
                raise ValueError(f"missing or nonfinite five-metric evaluation: {case}")
        for i in indices:
            ref = base / (f"data/frames/sample/{i}.png" if case == "clean_keys"
                          else f"receiver/frames/sample/key_frames_received/{i}.png")
            if sha256(run / f"receiver/frames/sample/key_frames_received/{i}.png") != sha256(ref):
                raise ValueError(f"paired keyframe PNG changed: {case}/{i}")
        if case in {"aligned", "clean_keys"} and (run / "receiver/metadata.csv").read_bytes() != (base / "receiver/metadata.csv").read_bytes():
            raise ValueError(f"caption/flow control changed: {case}")
        if case == "dense_1s":
            expected = subdivide_metadata(indices, expected_keys, read_json(base / "metadata_tx.json"))
            if read_json(run / "metadata_tx.json") != expected:
                raise ValueError("dense condition changed automatic captions/flow")
        account = None
        if case != "clean_keys":
            account = read_json((run if case == "dense_1s" else base) / "channel_accounting.json")
            if account["status"] != "PASSED" or not account["metadata_exact_match"] or account["bit_errors"] != 0:
                raise ValueError(f"channel metadata failed: {case}")
        if case in {"baseline", "dense_1s"}:
            audit_wire(run, base)
        uses = account["total_complex_channel_uses"] if account else None
        rows.append(dict(case=case, label=LABELS[case], indices=expected_keys, output_frames=len(mapping),
                         kept_output_positions=positions, lossless_frames=view["lossless_frames"],
                         delivered_mp4=view["delivered_mp4"], channel=account,
                         total_complex_channel_uses=uses, rate_multiple=uses / baseline_uses if uses is not None else None,
                         clean_keyframe_oracle=case == "clean_keys"))
    return protocol, rows


def audit_wire(run, base):
    account = read_json(run / "channel_accounting.json")
    sender = read_json(run / "sender_accounting.json")
    for folder, files in (("transmitter", sender["transmitter_files"]), ("received", account["received_files"])):
        for name, record in files.items():
            if sha256(run / folder / name) != record["sha256"]:
                raise ValueError(f"transport hash changed: {run.name}/{folder}/{name}")
    if (run / "received/metadata.bin").read_bytes() != (run / "transmitter/metadata.bin").read_bytes():
        raise ValueError("received metadata differs from transmitter")
    header, _ = unpack((run / "received/metadata.bin").read_bytes())
    count = sum(k["complex_count"] for k in header["keyframes"])
    if ((run / "received/visual.c64").stat().st_size != count * 8
            or account["visual_complex_channel_uses"] != count
            or account["total_complex_channel_uses"] != count + account["digital_complex_channel_uses"]):
        raise ValueError("channel accounting disagrees with payload")
    old, _ = unpack((base / "transmitter/metadata.bin").read_bytes())
    lookup = {k["index"]: k for k in header["keyframes"]}
    for folder in ("transmitter", "received"):
        a, b = (base / folder / "visual.c64").read_bytes(), (run / folder / "visual.c64").read_bytes()
        for k in old["keyframes"]:
            new = lookup[k["index"]]
            s, t, n = k["complex_offset"] * 8, new["complex_offset"] * 8, k["complex_count"] * 8
            if new["complex_count"] != k["complex_count"] or a[s:s + n] != b[t:t + n]:
                raise ValueError("baseline visual symbol pairing changed")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-threads", "2", *map(str, args)], check=True)


def panels(root, names, labels, destination, *, grid):
    args, filters = [], []
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    for i, (name, label) in enumerate(zip(names, labels)):
        args.extend(["-i", root / "review_media" / f"{name}.mp4"])
        filters.append(f"[{i}:v]pad=iw:ih+32:0:32:color=white,drawtext=fontfile={font}:text='{label}':x=8:y=8:fontsize=16:fontcolor=black[p{i}]")
    inputs = "".join(f"[p{i}]" for i in range(len(names)))
    filters.append(inputs + ("xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]" if grid else f"hstack=inputs={len(names)}[out]"))
    ffmpeg(*args, "-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[out]",
           "-an", "-c:v", "libx264", "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", root / destination)


def build(root):
    protocol, rows = collect(root)
    cfg = protocol["config"]
    media = root / "review_media"
    media.mkdir()
    shutil.copyfile(root / "baseline/data/normalized.mp4", media / "source.mp4")
    for row in rows:
        video = root / row["case"] / "receiver/reconstruction/sample_0000.mp4"
        kept = set(row["kept_output_positions"])
        dropped = [i for i in range(row["output_frames"]) if i not in kept]
        if dropped:
            expression = "+".join(f"eq(n\\,{i})" for i in dropped)
            ffmpeg("-i", video, "-vf", f"select=not({expression}),setpts=N/({cfg['fps']}*TB)",
                   "-an", "-c:v", "libx264", "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", media / f"{row['case']}.mp4")
        else:
            shutil.copyfile(video, media / f"{row['case']}.mp4")
    panels(root, ["source", "baseline", "aligned", "dense_1s"],
           ["SOURCE", "BASELINE", "ALIGNED", "ALIGNED + MAX GAP 1s"], "comparison_four_panel.mp4", grid=True)
    panels(root, ["source", "aligned", "clean_keys"],
           ["SOURCE", "ALIGNED", "CLEAN KEYS - DIAGNOSTIC ONLY"], "comparison_oracle.mp4", grid=False)
    for path in [*media.glob("*.mp4"), root / "comparison_four_panel.mp4", root / "comparison_oracle.mp4"]:
        info = probe(path)
        if info["frames"] != cfg["frames"] or info["fps"] != cfg["fps"]:
            raise ValueError(f"review video timeline mismatch: {path}")
    write_json(root / "comparison_summary.json", {"scope": "DEVELOPMENT_PILOT", "rows": rows,
               "manual_semantic_review": "PENDING", "protocol_signature": protocol["signature"]})
    table = ["| 조건 | 키프레임 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | CLIP ↑ | 채널 사용량 | 배수 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    html_rows = []
    for row in rows:
        q = row["lossless_frames"]
        uses = f"{row['total_complex_channel_uses']:,}" if row["channel"] else "진단 전용"
        rate = f"{row['rate_multiple']:.3f}" if row["channel"] else "—"
        values = [row["label"], str(len(row["indices"])), *[f"{q[k]:.4f}" for k in ("psnr_db", "ssim", "lpips_vgg", "dists", "clip")], uses, rate]
        table.append("| " + " | ".join(values) + " |")
        html_rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    lines = ["# WebVid 한 편: 네 조건 비교", "", protocol["selection"]["filename"], "",
             f"{cfg['frames']}프레임 · {cfg['fps']}fps · {cfg['width']}×{cfg['height']} · seed {cfg['seed']} · {cfg['steps']} steps · {cfg['snr_db']}dB", "",
             "[동기 비교 페이지](comparison.html) · [4분할 비교](comparison_four_panel.mp4) · [원본 키프레임 진단](comparison_oracle.mp4) · [상세 JSON](comparison_summary.json)", "",
             "## 같은 원본 시점의 화질과 전송량", "",
             "PNG 기준. 기존 설정의 중복 경계 프레임을 제거하여 모든 조건을 동일한 원본 시점으로 평가했다. MP4 지표와 프레임별 CSV는 evaluations/ 아래에 별도로 보관한다.", "", *table, "",
             "## 해석 범위", "",
             "- 한 개발 영상·한 시드의 비교이며 학습하거나 모델 가중치를 바꾸지 않았다.",
             "- 자동 캡션과 광류를 고정했다. 1초 간격 조건은 원래 구간의 캡션·광류를 세부 구간에 반복한다.",
             "- 원본 키프레임 조건은 시각 코덱/채널을 우회한 진단이다. 전송 성능 순위에서 제외한다.",
             "- 정렬은 실제 끝 잠재 프레임 조건과 마지막 17프레임 overlap, 중복 경계 제거를 함께 적용한다. 픽셀을 그대로 붙이는 처리는 아니다.",
             "- 키프레임 추가 조건은 원래 송수신 심벌과 수신 PNG를 보존한다. 추가 프레임은 독립 AWGN을 적용하고 전체 디지털 메타데이터를 다시 전송한다.",
             "- 전송량은 시각 심벌과 LDPC/16QAM 메타데이터를 합친 복소 채널 사용 횟수다. 물리 링크 헤더는 제외한다.",
             "- 같은 시드라도 분할과 overlap 길이가 달라지면 난수 소비가 달라진다. 서로 다른 분할 간 동일 latent noise를 보장하지 않는다.",
             "- 이 결과만으로 일반화나 동일 전송량 우월성을 주장하지 않는다. ETRI 전용 색상 추적 지표는 적용하지 않았다.", "",
             "## 동작 검수 — 미완료", "",
             protocol["selection"]["review_focus"], "",
             "비교 페이지에서 등장 객체 수·이동 방향·회전/동작 순서·시점·배경 변형을 직접 확인한다. 자동 계산 완료가 의미적 정확성 검수 완료를 뜻하지 않는다.", "",
             "## 재실행", "",
             "같은 실행 명령은 완료 단계의 해시를 검증하고 재사용한다. 실패 단계의 부분 파일은 failed_attempts/로 옮기고 그 단계를 처음부터 실행한다. SKEM 비교 도중 중단된 경우 선택 단계 전체를 다시 실행한다.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    options = "".join(f'<option value="{c}" {"selected" if c == "dense_1s" else ""}>{html.escape(LABELS[c])}</option>' for c in CASES if c != "baseline")
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>WebVid 네 조건 비교</title>
<style>body{font:16px sans-serif;margin:24px;background:#f5f6f8;color:#18202c}section{display:flex;gap:12px}figure{margin:0;flex:1}video{width:100%;background:#111}table{border-collapse:collapse;margin-top:20px}th,td{padding:8px;border:1px solid #bbb}button,select{padding:8px;margin:5px}input{width:50%}</style>
<h1>WebVid 한 편 비교</h1><p>__NAME__ · 동일한 원본 시점으로 정렬. 원본 키프레임은 진단 조건입니다.</p>
<button id="play">동시 재생</button><button id="pause">일시 정지</button><select id="variant">__OPTIONS__</select>
<input id="seek" type="range" min="0" max="__LAST__" value="0" step="1"><span id="frame">0</span>
<section><figure><figcaption>원본</figcaption><video id="source" src="review_media/source.mp4" muted playsinline preload="metadata"></video></figure>
<figure><figcaption>기존 LGVSC</figcaption><video id="baseline" src="review_media/baseline.mp4" muted playsinline preload="metadata"></video></figure>
<figure><figcaption id="chosenLabel">정렬 + 최대 1초 간격</figcaption><video id="chosen" src="review_media/dense_1s.mp4" muted playsinline preload="metadata"></video></figure></section>
<table><thead><tr><th>조건</th><th>키프레임</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>DISTS ↓</th><th>CLIP ↑</th><th>채널 사용량</th><th>배수</th></tr></thead><tbody>__ROWS__</tbody></table>
<p>한 개발 영상·한 시드. 키프레임 추가는 전송량도 증가합니다. 자동 지표 계산 완료이며 동작의 의미적 정확성은 직접 검수해야 합니다.</p>
<p><a href="REPORT.md">보고서</a> · <a href="comparison_four_panel.mp4">4분할 영상</a> · <a href="comparison_oracle.mp4">원본 키프레임 진단 영상</a></p>
<script>
const videos=[...document.querySelectorAll('video')], source=document.getElementById('source'), chosen=document.getElementById('chosen'), seek=document.getElementById('seek'), fps=__FPS__;
const pause=()=>videos.forEach(v=>v.pause());
document.getElementById('pause').onclick=pause;
document.getElementById('play').onclick=()=>{videos.forEach(v=>{v.currentTime=source.currentTime;v.play().catch(()=>{});});};
seek.oninput=()=>{pause();videos.forEach(v=>v.currentTime=Number(seek.value)/fps);document.getElementById('frame').textContent=seek.value;};
document.getElementById('variant').onchange=e=>{const t=source.currentTime;pause();chosen.onloadedmetadata=()=>{chosen.currentTime=t;};chosen.src='review_media/'+e.target.value+'.mp4';document.getElementById('chosenLabel').textContent=e.target.selectedOptions[0].text;};
source.ontimeupdate=()=>{seek.value=Math.round(source.currentTime*fps);document.getElementById('frame').textContent=seek.value;};
source.onended=pause;
setInterval(()=>{if(!source.paused)videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-source.currentTime)>0.12)v.currentTime=source.currentTime;});},250);
</script></html>'''
    for key, value in {"__NAME__": html.escape(protocol["selection"]["filename"]), "__OPTIONS__": options,
                       "__LAST__": str(cfg["frames"] - 1), "__FPS__": str(cfg["fps"]), "__ROWS__": "".join(html_rows)}.items():
        page = page.replace(key, value)
    (root / "comparison.html").write_text(page)
    write_json(root / "AUDIT.json", {"status": "PASSED", "source_preserved": True,
               "unique_frames_per_condition": cfg["frames"], "original_visual_blocks_and_pngs_paired": True,
               "all_five_metrics": True, "manual_semantic_review": "PENDING"})
