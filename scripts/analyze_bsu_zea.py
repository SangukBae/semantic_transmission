"""Compare completed Zea outputs; never start or change reconstruction."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
import time

from semantic_transmission.artifacts import sha256, write_json

REPO = Path(__file__).resolve().parents[1]
FIELDS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")
LABELS = {"lgvsc": "LGVSC 공개 정책 기준선", "fixed": "BSU 기존 고정 22키", "bsu": "BSU 현재 의미 변화 선택"}
START = time.monotonic()


def read(path):
    return json.loads(Path(path).read_text())


def progress(stage, done, total):
    elapsed = time.monotonic() - START
    print(f"{stage}: {done}/{total} ({done / total * 100:.1f}%) | 경과 {elapsed:.1f}초", flush=True)


def mapped_rows(rows, mapping):
    """Keep the first generated occurrence of each source time, explicitly."""
    if [row["frame"] for row in rows] != list(range(len(mapping))):
        raise ValueError("quality rows do not cover the audited output timeline")
    if not mapping or any(type(i) is not int or i < 0 for i in mapping) or mapping != sorted(mapping):
        raise ValueError("invalid source timeline")
    result = {}
    for row, source_index in zip(rows, mapping):
        result.setdefault(source_index, row)
    return result


def mean_rows(rows, indices):
    if not indices or any(i not in rows for i in indices):
        raise ValueError("missing common quality frame")
    result = {k: statistics.mean(rows[i][k] for i in indices) for k in FIELDS}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite quality")
    return dict(result, frames=len(indices))


def quality_rows(run, video_hash, source_hash):
    quality = read(run / "quality.json")
    if (quality["status"] != "PASSED" or quality["video_sha256"] != video_hash
            or quality["reference_sha256"] != source_hash
            or quality["evaluation_profile"] != "lgvsc_official_metrics_v1"):
        raise ValueError("quality provenance mismatch")
    with (run / "quality_delivered_mp4.csv").open() as stream:
        rows = [{k: int(v) if k == "frame" else float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    if any(abs(statistics.mean(r[k] for r in rows) - quality["delivered_mp4"][k]) > 1e-9 for k in FIELDS):
        raise ValueError("quality CSV no longer matches its completed receipt")
    return rows


def analyze(root, historical, output):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from semantic_transmission.research_quality import read_video
    from semantic_transmission.receiver_alignment_evaluation import comparison_video
    from semantic_transmission.etri_60s_check import pts_audit

    cv2.setNumThreads(2)
    old = read(historical / "RESULT.json")
    current = read(root / "RESULT.json")
    if not current["status"].startswith("COMPLETE_RECONSTRUCTION"):
        raise ValueError("BSU reconstruction is not complete")
    plan = read(root / "plan.json")
    protocol = read(root / "protocol.json")
    if plan.get("selection_policy") != "semantic" or plan["key_budget"] is not None or plan["max_gap"] is not None:
        raise ValueError("expected current variable-count BSU policy")
    runs = {"lgvsc": historical / "lgvsc/run", "fixed": historical / "qwen/reconstruction/event_pairs/run",
            "bsu": root / "reconstruction/event_pairs/run"}
    paths = {"source": Path(old["source"]), "lgvsc": Path(old["videos"]["lgvsc"]),
             "fixed": Path(old["videos"]["qwen"]), "bsu": Path(current["video"])}
    hashes = {k: sha256(p) for k, p in paths.items()}
    if (hashes["source"] != old["source_sha256"] or hashes["source"] != plan["source"]["normalized_sha256"]
            or sha256(plan["source"]["normalized_video"]) != hashes["source"]
            or hashes["lgvsc"] != old["video_sha256"]["lgvsc"] or hashes["fixed"] != old["video_sha256"]["qwen"]):
        raise ValueError("source or historical video changed")
    audits = {k: read(run / "output_audit.json") for k, run in runs.items()}
    maps = {}
    for k, audit in audits.items():
        if audit["status"] != "PASSED" or audit["video_sha256"] != hashes[k]:
            raise ValueError(f"output audit mismatch: {k}")
        maps[k] = audit.get("output_source_indices", audit.get("source_indices"))
        if not maps[k]:
            raise ValueError(f"missing temporal mapping: {k}")
    keys = {"lgvsc": old["lgvsc"]["keys"], "fixed": old["qwen"]["keys"], "bsu": protocol["keys"]["event_pairs"]}
    if len(keys["fixed"]) != 22:
        raise ValueError("historical comparator is not the requested 22-key run")
    cfg = {k: read(run / "run_config.json") for k, run in runs.items()}
    shared = ("fps", "width", "height", "snr_db", "channel_seed", "seed", "steps", "models")
    for field in shared:
        if not all(cfg[k][field] == cfg["bsu"][field] for k in runs):
            raise ValueError(f"comparison configuration differs: {field}")
    if cfg["bsu"]["caption_adapter"] != cfg["fixed"]["caption_adapter"]:
        raise ValueError("BSU caption adapter differs from historical fixed-budget comparator")
    cache = read(historical / "review/lgvsc_quality.json")
    if (cache["sha256"] != {"source": hashes["source"], "lgvsc": hashes["lgvsc"], "qwen": hashes["fixed"]}
            or cache["metric_code_sha256"] != sha256(REPO / "src/semantic_transmission/official_quality.py")):
        raise ValueError("cached LGVSC metric provenance differs")
    raw = {"lgvsc": cache["rows"], **{k: quality_rows(runs[k], hashes[k], hashes["source"]) for k in ("fixed", "bsu")}}
    mapped = {k: mapped_rows(raw[k], maps[k]) for k in runs}
    common = sorted(set.intersection(*(set(v) for v in mapped.values())))
    if common != list(range(381)):
        raise ValueError("expected common Zea source frames 0..380")
    nonkey = sorted(set(common) - set().union(*(set(v) for v in keys.values())))
    quality = {k: dict(common_all=mean_rows(mapped[k], common), common_nonkey=mean_rows(mapped[k], nonkey)) for k in runs}
    evidence = [historical / "RESULT.json", historical / "review/lgvsc_quality.json", root / "RESULT.json",
                root / "plan.json", root / "protocol.json", Path(__file__), REPO / "src/semantic_transmission/official_quality.py"]
    for k, run in runs.items():
        evidence += [run / n for n in ("output_audit.json", "run_config.json", "channel_accounting.json")]
        evidence.append(run.parent / "resources/reconstruct.json")
        if k != "lgvsc":
            evidence += [run / "quality.json", run / "quality_delivered_mp4.csv"]
    frozen = {str(p): sha256(p) for p in evidence}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inputs.json", dict(video_sha256=hashes, evidence_sha256=frozen, common_source_indices=common))
    progress("분석 전체 단계", 1, 5)
    pixels = {k: read_video(p) for k, p in paths.items()}
    positions = {k: {i: maps[k].index(i) for i in common} for k in runs}
    for k in runs:
        if len(pixels[k]) != len(maps[k]):
            raise ValueError("decoded output length differs from audit")
    pixels = {k: (v[common] if k == "source" else v[[positions[k][i] for i in common]]) for k, v in pixels.items()}
    if any(v.shape != (381, 320, 576, 3) for v in pixels.values()):
        raise ValueError("aligned video dimensions differ")
    gray = {k: [cv2.cvtColor(cv2.resize(f, (288, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in a]
            for k, a in pixels.items()}
    transitions = {k: [] for k in pixels}
    for i in range(1, len(common)):
        flow = {k: cv2.calcOpticalFlowFarneback(a[i-1], a[i], None, .5, 3, 15, 3, 5, 1.2, 0) for k, a in gray.items()}
        for k, a in gray.items():
            transitions[k].append(dict(to_frame=common[i], seconds=common[i]/24,
                mad=float(np.abs(a[i].astype(float)-a[i-1]).mean()),
                flow_error_to_source=float(np.linalg.norm(flow[k]-flow["source"], axis=2).mean()),
                flow_magnitude=float(np.linalg.norm(flow[k], axis=2).mean()),
                exact_rgb_duplicate=int(np.array_equal(pixels[k][i], pixels[k][i-1]))))
        if i % 48 == 0 or i == len(common)-1:
            progress("움직임 비교", i, len(common)-1)
    temporal = {}
    for k, rows in transitions.items():
        mad = np.array([r["mad"] for r in rows])
        src = np.array([r["mad"] for r in transitions["source"]])
        temporal[k] = dict(mad_mean=float(mad.mean()), mad_p95=float(np.percentile(mad, 95)), mad_max=float(mad.max()),
            mad_error_to_source=float(np.abs(mad-src).mean()),
            flow_error_to_source=statistics.mean(r["flow_error_to_source"] for r in rows),
            flow_magnitude_mean=statistics.mean(r["flow_magnitude"] for r in rows),
            exact_rgb_duplicates=sum(r["exact_rgb_duplicate"] for r in rows),
            largest_mad_transitions=sorted(rows, key=lambda r:r["mad"], reverse=True)[:5])
    write_json(output / "transitions.json", transitions)
    progress("분석 전체 단계", 2, 5)
    channels = {}
    timing = {}
    intervals = {}
    for k, run in runs.items():
        c = read(run / "channel_accounting.json")
        if c["status"] != "PASSED" or c["bit_errors"] or not c["metadata_exact_match"]:
            raise ValueError("failed channel audit")
        channels[k] = {n:c[n] for n in ("visual_complex_channel_uses", "digital_complex_channel_uses", "total_complex_channel_uses")}
        resource = read(run.parent / "resources/reconstruct.json")
        if resource.get("returncode", 0) != 0:
            raise ValueError("unsuccessful reconstruction resource receipt")
        timing[k] = dict(reconstruction_seconds=resource["seconds"],
                        scope="Measured reconstruction worker only; excludes selection, captions, T5, channel and evaluation; one observation.")
        gaps = [(b-a)/24 for a,b in zip(keys[k][:-1], keys[k][1:])]
        intervals[k] = dict(count=len(gaps), min_seconds=min(gaps), median_seconds=statistics.median(gaps), max_seconds=max(gaps), seconds=gaps)
    regular = list(range(0, 381, 24)) + [380]
    adaptive = sorted({r["to_frame"] for k in runs for r in temporal[k]["largest_mad_transitions"][:3]})
    samples = output / "samples"
    samples.mkdir(exist_ok=True)
    names = list(pixels)
    def sheet(name, ids):
        canvas = Image.new("RGB", (576*len(names), 344*len(ids)), "#161b22")
        draw = ImageDraw.Draw(canvas)
        for row, index in enumerate(ids):
            for col, k in enumerate(names):
                canvas.paste(Image.fromarray(pixels[k][index]), (col*576, row*344+24))
                draw.text((col*576+8, row*344+5), f"{k.upper()} frame {index}  {index/24:.3f}s", fill="white")
        canvas.save(samples / name, quality=94)
    for n, offset in enumerate(range(0, len(regular), 3)):
        sheet(f"regular_{n:02d}.jpg", regular[offset:offset+3])
    for index in adaptive:
        sheet(f"transition_{index:03d}.jpg", [index-1, index])
    comparison_video(output / "comparison.mp4", list(pixels.values()), ["SOURCE", "LGVSC", "BSU fixed 22", "BSU semantic"], 24)
    pts_audit(output / "comparison.mp4", len(common), 24)
    progress("분석 전체 단계", 3, 5)
    changes = {}
    for before in ("lgvsc", "fixed"):
        a, b = quality[before]["common_all"], quality["bsu"]["common_all"]
        changes[before] = {k: b[k]-a[k] for k in FIELDS}
        for field in ("flow_error_to_source", "mad_p95", "mad_max"):
            changes[before][field+"_percent"] = 100*(temporal["bsu"][field]/temporal[before][field]-1)
        changes[before]["channel_uses_percent"] = 100*(channels["bsu"]["total_complex_channel_uses"]/channels[before]["total_complex_channel_uses"]-1)
    limitations = [
        "항구 개발 영상 1편·한 시드·SNR 10dB 결과이며 다른 영상으로 일반화하지 않습니다.",
        "LGVSC는 공개 선택·캡션·수신부 정책을 로컬 실행기에 연결한 기준선이며 논문 수치의 완전 재현은 아닙니다.",
        "화질·움직임은 세 출력에 공통인 원본 0~380프레임으로 비교합니다. BSU 두 출력의 마지막 3장은 공통 지표에서 제외합니다.",
        "전송량은 각 실행의 전체 실제 사용량입니다. LGVSC 출력 381장과 BSU 출력 384장의 길이 차이를 함께 기록합니다.",
        "BSU 간 선택 정책·키 위치·캡션·구간 길이가 함께 달라지므로 키 개수만의 독립 효과는 아닙니다.",
        "생성 seed는 같지만 구간 모양·난수 정책이 달라 생성 잡음 텐서까지 동일하지 않습니다.",
        "MAD가 낮아지는 것은 정지 영상에서도 가능하므로 원본 대비 움직임 오차·화질·시각 검수를 함께 봐야 합니다.",
        "시각 표본 검수는 AI 진단이며 독립적인 사람 정답·전체 의미 오류율을 대신하지 않습니다."]
    result = dict(status="QUANTITATIVE_COMPLETE_VISUAL_REVIEW_PENDING", labels=LABELS, common_frames=common,
        nonkey_frames=nonkey, keys=keys, intervals=intervals, quality=quality, temporal=temporal,
        transmission=channels, timing=timing, changes_current_vs_baseline=changes, output_frames={k:len(v) for k,v in maps.items()},
        shared_settings={k:cfg["bsu"][k] for k in shared}, caption_adapter=cfg["bsu"]["caption_adapter"]["id"],
        videos={k:str(v) for k,v in paths.items()}, video_sha256=hashes,
        visual_samples=dict(regular=regular, adaptive_pairs=[[i-1,i] for i in adaptive]),
        limitations=limitations, independent_semantic_review="PENDING", email_sent=False,
        requested_recipient="sangukbae99@gmail.com", generation_modified=False, elapsed_seconds=time.monotonic()-START)
    if frozen != {str(p):sha256(p) for p in evidence} or hashes != {k:sha256(p) for k,p in paths.items()}:
        raise ValueError("inputs changed during comparison")
    write_json(output / "RESULT.json", result)
    progress("분석 전체 단계", 4, 5)
    render(output, result)
    progress("분석 전체 단계", 5, 5)


def render(output, result):
    rows = []
    for k, label in LABELS.items():
        q,t,c = result["quality"][k]["common_all"],result["temporal"][k],result["transmission"][k]
        rows.append(f'<tr><td>{label}</td><td>{len(result["keys"][k])}</td><td>{q["psnr_db"]:.3f}</td>'
                    f'<td>{q["ssim"]:.4f}</td><td>{q["lpips_vgg"]:.4f}</td><td>{q["clip"]:.4f}</td><td>{q["dists"]:.4f}</td>'
                    f'<td>{t["flow_error_to_source"]:.6f}</td><td>{t["mad_p95"]:.3f}</td><td>{c["total_complex_channel_uses"]:,}</td>'
                    f'<td>{result["timing"][k]["reconstruction_seconds"]/60:.2f}</td></tr>')
    doc = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU 항구 영상 3종 비교</title>'
    doc += '<style>body{font:16px system-ui;max-width:1600px;margin:24px auto;padding:0 16px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}video,img{width:100%}pre{white-space:pre-wrap}</style>'
    doc += '<h1>BSU 항구 영상 비교</h1><p>왼쪽부터 원본 / LGVSC / BSU 고정 22키 / BSU 의미 변화 선택. 공통 381프레임·24fps.</p>'
    doc += '<video controls src="comparison.mp4"></video><table><tr><th>모델</th><th>키</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>CLIP ↑</th><th>DISTS ↓</th><th>움직임 오차 ↓</th><th>MAD p95</th><th>채널 사용량</th><th>생성 시간(분)</th></tr>' + ''.join(rows) + '</table><p>생성 시간은 복원 작업만의 실측이며 선택·캡션·T5·송수신·평가 시간을 제외합니다.</p>'
    review = output / "AI_REVIEW.json"
    doc += '<h2>시각·의미 검수</h2><pre>' + html.escape(review.read_text() if review.exists() else '시각 검수 대기 중입니다. 수치만으로 의미 품질 우위를 판정하지 않습니다.') + '</pre>'
    doc += '<h2>비교 범위</h2><ul>' + ''.join('<li>'+html.escape(s)+'</li>' for s in result['limitations'])+'</ul>'
    doc += '<h2>같은 시점의 장면</h2>' + ''.join(f'<img loading="lazy" src="samples/{p.name}">' for p in sorted((output/'samples').glob('*.jpg')))
    doc += '<p><a href="RESULT.json">수치·출처 기록</a></p></html>'
    (output / "report.html").write_text(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "outputs/bsu_zea_16s_20261009")
    parser.add_argument("--historical", type=Path, default=REPO / "outputs/lgvsc_qwen_zea_16s_20261008")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    output = args.root.resolve() / "comparison_lgvsc_fixed"
    if args.render_only:
        render(output, read(output / "RESULT.json"))
    else:
        analyze(args.root.resolve(), args.historical.resolve(), output)


if __name__ == "__main__":
    main()
