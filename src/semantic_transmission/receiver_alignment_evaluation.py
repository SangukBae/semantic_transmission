"""Offline source-paired evaluation, kept outside receiver generation."""
import html
from pathlib import Path
import subprocess
import time

from .artifacts import sha256, write_json
from .webvid5 import read_json
from .research_quality import read_frames, read_video
from .official_quality import OfficialMetrics, summarize


def temporal_scores(source, generated, seam):
    import cv2
    import numpy as np
    def gray(frames):
        return [cv2.cvtColor(cv2.resize(f, (288, 160)), cv2.COLOR_RGB2GRAY) for f in frames]
    a, b = gray(source), gray(generated)
    source_mad, generated_mad, error, source_motion, generated_motion = [], [], [], [], []
    for i in range(1, len(a)):
        flows = [cv2.calcOpticalFlowFarneback(x[i-1], x[i], None, .5, 3, 15, 3, 5, 1.2, 0) for x in (a, b)]
        source_mad.append(float(np.abs(a[i].astype(float)-a[i-1]).mean()))
        generated_mad.append(float(np.abs(b[i].astype(float)-b[i-1]).mean()))
        error.append(float(np.linalg.norm(flows[0]-flows[1], axis=2).mean()))
        source_motion.append(float(np.linalg.norm(flows[0], axis=2).mean()))
        generated_motion.append(float(np.linalg.norm(flows[1], axis=2).mean()))
    nonseam = [v for i, v in enumerate(generated_mad, 1) if i != seam]
    return dict(flow_error_mean=float(np.mean(error)), source_flow_mean=float(np.mean(source_motion)),
        generated_flow_mean=float(np.mean(generated_motion)), seam_frame=seam,
        source_seam_mad=source_mad[seam-1], generated_seam_mad=generated_mad[seam-1],
        source_mean_mad=float(np.mean(source_mad)), generated_mean_mad=float(np.mean(generated_mad)),
        generated_nonseam_mad=float(np.mean(nonseam)),
        per_transition=dict(source_mad=source_mad, generated_mad=generated_mad, flow_error=error))


def comparison_video(path, arrays, labels, fps):
    import cv2
    import numpy as np
    height, width = arrays[0].shape[1:3]
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width*len(arrays)}x{height+24}", "-r", str(fps), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    try:
        for i in range(len(arrays[0])):
            canvas = np.zeros((height+24, width*len(arrays), 3), dtype=np.uint8)
            for j, (frames, label) in enumerate(zip(arrays, labels)):
                canvas[24:, j*width:(j+1)*width] = frames[i]
                cv2.putText(canvas, label, (j*width+8, 17), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"comparison encoder failed: {code}")


def evaluate(output, protocol, signature):
    import gc
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    gc.collect()
    torch.cuda.empty_cache()
    start = time.monotonic()
    arms = protocol["arms"]
    sources, generated, groups, sources_by_case, artifact_hashes = [], [], [], {}, {}
    for case in protocol["cases"]:
        path = Path(case["run"])/"data/normalized.mp4"
        source = read_video(path)[case["keys"][0]:case["keys"][-1]+1]
        sources_by_case[case["name"]] = source
        artifact_hashes[str(path)] = sha256(path)
        for arm in arms:
            directory = output/case["name"]/arm
            pixels = read_frames(directory/"frames")
            if source.shape != pixels.shape:
                raise ValueError("source/output window mismatch")
            groups.append((case, arm, len(source)))
            sources.append(source)
            generated.append(pixels)
    print(f"EVALUATE {len(groups)} arms, {sum(n for _, _, n in groups)} frames; source pixels used only here", flush=True)
    metrics = OfficialMetrics()
    def progress(done, total):
        if done == 1 or done % 49 == 0 or done == total:
            print(f"QUALITY {done}/{total} ({100*done/total:.1f}%) elapsed={time.monotonic()-start:.0f}s", flush=True)
    _, all_rows = metrics.evaluate(np.concatenate(sources), np.concatenate(generated), progress=progress)
    results, offset = [], 0
    for number, ((case, arm, n), source, pixels) in enumerate(zip(groups, sources, generated), 1):
        rows = [dict(row, frame=i) for i, row in enumerate(all_rows[offset:offset+n])]
        offset += n
        keys = [k-case["keys"][0] for k in case["keys"]]
        nonkeys = [r for r in rows if r["frame"] not in keys]
        temporal = temporal_scores(source, pixels, keys[1]+1)
        received = [np.asarray(Image.open(Path(case["run"])/f"receiver/frames/sample/key_frames_received/{k}.png").convert("RGB"))
                    for k in case["keys"]]
        endpoints = [dict(frame=k, psnr_received_db=float(cv2.PSNR(reference, pixels[k])))
                     for k, reference in zip(keys, received)]
        result = dict(case=case["name"], arm=arm, source_range=case["keys"][::2],
            all_frames=summarize(rows, source.shape), nonkey_frames=summarize(nonkeys, (len(nonkeys), *source.shape[1:])),
            temporal=temporal, received_key_fidelity=endpoints)
        results.append(result)
        write_json(output/case["name"]/arm/"metrics.json", dict(result, per_frame=rows))
        print(f"TEMPORAL {number}/{len(groups)} ({100*number/len(groups):.1f}%) {case['name']}/{arm}: "
              f"PSNR={result['all_frames']['psnr_db']:.3f} LPIPS={result['all_frames']['lpips_vgg']:.4f} "
              f"flow_error={temporal['flow_error_mean']:.4f} seam_MAD={temporal['generated_seam_mad']:.3f}", flush=True)
    print("EXPORT synchronized source / legacy / position-only / phase-conditioned comparisons", flush=True)
    for case in protocol["cases"]:
        directory = output/case["name"]
        arrays = [sources_by_case[case["name"]]] + [read_frames(directory/a/"frames") for a in arms]
        labels = ["Source (evaluation only)", "Legacy aligned", "Exact last (position only)", "Phase last (candidate)"]
        video_path = directory/"comparison.mp4"
        comparison_video(video_path, arrays, labels, protocol["generator"]["fps"])
        artifact_hashes[str(video_path)] = sha256(video_path)
        indices = [0, 12, 23, 24, 25, 36, 48]
        for frame in indices:
            panels = []
            for array, label in zip(arrays, labels):
                panel = np.zeros((344, 576, 3), dtype=np.uint8)
                panel[24:] = array[frame]
                cv2.putText(panel, f"{label} / {frame}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
                panels.append(panel)
            Image.fromarray(np.hstack(panels)).save(directory/f"comparison_frame_{frame:02d}.png")
    result = dict(status="PILOT_COMPLETE_INDEPENDENT_REVIEW_PENDING", signature=signature,
        training_executed=False, adopted=False, source_pixels_used_for_generation=False,
        output_pixel_interpolation=False, cases=results, evaluation_artifacts=artifact_hashes,
        evaluation_code_sha256=sha256(Path(__file__)), elapsed_seconds=time.monotonic()-start,
        limitations=["four 49-frame windows with restarted context and one seed",
                     "one internal seam per window; no full-video or long-range claim",
                     "Zea uses Qwen captions; three older cases retain their own frozen captions",
                     "independent Added/Missing/Distorted review pending; metrics alone do not establish semantic safety"])
    write_json(output/"results.json", result)
    body = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Receiver temporal conditioning pilot</title>",
        "<style>body{font:15px system-ui;max-width:1300px;margin:30px auto}video{width:100%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>",
        "<h1>Receiver temporal conditioning pilot</h1><p>Experimental; not adopted. No output pixel interpolation. "
        "Four short windows, one seed, restarted context. Independent semantic review pending.</p>"]
    for case in protocol["cases"]:
        body.append(f"<h2>{html.escape(case['name'])}: {html.escape(case['scene'])}</h2>"
                    f"<video controls loop src='{case['name']}/comparison.mp4'></video>"
                    "<p>Left to right: source / legacy / position only / VAE phase candidate.</p>"
                    "<table><tr><th>Arm</th><th>PSNR ↑</th><th>LPIPS ↓</th><th>Flow error ↓</th><th>Seam MAD</th></tr>")
        for row in results:
            if row["case"] == case["name"]:
                body.append(f"<tr><td>{row['arm']}</td><td>{row['all_frames']['psnr_db']:.3f}</td>"
                            f"<td>{row['all_frames']['lpips_vgg']:.4f}</td><td>{row['temporal']['flow_error_mean']:.4f}</td>"
                            f"<td>{row['temporal']['generated_seam_mad']:.3f}</td></tr>")
        body.append("</table>")
    body.append("<p>Lower motion or seam MAD alone can mean freezing; judge alongside source motion, fidelity and semantic review.</p></html>")
    (output/"report.html").write_text("\n".join(body))
    write_json(output/"status.json", {k: result[k] for k in ("status", "signature", "adopted")})
    print(f"DONE {result['status']}: {output/'report.html'}", flush=True)
