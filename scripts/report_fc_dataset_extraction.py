#!/usr/bin/env python3
"""Report actual source preparation, selected keyframes, and frozen captions."""
import argparse
from datetime import datetime, timezone
import html
from pathlib import Path

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.assisted_captions import validate_bundle
from semantic_transmission.hybrid_selection import verify_selection
from semantic_transmission.webvid5 import read_json

REPO=Path(__file__).resolve().parents[1]


def report(root):
    manifest=read_json(root / "inventory.json")
    records=[]
    for source in manifest["videos"]:
        directory=root / source["dataset"].lower() / source["id"]
        extraction=directory / "extraction"
        record=dict(id=source["id"],dataset=source["dataset"],source_path=source["source_path"],
            source_sha256=source["source_sha256"],duration_seconds=source["source_duration_seconds"],
            benchmark_memberships=source["benchmark_memberships"],
            source_prepared=(directory / "source_prepared.json").exists(),
            status="PENDING_REVIEW",keyframes=0,captions=0)
        if (extraction / "selection_freeze.json").exists():
            protocol,selection=verify_selection(extraction)
            if protocol.get("raw_source_sha256") != source["source_sha256"]:
                raise ValueError("extraction belongs to another raw video")
            record.update(status="KEYFRAMES_COMPLETE_CAPTIONS_PENDING",keyframes=len(selection["indices"]),
                keyframes_path=str(extraction / "selected_frames"),selection_path=str(extraction / "selection.json"),
                skem_comparisons=selection["skem_comparisons"],max_gap_frames=selection["max_gap_frames"])
            caption_path=extraction / "assistant_captions/captions_bundle.json"
            if caption_path.exists():
                bundle=validate_bundle(caption_path,extraction)
                record.update(status="EXTRACTION_COMPLETE",captions=len(bundle["records"]),
                    captions_path=str(caption_path),captions_sha256=sha256(caption_path),author=bundle["author"])
                page=directory / "review.html"
                parts=[]
                for row in bundle["records"]:
                    start,end=row["start"],row["end_exclusive"]
                    parts.append(f'<article><h3>Segment {row["segment"]} | {start/24:.3f}–{end/24:.3f} s</h3>'
                        f'<img src="extraction/selected_frames/{start:05d}.png"><img src="extraction/selected_frames/{end:05d}.png">'
                        f'<p>{html.escape(row["text"])}</p><small>Caption source samples: {row["source_indices"]}</small></article>')
                page.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<style>body{font:16px system-ui;max-width:1180px;margin:30px auto}article{border-bottom:1px solid #ccc;padding:15px}img{width:48%}</style>'
                    f'<h1>{html.escape(source["source_id"])}</h1>'
                    f'<p>{record["keyframes"]} keyframes; {record["captions"]} captions. Source-derived offline assistant extraction. No reconstruction or independent hallucination validation.</p>'
                    '<p>Each pair below shows the segment boundary keyframes. Captions were authored from the four recorded interior samples.</p>'
                    +''.join(parts)+'</html>')
                record["review_path"]=str(page)
        records.append(record)
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),scope=manifest["scope"],
        videos=len(records),sources_prepared=sum(r["source_prepared"] for r in records),
        extractions_complete=sum(r["status"]=="EXTRACTION_COMPLETE" for r in records),
        keyframes=sum(r["keyframes"] for r in records),captions=sum(r["captions"] for r in records),
        reconstruction_started=False,hallucination_mitigation_verified=False,records=records)
    write_json(root / "extraction_status.json",result)
    rows=[]
    for r in records:
        label=html.escape(Path(r["source_path"]).name)
        if "review_path" in r:
            label=f'<a href="{Path(r["review_path"]).relative_to(root)}">{label}</a>'
        rows.append(f'<tr><td>{r["dataset"]}</td><td>{label}</td><td>{r["duration_seconds"]:.2f}</td>'
                    f'<td>{r["status"]}</td><td>{r["keyframes"]}</td><td>{r["captions"]}</td></tr>')
    (root / "index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>'
        '<h1>FC-LGVSC · WebVid / TVSum 전체 원본 추출</h1>'
        f'<p>원본 준비 {result["sources_prepared"]}/{result["videos"]}편 · 키프레임 및 캡션 추출 완료 {result["extractions_complete"]}/{result["videos"]}편</p>'
        '<p>완료한 영상 이름을 누르면 시간별 키프레임과 캡션을 볼 수 있습니다. 준비된 원본 이미지와 최종 키프레임은 서로 다른 산출물입니다.</p>'
        '<p>원본 관찰에 기반한 실험용 추출 결과이며, 복원 및 독립적인 할루시네이션 평가는 수행하지 않았습니다.</p>'
        '<table><tr><th>Dataset</th><th>Video</th><th>Seconds</th><th>Extraction status</th><th>Keyframes</th><th>Captions</th></tr>'
        +''.join(rows)+'</table></html>')
    print({k:v for k,v in result.items() if k not in {"records","scope"}},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=REPO / "outputs/fc_lgvsc_webvid_tvsum_full_20261001")
    report(parser.parse_args().output.resolve())
