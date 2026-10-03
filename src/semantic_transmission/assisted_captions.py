"""Prepare source observations and import frozen assistant-authored captions.

No external API, PLLaVA inference, transmission or reconstruction is launched.
The author is the assistant in this conversation; this is an offline assisted
condition, not an independently blinded model benchmark or automatic captioner.
"""
import argparse
import html
from pathlib import Path

from .artifacts import sha256, write_json
from .hybrid_selection import DEFAULT_ROOT, verify_selection
from .webvid5 import fingerprint, read_json


def sample_indices(start, end):
    """Match PLLaVA's released four samples from a half-open [start,end) clip."""
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise ValueError("invalid caption interval")
    size = (end - start - 1) / 4
    return [start + int(size / 2) + round(size * i) for i in range(4)]


def expected_samples(protocol, selection):
    return [dict(segment=n, start=a, end_exclusive=b,
                 source_indices=sample_indices(a,b))
            for n,(a,b) in enumerate(zip(selection["indices"],selection["indices"][1:]))]


def prepare(root):
    from PIL import Image, ImageDraw, ImageFont
    protocol, selection = verify_selection(root)
    folder = root / "assistant_captions"
    samples = expected_samples(protocol, selection)
    identity = dict(selection_freeze_sha256=sha256(root / "selection_freeze.json"),
        input_sha256=protocol["input_sha256"], selection_root=str(root), samples=samples,
        source_frame_hashes={str(i):protocol["source_frame_hashes"][str(i)]
                            for r in samples for i in r["source_indices"]},
        sampling="PLLaVA four-sample indices, half-open intervals; source PNGs, not reconstructions",
        display="Each row is one segment, four columns in temporal order; native 576x320 reduced to 288x160 for overview.")
    path = folder / "samples.json"
    if path.exists():
        if read_json(path) != identity:
            raise ValueError("caption observation inputs changed")
    else:
        write_json(path, identity)
    sheets = folder / "sheets"
    sheets.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for start in range(0, len(samples), 4):
        rows = samples[start:start+4]
        canvas = Image.new("RGB", (1152, len(rows)*205), "#202020")
        draw = ImageDraw.Draw(canvas)
        for row, sample in enumerate(rows):
            y = row*205
            draw.text((4,y+2),f"SEG {sample['segment']:02d} | [{sample['start']},{sample['end_exclusive']}) | {sample['start']/24:.3f}-{sample['end_exclusive']/24:.3f}s",font=font,fill="white")
            for col,i in enumerate(sample["source_indices"]):
                x=col*288
                with Image.open(Path(protocol["source_frames"]) / f"{i}.png") as im:
                    canvas.paste(im.convert("RGB").resize((288,160)),(x,y+43))
                draw.text((x+4,y+23),f"frame {i}",font=font,fill="white")
        canvas.save(sheets / f"{start//4:02d}.jpg",quality=96)
    print(f"PREPARED {len(samples)} segments / {len(samples)*4} observations; no model inference")
    return folder


def validate_bundle(path, selection_root):
    bundle = read_json(path)
    payload = {k:v for k,v in bundle.items() if k != "checksum"}
    if bundle.get("checksum") != fingerprint(payload) or bundle.get("version") != 1:
        raise ValueError("assistant caption bundle checksum/version mismatch")
    protocol, selection = verify_selection(selection_root)
    if (bundle["selection_freeze_sha256"] != sha256(selection_root / "selection_freeze.json")
            or bundle["input_sha256"] != protocol["input_sha256"]):
        raise ValueError("assistant captions belong to a different source or selection")
    expected = expected_samples(protocol,selection)
    records = bundle["records"]
    if len(records) != len(expected):
        raise ValueError("assistant captions do not cover every segment")
    for row, wanted in zip(records, expected):
        if any(row[k] != v for k,v in wanted.items()):
            raise ValueError("assistant caption segment/sample alignment changed")
        if (not isinstance(row["text"],str) or not row["text"].strip()
                or "\n" in row["text"] or len(row["text"].split()) > 80):
            raise ValueError("caption must be one nonempty line of at most 80 words")
    hashes = {str(i):protocol["source_frame_hashes"][str(i)] for r in expected for i in r["source_indices"]}
    if bundle["source_frame_hashes"] != hashes:
        raise ValueError("assistant caption source-frame hashes changed")
    for relative, digest in bundle.get("evidence", {}).items():
        part=Path(relative)
        if part.is_absolute() or ".." in part.parts or sha256(Path(path).parent / part) != digest:
            raise ValueError("assistant caption observation evidence changed")
    return bundle


def freeze(root, authored):
    protocol, selection = verify_selection(root)
    folder = root / "assistant_captions"
    observations = read_json(folder / "samples.json")
    expected = expected_samples(protocol,selection)
    texts = read_json(authored)
    if len(texts) != len(expected) or sorted(map(int,texts)) != list(range(len(expected))):
        raise ValueError("authored text must have every segment ID exactly once")
    records = [dict(row, text=texts[str(row["segment"])]) for row in expected]
    payload = dict(version=1, status="CAPTIONS_PREPARED_RECONSTRUCTION_NOT_STARTED",
        input_sha256=protocol["input_sha256"],
        selection_freeze_sha256=sha256(root / "selection_freeze.json"),
        source_frame_hashes=observations["source_frame_hashes"], records=records,
        author="Codex assistant in this conversation; no external caption API invoked",
        protocol="Visible objects, actions, camera changes and background; short English captions. Avoid inferred identities and unseen objects.",
        scope="Offline assistant-authored source captions; conversation context and overview sheets were available. Not a blinded model-only comparison or independent ground truth.",
        sampling="Same four source-frame indices as PLLaVA; different display, prompt and model context.",
        hallucination_mitigation_verified=False, reconstruction_started=False,
        evidence={str(p.relative_to(folder)):sha256(p) for p in [folder / "samples.json",*sorted((folder / "sheets").glob("*.jpg"))]})
    path = folder / "captions_bundle.json"
    frozen = dict(payload,checksum=fingerprint(payload))
    if path.exists() and read_json(path) != frozen:
        raise ValueError("preserve existing captions; use a new bundle for revisions")
    write_json(path,frozen)
    validate_bundle(path,root)
    body=''.join(f'<tr><td>{r["segment"]}</td><td>{r["start"]/24:.3f}–{r["end_exclusive"]/24:.3f}</td><td>{html.escape(r["text"])}</td></tr>' for r in records)
    (folder / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>AI 캡션</title>'
        '<style>body{font:16px system-ui;max-width:1152px;margin:24px auto}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}img{max-width:100%}</style>'
        '<h1>혼합 키프레임 78구간 · AI 작성 캡션</h1><p>원본 표본으로 작성한 실험용 설명. 복원·독립 품질 검증은 미실행.</p>'
        '<p><a href="captions_bundle.json">설명·입력·출처 기록</a></p><table><tr><th>구간</th><th>시간(초)</th><th>전송할 설명</th></tr>'+body+'</table>'
        +''.join(f'<img src="sheets/{p.name}">' for p in sorted((folder / "sheets").glob("*.jpg")))+'</html>')
    print(f"FROZEN {len(records)} captions: {path}")


def import_captions(run):
    cfg=read_json(run / "run_config.json")
    bundle=validate_bundle(Path(cfg["caption_bundle"]),Path(cfg["hybrid_selection_root"]))
    indices=read_json(run / "keyframes.json")["indices"]
    if [r["start"] for r in bundle["records"]]+[bundle["records"][-1]["end_exclusive"]] != indices:
        raise ValueError("run keyframes differ from assistant captions")
    # Verify the actual frame-exact clips before importing source-derived texts.
    audit=read_json(run / "semantic_clips_audit.json")
    if audit["status"] != "PASSED" or audit["policy"] != "frame_exact" or audit["source"]["keyframes"] != indices:
        raise ValueError("caption import requires matching frame-exact clips")
    rows=[dict(path=f"clips/sample/{r['segment']:05d}.mp4",text=r["text"],flow=0.0) for r in bundle["records"]]
    sampling=[dict(clip=str(run / "data" / r["path"]),
        decoded_frames=b-a,indices=[i-a for i in sample_indices(a,b)],
        source_indices=sample_indices(a,b),caption_provider="assistant_provided")
        for r,a,b in zip(rows,indices,indices[1:])]
    write_json(run / "captions.json",rows)
    write_json(run / "caption_sampling.json",sampling)
    write_json(run / "caption_provenance.json",dict(provider="assistant_provided",bundle=cfg["caption_bundle"],
        bundle_sha256=sha256(cfg["caption_bundle"]),segments=len(rows),scope=bundle["scope"],
        pllava_inference_executed=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","freeze","verify"])
    parser.add_argument("--selection-root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--authored",type=Path)
    args=parser.parse_args()
    root=args.selection_root.resolve()
    if args.action == "prepare": prepare(root)
    elif args.action == "freeze":
        if args.authored is None: parser.error("freeze requires --authored")
        freeze(root,args.authored)
    else:
        value=validate_bundle(root / "assistant_captions/captions_bundle.json",root)
        print(f"VERIFIED {len(value['records'])} captions")


if __name__ == "__main__":
    main()
