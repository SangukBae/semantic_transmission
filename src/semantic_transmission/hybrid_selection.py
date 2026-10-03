"""Offline AI proposals, mandatory events, sequential SKEM, and time anchors.

Selection only: this module never captions, transmits, or reconstructs video.
The tv_low_08 proposals are prior AI observations, not independent ground truth.
"""
import argparse
import ast
import datetime
import html
import importlib.util
import math
from pathlib import Path
import shutil
import sys
import time
import types

from .artifacts import sha256, write_json
from .cli import repository
from .webvid5 import fingerprint, read_json

REPO = repository()
DEFAULT_ROOT = REPO / "outputs/etri_hybrid_keys_tv_low_08_v1"
BASE = REPO / "outputs/etri_60s_tv_low_08_42057b2ee8ed/baseline"
PROPOSALS = REPO / "outputs/etri_visual_keys_20260929/assistant_selection.json"
SKEM = REPO / "02_semantic_encoder/skem/MLM-keyframe-internvl.py"
PROTECTED_KINDS = {"boundary", "object_visibility", "occlusion", "dog_entry_exit",
                   "dog_pose_occlusion", "people_entry_exit", "rapid_pan"}
# Earlier annotations also record people leaving/reappearing in these viewpoint
# proposals. Keep the event observations instead of vetoing them with SKEM.
PROTECTED_VIEWPOINTS = {32, 180, 192}


def choose_keys(frames, proposals, mandatory, scorer, *, max_gap=24, threshold=0.35):
    """Score each optional candidate against the latest *accepted* frame.

    Insert elapsed-time anchors before scoring the next proposal. Rejections
    do not update the reference. Endpoints and mandatory events bypass SKEM.
    """
    candidates = sorted(set(proposals))
    required = set(mandatory) | {0, frames - 1}
    if (frames < 2 or max_gap < 1 or not math.isfinite(threshold)
            or not -1 <= threshold <= 1 or not candidates
            or candidates[0] != 0 or candidates[-1] != frames - 1
            or any(type(i) is not int or not 0 <= i < frames for i in candidates)
            or not required.issubset(candidates)):
        raise ValueError("invalid hybrid timeline/proposals/mandatory frames")
    keys = [0]
    records = [dict(frame=0, reference=None, selected=True, reason="mandatory", mandatory=True)]
    for index in candidates[1:]:
        while keys[-1] + max_gap <= index:
            previous = keys[-1]
            anchor = previous + max_gap
            keys.append(anchor)
            records.append(dict(frame=anchor, reference=previous, selected=True,
                reason="mandatory_and_max_gap" if anchor in required else "max_gap",
                mandatory=anchor in required))
        if keys[-1] == index:
            continue
        previous = keys[-1]
        if index in required:
            record = dict(frame=index, reference=previous, selected=True,
                          reason="mandatory", mandatory=True)
        else:
            score = scorer(previous, index)
            p_yes, p_no = score["p_yes"], score["p_no"]
            if (not all(math.isfinite(p) and 0 <= p <= 1 for p in (p_yes, p_no))
                    or p_yes + p_no > 1.000001):
                raise ValueError("invalid SKEM probabilities")
            diff = p_no - p_yes
            record = dict(frame=index, reference=previous, selected=diff > threshold,
                reason="skem", mandatory=False, p_yes=p_yes, p_no=p_no, psss=diff)
        records.append(record)
        if record["selected"]:
            keys.append(index)
    return keys, records


def model_inventory(directory):
    """Bind the installed snapshot, including actual weight bytes, once per run."""
    return {str(p): {"sha256": sha256(p), "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


def prepare(root):
    if (root / "protocol.json").exists():
        return validate_protocol(root)
    cfg = read_json(BASE / "run_config.json")
    proposal = read_json(PROPOSALS)
    source = PROPOSALS.parent
    if sha256(PROPOSALS) != read_json(source / "selection_freeze.json")["selection_sha256"]:
        raise ValueError("AI proposals changed after visual review")
    if proposal["input_sha256"] != cfg["input_sha256"] or sha256(cfg["input"]) != cfg["input_sha256"]:
        raise ValueError("source video differs from reviewed proposals")
    candidates = [dict(r) for r in proposal["records"] if r["kind"] != "max_gap"]
    for row in candidates:
        row["mandatory"] = row["kind"] in PROTECTED_KINDS or row["frame"] in PROTECTED_VIEWPOINTS
    tree = ast.parse(SKEM.read_text())
    prompts = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("prompt_ask_image", "prompt_compare_image")}
    frame_hashes = read_json(source / "source_frame_hashes.json")
    code = [Path(__file__), SKEM, REPO / "src/semantic_transmission/internvl_memory.py",
            REPO / "src/semantic_transmission/exact_reuse.py"]
    protocol = dict(version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_id="tv_low_08", source_split="development", baseline=str(BASE),
        input=cfg["input"], input_sha256=cfg["input_sha256"], frames=1440, fps=24,
        width=576, height=320, max_gap_frames=24, threshold=0.35,
        candidates=candidates, proposal_path=str(PROPOSALS), proposal_sha256=sha256(PROPOSALS),
        source_frames=str(BASE / "data/frames/sample"), source_frame_hashes=frame_hashes,
        normalized_video=str(BASE / "data/normalized.mp4"),
        normalized_sha256=sha256(BASE / "data/normalized.mp4"),
        config=cfg, prompts=prompts, model_files=model_inventory(cfg["models"]["internvl"]),
        code={str(p.relative_to(REPO)): sha256(p) for p in code},
        scope="Prior AI source observations reused; not independent labels or all-frame event recall.",
        sampling="Prior 6fps overview plus dense focus, 469 unique inspected source frames.",
        protection="Preserve event-related proposal groups conservatively, including rapid-pan visibility boundaries; no hard cut was confirmed.",
        score_reuse="Only this protocol and the exact reference/candidate pair; historical SKEM scores are not imported.",
        reconstruction_status="NOT_STARTED", hallucination_mitigation_verified=False)
    protocol["signature"] = fingerprint(protocol)
    write_json(root / "protocol.json", protocol)
    validate_protocol(root)
    return protocol


def validate_protocol(root, *, check_model=False):
    protocol = read_json(root / "protocol.json")
    if fingerprint({k:v for k,v in protocol.items() if k != "signature"}) != protocol["signature"]:
        raise ValueError("hybrid protocol changed")
    for name, digest in protocol["code"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"selection code changed: {name}")
    for path, digest in [(protocol["input"], protocol["input_sha256"]),
                         (protocol["proposal_path"], protocol["proposal_sha256"]),
                         (protocol["normalized_video"], protocol["normalized_sha256"])]:
        if sha256(path) != digest:
            raise ValueError(f"source/proposal changed: {path}")
    for index, digest in protocol["source_frame_hashes"].items():
        if sha256(Path(protocol["source_frames"]) / f"{index}.png") != digest:
            raise ValueError(f"source frame changed: {index}")
    if check_model:
        for path, info in protocol["model_files"].items():
            if sha256(path) != info["sha256"]:
                raise ValueError(f"InternVL model changed: {path}")
    return protocol


class InternVLScorer:
    def __init__(self, protocol):
        self.protocol = protocol
        self.model = None
        self.load_seconds = 0

    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from .exact_reuse import FrameTensorCache
        from .internvl_memory import place_internvl
        cfg = self.protocol["config"]
        sys.path.insert(0, str(REPO / ".local/vendor/InternVL"))
        spec = importlib.util.spec_from_file_location("hybrid_official_skem", SKEM)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        begin = time.perf_counter()
        torch.manual_seed(cfg["seed"])
        self.model = AutoModel.from_pretrained(cfg["models"]["internvl"], torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_flash_attn=cfg["flash_attn"], trust_remote_code=True).eval()
        self.model = place_internvl(self.model, gpu_head=cfg["internvl_gpu_head"],
            cpu_layers=cfg["internvl_offload_layers"], compact_cache=cfg["internvl_compact_kv_cache"])
        self.model.chat = types.MethodType(module.custom_chat, self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["internvl"], trust_remote_code=True, use_fast=True)
        self.cache = FrameTensorCache(lambda p: module.load_image(p, max_num=cfg["max_tiles"]).to(torch.bfloat16).cuda())
        self.generation = dict(max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                              output_scores=True, return_dict_in_generate=True)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - begin

    def __call__(self, reference, candidate):
        import torch
        if self.model is None:
            self.load()
        begin = time.perf_counter()
        directory = Path(self.protocol["source_frames"])
        images = [self.cache.get(str(directory / f"{i}.png")) for i in (reference, candidate)]
        pixels, patches = torch.cat(images), [len(im) for im in images]
        with torch.inference_mode():
            text, history, _ = self.model.chat(self.tokenizer, pixels,
                self.protocol["prompts"]["prompt_ask_image"], dict(self.generation, output_scores=False),
                num_patches_list=patches, return_history=True)
            torch.cuda.synchronize()
            middle = time.perf_counter()
            answer, _, scores = self.model.chat(self.tokenizer, pixels,
                self.protocol["prompts"]["prompt_compare_image"], dict(self.generation),
                num_patches_list=patches, history=history, return_history=True)
            probabilities = torch.softmax(scores[0], dim=-1)[0]
            result = {f"p_{word.lower()}": float(probabilities[self.tokenizer.encode(word, add_special_tokens=False)[0]])
                      for word in ("Yes", "No")}
        torch.cuda.synchronize()
        result.update(description=text, answer=answer, round1_seconds=middle-begin,
                      round2_seconds=time.perf_counter()-middle, total_seconds=time.perf_counter()-begin)
        return result


def cached_score(root, protocol, reference, candidate, scorer):
    path = root / f"scores/{reference:05d}_{candidate:05d}.json"
    identity = dict(protocol_signature=protocol["signature"], reference=reference, candidate=candidate)
    if path.exists():
        saved = read_json(path)
        if (saved["identity"] != identity or fingerprint(saved["score"]) != saved["score_sha256"]):
            raise ValueError(f"score cache changed: {path}")
        return saved["score"]
    score = scorer(reference, candidate)
    write_json(path, dict(identity=identity, score=score, score_sha256=fingerprint(score)))
    return score


def export_selection(root, protocol, keys, records, elapsed, model_load_seconds):
    from PIL import Image, ImageDraw, ImageFont
    folder = root / "selected_frames"
    folder.mkdir(exist_ok=True)
    hashes = {}
    for i in keys:
        target = folder / f"{i:05d}.png"
        source = Path(protocol["source_frames"]) / f"{i}.png"
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError(f"existing selected PNG changed: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
        hashes[target.name] = sha256(target)
    if set(p.name for p in folder.glob("*.png")) != set(hashes):
        raise ValueError("unexpected selected frames from a different selection")
    used = [r for r in records if r["reason"] == "skem"]
    source_reasons = {r["frame"]: r["reason_ko"] for r in protocol["candidates"]}
    for row in records:
        row["time_seconds"] = row["frame"] / protocol["fps"]
        row["source_reason_ko"] = source_reasons.get(row["frame"], "최대 1초 간격 보강")
    selection = dict(status="COMPLETE", selector="ai_candidates_skem_events_maxgap_v1",
        protocol_signature=protocol["signature"], input_sha256=protocol["input_sha256"],
        frames=protocol["frames"], fps=protocol["fps"], indices=keys, records=records,
        candidate_count=len(protocol["candidates"]), mandatory_count=sum(r["mandatory"] for r in protocol["candidates"]),
        skem_comparisons=len(used), skem_accepted=sum(r["selected"] for r in used),
        skem_rejected=sum(not r["selected"] for r in used),
        max_gap_frames=max(b-a for a,b in zip(keys, keys[1:])),
        selection_seconds_this_attempt=elapsed, model_load_seconds_this_attempt=model_load_seconds,
        score_seconds_total=sum(read_json(root / f"scores/{r['reference']:05d}_{r['frame']:05d}.json")["score"]["total_seconds"] for r in used),
        timing_scope="Current hybrid selection and export preparation; prior AI proposal inspection excluded; retries can reuse exact score pairs.",
        reconstruction_status="NOT_STARTED", hallucination_mitigation_verified=False,
        independent_ground_truth=False)
    write_json(root / "selection.json", selection)
    write_json(root / "keyframes.json", dict(indices=keys, selector=selection["selector"]))
    write_json(root / "selection_freeze.json", dict(protocol_sha256=sha256(root / "protocol.json"),
        selection_sha256=sha256(root / "selection.json"), keyframes_sha256=sha256(root / "keyframes.json"),
        frames=hashes, score_files={str(p.relative_to(root)): sha256(p) for p in sorted((root / "scores").glob("*.json"))}))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    sheets = []
    for page, start in enumerate(range(0, len(keys), 48)):
        values = keys[start:start+48]
        canvas = Image.new("RGB", (1536, math.ceil(len(values)/8)*132), "#202020")
        draw = ImageDraw.Draw(canvas)
        for n, index in enumerate(values):
            x, y = n%8*192, n//8*132
            with Image.open(folder / f"{index:05d}.png") as im:
                canvas.paste(im.convert("RGB").resize((192, 107)), (x, y+23))
            draw.text((x+3, y+3), f"f{index} {index/24:.3f}s", font=font, fill="white")
        target = root / f"keys_{page:02d}.jpg"
        canvas.save(target, quality=94)
        sheets.append(target.name)
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
        (r['frame'], round(r['time_seconds'],3), r['selected'], r['reason'],
         round(r['psss'],4) if 'psss' in r else '-', r['source_reason_ko']))+'</tr>' for r in records)
    (root / "selection.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>tv_low_08 혼합 키프레임</title><style>body{font:16px system-ui;max-width:1560px;margin:24px auto}img{max-width:100%}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        f'<h1>혼합 키프레임 {len(keys)}개 — 선택 완료</h1><p>AI 후보·필수 사건 + SKEM 확률 차이 0.35 + 최대 1초 간격. 복원은 별도 명령으로 실행합니다.</p>'
        '<p><a href="selection.json">선택·거절 기록</a> · <a href="keyframes.json">프레임 목록</a> · <a href="protocol.json">기준·검토 범위</a></p>'
        + ''.join(f'<img src="{p}">' for p in sheets)
        + '<table><tr><th>프레임</th><th>초</th><th>선택</th><th>판정</th><th>PSSS</th><th>후보 이유</th></tr>'+table+'</table></html>')
    return selection


def verify_selection(root):
    protocol = validate_protocol(root)
    freeze = read_json(root / "selection_freeze.json")
    for filename, field in [("protocol.json", "protocol_sha256"), ("selection.json", "selection_sha256"), ("keyframes.json", "keyframes_sha256")]:
        if sha256(root / filename) != freeze[field]:
            raise ValueError(f"frozen selection changed: {filename}")
    selection = read_json(root / "selection.json")
    if selection["status"] != "COMPLETE" or selection["protocol_signature"] != protocol["signature"]:
        raise ValueError("hybrid selection is incomplete or from a different protocol")
    mandatory = [r["frame"] for r in protocol["candidates"] if r["mandatory"]]
    def score(a,b):
        def missing(*_):
            raise ValueError("completed selection has a missing SKEM score")
        return cached_score(root, protocol, a, b, missing)
    keys, records = choose_keys(protocol["frames"], [r["frame"] for r in protocol["candidates"]], mandatory,
                                score, max_gap=protocol["max_gap_frames"], threshold=protocol["threshold"])
    if keys != selection["indices"] or keys != read_json(root / "keyframes.json")["indices"]:
        raise ValueError("selected keys disagree with sequential SKEM replay")
    expected = {f"{i:05d}.png" for i in keys}
    if set(freeze["frames"]) != expected or set(p.name for p in (root / "selected_frames").glob("*.png")) != expected:
        raise ValueError("selected PNG inventory differs from selected indices")
    for p, digest in freeze["frames"].items():
        if sha256(root / "selected_frames" / p) != digest or digest != protocol["source_frame_hashes"][str(int(Path(p).stem))]:
            raise ValueError(f"selected PNG differs from source: {p}")
    for p, digest in freeze["score_files"].items():
        if sha256(root / p) != digest:
            raise ValueError(f"score file changed: {p}")
    return protocol, selection


def select(root):
    from .etri_60s_check import lock
    root.mkdir(parents=True, exist_ok=True)
    with lock(REPO / ".local/etri_60s_check.lock"), lock(root / ".selection.lock"):
        if (root / "selection_freeze.json").exists():
            _, selection = verify_selection(root)
            print(f"HYBRID_SELECTION_REUSED keys={len(selection['indices'])}", flush=True)
            return
        protocol = prepare(root)
        validate_protocol(root, check_model=True)
        scorer = InternVLScorer(protocol)
        begin = time.perf_counter()
        keys, records = choose_keys(protocol["frames"], [r["frame"] for r in protocol["candidates"]],
            [r["frame"] for r in protocol["candidates"] if r["mandatory"]],
            lambda a,b: cached_score(root, protocol, a, b, scorer),
            max_gap=protocol["max_gap_frames"], threshold=protocol["threshold"])
        result = export_selection(root, protocol, keys, records, time.perf_counter()-begin, scorer.load_seconds)
        verify_selection(root)
        print(f"HYBRID_SELECTION_COMPLETE keys={len(keys)} SKEM_pairs={result['skem_comparisons']} max_gap={result['max_gap_frames']} frames; reconstruction NOT_STARTED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "verify"])
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == "select":
        select(root)
    else:
        _, result = verify_selection(root)
        print(f"VERIFIED {len(result['indices'])} keyframes; no reconstruction executed")


if __name__ == "__main__":
    main()
