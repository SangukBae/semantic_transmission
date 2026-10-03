"""Local, resumable FC-LGVSC condition extraction; no inference API calls.

The open-model proposal/caption variant is not the assistant-authored condition
and does not establish equal quality, reconstruction quality, or hallucination
mitigation. Original sequential InternVL SKEM is retained for optional proposals.
"""
import argparse
import datetime
import fcntl
import hashlib
import html
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .artifacts import sha256, write_json
from .assisted_captions import sample_indices
from .hybrid_selection import choose_keys

REPO = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DEFAULT_PREPARED = REPO / "outputs/fc_lgvsc_webvid_tvsum_full_20261001"
DEFAULT_ROOT = REPO / "outputs/fc_lgvsc_auto_20261002"
TRANSNET = REPO / "data/etri_benchmark_v1_20260924/metadata/transnetv2"
KINDS = {"entry_exit", "occlusion", "pose", "viewpoint", "scene_change", "appearance"}
PROTECTED = {"entry_exit", "occlusion", "scene_change", "abrupt_visual_change"}

PROPOSAL_PROMPT = '''You are observing source video frames for faithful video reconstruction.
The separate images are in chronological order. Each metadata label gives the
SOURCE FRAME ID, not an object or part of the video. Compare the first image,
intermediate images and last image, including background and visible body parts.
Never infer video content from filenames, topic or world knowledge.
Find visible changes that need an extra keyframe: an object/person entering or
leaving view, becoming occluded/revealed, a discrete pose change, viewpoint change,
or a shot transition. Do not propose every frame or ordinary tiny motion. Do not
invent a face when only a back is visible. Do not mistake camera motion for an
object entering. Gradual focus/color changes are appearance changes, not cuts.
Return ONLY JSON: {"events":[{"before":INT,"after":INT,"kind":"entry_exit|occlusion|pose|viewpoint|scene_change|appearance","description":"short visible before/after change"}],"uncertainties":["short uncertain observation"]}.
Use only provided SOURCE FRAME IDs. before < after, preferably neighboring
observed frames enclosing the change. Use [] if no discrete change is visible.
Do not assert exact transition timing between sampled frames. At most 12 events.
'''

CAPTION_PROMPT = '''Describe ONLY the visible content of these chronologically ordered
source frames, for faithful video reconstruction. Frame labels and timestamps
are metadata, not objects in the scene. Do not infer the topic, location, purpose,
identity, emotion, hidden anatomy or future action. Avoid guessing sleep, hugging,
smiling or other behavior from a static pose. Distinguish real footage,
animation, title cards and dissolves. Preserve visible object appearance, position,
facing direction, occlusion, background and actual movement. A person seen only
from behind must not acquire a visible face. Mention blur/partial visibility when
it limits recognition. If motion is unclear, describe the visible states without
claiming direction or action, and omit claims that no movement or change occurs.
Avoid decorative praise, guessed text and tiny
unresolvable details. Prefer a broad visible object/place category over an inferred
specific subtype or room type. Describe visible spatial proximity without guessing
holding, wrapping, or other physical relationships hidden by overlap.
Do not claim "nothing else" or "no people" unless evident.
Return ONLY JSON: {"caption":"2-4 concise English sentences, at most 72 words",
"uncertainties":["unresolved visible ambiguity, or empty list"]}.
'''

REVIEW_PROMPT = '''Check the draft caption against ONLY the provided source frames.
Remove or generalize unsupported objects, identities, actions, directions,
expressions, readable text and spatial relations. Keep useful directly visible
appearance, location, occlusion and temporal change. Check each claim, even if the
draft sounds plausible. Do not add speculative details. Do not describe metadata
labels. Keep uncertainty about ambiguous blur/occlusion; do not invent hidden
content. Remove claims about sleep, emotion, absence of motion, inferred room
types or hidden physical relationships unless directly evidenced. Use generic
object/place terms if subtype is uncertain. Return ONLY JSON: {"caption":"corrected English description, 1-4
sentences, at most 72 words", "removed_claims":["unsupported claim removed"],
"uncertainties":["remaining ambiguity, or empty list"]}.
Draft caption:
'''


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, checksum=digest(value))


def unseal(path):
    value = read(path)
    expected = value.pop("checksum")
    if digest(value) != expected:
        raise ValueError(f"checksum mismatch: {path}")
    return value


def parse_json(text):
    """Accept a JSON object, optionally fenced; reject trailing prose/objects."""
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("incomplete JSON fence")
        value = "\n".join(lines[1:-1]).strip()
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def validate_events(obj, ids):
    if not isinstance(obj.get("events"), list) or len(obj["events"]) > 12:
        raise ValueError("invalid event list")
    for event in obj["events"]:
        a, b = event.get("before"), event.get("after")
        if (type(a) is not int or type(b) is not int or a not in ids or b not in ids
                or a >= b or event.get("kind") not in KINDS
                or not isinstance(event.get("description"), str)
                or not event["description"].strip()):
            raise ValueError("event does not refer to valid observed frames")
    validate_strings(obj, "uncertainties")
    return obj


def validate_strings(obj, key):
    if not isinstance(obj.get(key), list) or any(not isinstance(x, str) for x in obj[key]):
        raise ValueError(f"invalid {key}")


def validate_caption(obj, review=False):
    caption = obj.get("caption")
    if (not isinstance(caption, str) or not caption.strip() or "\n" in caption
            or len(caption.split()) > 80):
        raise ValueError("caption must be a nonempty single line of <=80 words")
    validate_strings(obj, "uncertainties")
    if review:
        validate_strings(obj, "removed_claims")
    return obj


def proposal_windows(frames, stride=4, width=48):
    """6 FPS observations at 24 FPS, one shared boundary per 2s window."""
    for start in range(0, frames - 1, width):
        end = min(start + width, frames - 1)
        yield sorted(set(range(start, end + 1, stride)) | {end})


def check_frame(source, index):
    path = Path(source["source_frames"]) / f"{index}.png"
    if sha256(path) != source["source_frame_hashes"][str(index)]:
        raise ValueError(f"source frame changed: {path}")
    return path


def source_for(config, video_id):
    video = next(v for v in config["videos"] if v["id"] == video_id)
    path = Path(config["prepared_root"]) / video["dataset"].lower() / video_id / "source_prepared.json"
    if sha256(path) != video["prepared_sha256"]:
        raise ValueError(f"source metadata changed: {path}")
    source = read(path)
    if source["fps"] != 24 or source["frames"] < 2:
        raise ValueError("expected full prepared 24 FPS video")
    return source


def video_root(root, source):
    result = root / source["dataset"].lower() / source["id"]
    result.mkdir(parents=True, exist_ok=True)
    return result


def contact_sheet(source, ids):
    from PIL import Image, ImageDraw, ImageFont
    cols, w, h = 5, 288, 182
    canvas = Image.new("RGB", (cols*w, math.ceil(len(ids)/cols)*h), "#171717")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for n, index in enumerate(ids):
        x, y = (n % cols)*w, (n // cols)*h
        with Image.open(check_frame(source, index)) as im:
            canvas.paste(im.convert("RGB").resize((288, 160)), (x, y+22))
        draw.text((x+4, y+1), str(index), font=font, fill="white")
    return canvas


class LocalVLM:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        self.config = config
        torch.set_num_threads(8)
        torch.manual_seed(2025)
        self.processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
        extra = {}
        if config.get("quantization") == "int8":
            extra["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
                llm_int8_skip_modules=["visual", "lm_head"])
        elif config.get("quantization") == "nf4":
            extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,llm_int8_skip_modules=["visual", "lm_head"])
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model_path"], dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True, **extra).eval()
        self.load_finished = now()

    def generate(self, images, labels, prompt, max_tokens):
        import torch
        content = []
        for image, label in zip(images, labels):
            content.extend([{"type": "text", "text": label}, {"type": "image", "image": image}])
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to("cuda")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                         use_cache=True)
        torch.cuda.synchronize()
        generated = output[0, inputs.input_ids.shape[1]:]
        result = {"raw_text": self.processor.decode(generated, skip_special_tokens=True),
                  "generated_tokens": len(generated), "input_tokens": inputs.input_ids.shape[1],
                  "seconds": time.monotonic()-started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "truncated": len(generated) >= max_tokens}
        del inputs, output, generated
        return result


def infer_json(vlm, path, identity, images, labels, prompt, validator, tokens=640):
    """Content-bound atomic cache; all failed responses retained for audit."""
    signature = digest(identity)
    if path.exists():
        value = unseal(path)
        if value["identity"] != identity:
            raise ValueError(f"inference cache identity changed: {path}")
        validator(value["result"])
        return value["result"]
    for attempt in range(3):
        task = prompt if attempt == 0 else prompt + "\nYour previous response failed JSON/schema validation. Return ONLY the exact requested JSON; keep it concise."
        response = vlm.generate(images, labels, task, tokens + attempt*192)
        record = dict(identity=identity, signature=signature, attempt=attempt, created_utc=now(), **response)
        try:
            if response["truncated"]:
                raise ValueError("generation reached token limit")
            result = validator(parse_json(response["raw_text"]))
        except (ValueError, TypeError, KeyError) as exc:
            record["error"] = str(exc)
            write_json(path.parent / "failed_attempts" / f"{path.stem}_{attempt}.json", seal(record))
            continue
        record["result"] = result
        write_json(path, seal(record))
        return result
    raise RuntimeError(f"three invalid model responses: {path}")


def pixel_change_events(frames, *, minimum=.05, ratio=4., history=24):
    """Conservative extra anchors for large abrupt changes, including overlays.

    This heuristic may also retain flashes/fast motion; it is not a calibrated
    semantic detector. It observes every normalized frame, unlike 6 FPS VLM input.
    """
    import numpy as np
    difference=np.r_[0.,np.abs(np.diff(frames.astype(np.float32),axis=0)).mean(axis=(1,2,3))/255.]
    events=[]
    for i in np.flatnonzero(difference>=minimum):
        if i==0:continue
        baseline=float(np.median(difference[max(0,i-history):i]))
        if difference[i]>=ratio*max(baseline,.01):
            events.append(dict(before=int(i)-1,after=int(i),kind="abrupt_visual_change",
                origin="pixel_change_guard",mean_rgb_change=float(difference[i]),past_median=baseline,
                description="Abrupt source pixel change; protect before/after even if the neural shot detector misses it"))
    return events


def transnet(source, dest, config):
    import numpy as np
    import torch
    path = dest / "shot_candidates.json"
    identity = dict(config_signature=config["signature"], source_sha256=source["normalized_sha256"])
    if path.exists():
        value = unseal(path)
        if value["identity"] != identity:
            raise ValueError("shot cache identity changed")
        return value["events"]
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("fc_transnet", TRANSNET / "transnetv2_pytorch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TransNetV2().eval().cuda()
    model.load_state_dict(torch.load(TRANSNET / "weights.pth", map_location="cpu", weights_only=True))
    torch.set_num_threads(4)
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "2", "-filter_threads", "2",
        "-i", source["normalized_video"], "-vf", "scale=48:27", "-vsync", "0",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1,27,48,3)
    if len(frames) != source["frames"]:
        raise ValueError("TransNet decoded frame count mismatch")
    padded = np.pad(frames, ((25,25+(-len(frames))%50),(0,0),(0,0),(0,0)), mode="edge")
    pred = []
    with torch.inference_mode():
        for i in range(0,len(padded)-99,50):
            one, _ = model(torch.from_numpy(padded[i:i+100][None]).cuda())
            pred.append(torch.sigmoid(one)[0,25:75,0].cpu().numpy())
    pred = np.concatenate(pred)[:len(frames)]
    mask = pred > .5
    starts = np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:],False])
    events = [{"before": max(0,int(a)-1), "after": min(len(frames)-1,int(b)+1),
               "peak_frame": int(a+np.argmax(pred[a:b+1])),
               "probability": float(pred[a:b+1].max()), "kind": "scene_change",
               "origin": "TransNetV2",
               "description": "TransNetV2 transition candidate; not human ground truth"}
              for a,b in zip(starts,ends)]
    guard_events=pixel_change_events(frames,**config["pixel_guard"])
    events.extend(guard_events)
    write_json(path, seal(dict(identity=identity, events=events, seconds=time.monotonic()-started)))
    del model, padded, frames, raw, pred
    torch.cuda.empty_cache()
    return events


def progress(dest, phase, done, total):
    write_json(dest / "progress.json", dict(phase=phase, completed_units=done,
        total_units=total, pid=os.getpid(), updated_utc=now()))


def propose(root, config, source):
    from PIL import Image
    dest = video_root(root, source)
    path = dest / "proposals.json"
    if path.exists():
        result = unseal(path)
        if result["config_signature"] != config["signature"]:
            raise ValueError("proposals configuration changed")
        return
    if sha256(source["normalized_video"]) != source["normalized_sha256"]:
        raise ValueError("normalized video changed")
    if sha256(source["source_path"]) != source["source_sha256"]:
        raise ValueError("original video changed")
    events = transnet(source, dest, config)
    vlm = LocalVLM(config)
    windows = list(proposal_windows(source["frames"],config["candidate_stride"],config["candidate_window_frames"]))
    uncertainties = []
    for n, ids in enumerate(windows):
        progress(dest, "visual_proposals", n, len(windows))
        images = [Image.open(check_frame(source,i)).convert("RGB") for i in ids]
        labels = [f"SOURCE FRAME ID {i}, time {i/24:.6f} seconds" for i in ids]
        identity = dict(config_signature=config["signature"], task="proposals", indices=ids,
                        frame_hashes=[source["source_frame_hashes"][str(i)] for i in ids])
        result = infer_json(vlm, dest / "observations" / f"{n:05d}.json", identity,
            images, labels, PROPOSAL_PROMPT,
            lambda x: validate_events(x, ids), tokens=768)
        for im in images: im.close()
        for event in result["events"]:
            events.append(dict(event, origin="Qwen3-VL", observation_window=n))
        uncertainties.extend(dict(window=n, text=t) for t in result["uncertainties"])
    by_frame = {0:dict(frame=0, mandatory=True, reasons=["first frame"]),
                source["frames"]-1:dict(frame=source["frames"]-1, mandatory=True, reasons=["last frame"])}
    for event in events:
        mandatory = event["kind"] in PROTECTED
        # Both observations protect appearance before/after, not an invented exact onset.
        for i in (event["before"],event["after"]):
            row = by_frame.setdefault(i,dict(frame=i, mandatory=False, reasons=[]))
            row["mandatory"] |= mandatory
            row["reasons"].append(event["description"])
    write_json(path, seal(dict(config_signature=config["signature"], events=events,
        candidates=[by_frame[i] for i in sorted(by_frame)], uncertainties=uncertainties,
        observed_unique_frames=len(set(i for ids in windows for i in ids)),
        temporal_resolution_frames=4, transition_timing="observed brackets, not exact VLM onset")))
    progress(dest, "visual_proposals", len(windows), len(windows))


def select(root, config, source):
    from .hybrid_selection import InternVLScorer
    dest = video_root(root, source)
    proposals = unseal(dest / "proposals.json")
    identity = dict(config_signature=config["signature"], proposals_sha256=sha256(dest / "proposals.json"))
    if (dest / "selection.json").exists():
        if unseal(dest / "selection.json")["identity"] != identity:
            raise ValueError("selection identity changed")
        return
    template = config["skem"]
    scorer = InternVLScorer(dict(config=template["config"],prompts=template["prompts"],
                                source_frames=source["source_frames"]))
    count = 0
    def score(a,b):
        nonlocal count
        check_frame(source,a); check_frame(source,b)
        path = dest / "skem_scores" / f"{a:06d}_{b:06d}.json"
        cache_id = dict(identity, reference=a, candidate=b,
                       frame_hashes=[source["source_frame_hashes"][str(i)] for i in (a,b)])
        if path.exists():
            cached = unseal(path)
            if cached["identity"] != cache_id:
                raise ValueError("SKEM cache mismatch")
            result = cached["result"]
        else:
            result = scorer(a,b)
            write_json(path,seal(dict(identity=cache_id,result=result)))
        count += 1
        progress(dest,"sequential_skem",count,len(proposals["candidates"]))
        return result
    keys, records = choose_keys(source["frames"], [r["frame"] for r in proposals["candidates"]],
        [r["frame"] for r in proposals["candidates"] if r["mandatory"]], score,
        max_gap=config["max_gap_frames"], threshold=config["psss_threshold"])
    selected = dest / "selected_frames"
    selected.mkdir(exist_ok=True)
    for i in keys:
        target = selected / f"{i:05d}.png"
        original = check_frame(source,i)
        if not target.exists():
            shutil.copyfile(original,target)
        if sha256(target) != source["source_frame_hashes"][str(i)]:
            raise ValueError("selected keyframe bytes changed")
    write_json(dest / "keyframes.json", dict(indices=keys,fps=24))
    write_json(dest / "selection.json",seal(dict(identity=identity,indices=keys,records=records,
        source_frames=source["frames"],fps=24,skem_pairs=count,
        keyframe_png_bytes=sum((selected/f"{i:05d}.png").stat().st_size for i in keys))))
    progress(dest,"selection_complete",len(keys),len(keys))


def caption(root, config, source):
    from PIL import Image
    dest = video_root(root,source)
    selection = unseal(dest / "selection.json")
    identity = dict(config_signature=config["signature"], selection_sha256=sha256(dest / "selection.json"))
    vlm = LocalVLM(config)
    rows = []
    pairs = list(zip(selection["indices"],selection["indices"][1:]))
    for n,(a,b) in enumerate(pairs):
        progress(dest,"captions_and_visual_review",n,len(pairs))
        ids = sample_indices(a,b)
        images = [Image.open(check_frame(source,i)).convert("RGB") for i in ids]
        labels = [f"Source frame {i}, time {i/24:.6f} seconds" for i in ids]
        item_id = dict(identity,segment=n,start=a,end_exclusive=b,source_indices=ids,
                       frame_hashes=[source["source_frame_hashes"][str(i)] for i in ids])
        draft = infer_json(vlm,dest/"captions_draft"/f"{n:05d}.json",dict(item_id,task="draft"),
                           images,labels,CAPTION_PROMPT,validate_caption,tokens=384)
        reviewed = infer_json(vlm,dest/"captions_review"/f"{n:05d}.json",
            dict(item_id,task="visual_review",draft_sha256=digest(draft)),images,labels,
            REVIEW_PROMPT+draft["caption"],lambda x:validate_caption(x,review=True),tokens=384)
        rows.append(dict(segment=n,start=a,end_exclusive=b,source_indices=ids,
            start_seconds=a/24,end_seconds=b/24,caption=reviewed["caption"],
            uncertainties=reviewed["uncertainties"],removed_claims=reviewed["removed_claims"]))
        for im in images: im.close()
    write_json(dest / "captions.json",seal(dict(identity=identity,author=config["model_id"],
        model_revision=config["model_revision"],caption_sampling="PLLaVA four-sample half-open intervals",
        independent_verification=False,captions=rows)))
    progress(dest,"captions_complete",len(rows),len(rows))


def verify_video(root, config, source):
    dest = video_root(root,source)
    proposals = unseal(dest/"proposals.json")
    selection = unseal(dest/"selection.json")
    captions = unseal(dest/"captions.json")
    for path_key,hash_key in [("normalized_video","normalized_sha256"),("source_path","source_sha256")]:
        if path_key in source and sha256(source[path_key]) != source[hash_key]:
            raise ValueError("source video bytes changed")
    if proposals["config_signature"] != config["signature"]:
        raise ValueError("proposal configuration mismatch")
    expected_id = dict(config_signature=config["signature"],proposals_sha256=sha256(dest/"proposals.json"))
    if selection["identity"] != expected_id:
        raise ValueError("selection provenance mismatch")
    if captions["identity"] != dict(config_signature=config["signature"],selection_sha256=sha256(dest/"selection.json")):
        raise ValueError("caption provenance mismatch")
    def score(a,b):
        cached = unseal(dest/"skem_scores"/f"{a:06d}_{b:06d}.json")
        expected = dict(expected_id,reference=a,candidate=b,
                        frame_hashes=[source["source_frame_hashes"][str(i)] for i in (a,b)])
        if cached["identity"] != expected:
            raise ValueError("sequential score provenance mismatch")
        return cached["result"]
    candidates = proposals["candidates"]
    keys,records = choose_keys(source["frames"],[r["frame"] for r in candidates],
        [r["frame"] for r in candidates if r["mandatory"]],score,
        max_gap=config["max_gap_frames"],threshold=config["psss_threshold"])
    if keys != selection["indices"] or records != selection["records"]:
        raise ValueError("sequential selection replay mismatch")
    if len(captions["captions"]) != len(keys)-1:
        raise ValueError("incomplete captions")
    for i in keys:
        if sha256(dest/"selected_frames"/f"{i:05d}.png") != source["source_frame_hashes"][str(i)]:
            raise ValueError("keyframe integrity mismatch")
    if read(dest/"keyframes.json")["indices"] != keys:
        raise ValueError("keyframe export mismatch")
    for n,(row,a,b) in enumerate(zip(captions["captions"],keys,keys[1:])):
        if (row["segment"],row["start"],row["end_exclusive"],row["source_indices"]) != (n,a,b,sample_indices(a,b)):
            raise ValueError("caption timeline mismatch")
        validate_caption(row,review=True)
        review = unseal(dest/"captions_review"/f"{n:05d}.json")
        draft = unseal(dest/"captions_draft"/f"{n:05d}.json")
        item_id = dict(captions["identity"],segment=n,start=a,end_exclusive=b,source_indices=sample_indices(a,b),
                       frame_hashes=[source["source_frame_hashes"][str(i)] for i in sample_indices(a,b)])
        if draft["identity"] != dict(item_id,task="draft"):
            raise ValueError("draft provenance mismatch")
        if review["identity"] != dict(item_id,task="visual_review",draft_sha256=digest(draft["result"])):
            raise ValueError("review provenance mismatch")
        if review["result"]["caption"] != row["caption"]:
            raise ValueError("caption differs from local inference record")
        for i in row["source_indices"]: check_frame(source,i)
    result = dict(status="COMPLETE_LOCAL_EXTRACTION",id=source["id"],dataset=source["dataset"],
        config_signature=config["signature"],frames=source["frames"],duration_seconds=source["frames"]/24,
        keyframes=len(keys),captions=len(captions["captions"]),max_gap_frames=max(b-a for a,b in zip(keys,keys[1:])),
        keyframe_png_bytes=selection["keyframe_png_bytes"],
        caption_utf8_bytes=sum(len(c["caption"].encode()) for c in captions["captions"]),
        uncertainty_segments=sum(bool(c["uncertainties"]) for c in captions["captions"]),
        reviewed_changed_segments=sum(bool(c["removed_claims"]) for c in captions["captions"]),
        finished_utc=now(),structural_validation="PASS",quality_equivalence="NOT_ESTABLISHED",
        transmission_or_reconstruction="NOT_RUN",independent_hallucination_validation="NOT_RUN")
    write_json(dest/"result.json",seal(result))
    render_video(dest,source,keys,captions["captions"])
    return result


def render_video(dest,source,keys,captions):
    rows=[]
    for row in captions:
        a,b=row["start"],row["end_exclusive"]
        rows.append(f'<article><h3>{a/24:.3f}–{b/24:.3f}s · frames [{a},{b})</h3>'
          f'<img src="selected_frames/{a:05d}.png"><img src="selected_frames/{b:05d}.png">'
          f'<p>{html.escape(row["caption"])}</p><small>Uncertainty: {html.escape(str(row["uncertainties"]))}</small></article>')
    (dest/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Local FC-LGVSC extraction</title>'
        '<style>body{font:16px sans-serif;background:#151920;color:#eee;max-width:1200px;margin:30px auto}'
        'img{width:48%;margin:1%}article{border-top:1px solid #555;padding:10px}a{color:#9ce}</style>'
        f'<h1>{html.escape(source["id"])}</h1><p>Local Qwen3-VL + TransNetV2 + sequential InternVL SKEM. '
        'Not independently quality-validated. Boundary images below; captions use four interior samples.</p>'
        +''.join(rows))


def report(root,config,status="RUNNING",active=None,error=None):
    rows=[]
    for video in config["videos"]:
        dest=root/video["dataset"].lower()/video["id"]
        if (dest/"result.json").exists():
            row=unseal(dest/"result.json")
        elif (dest/"error.json").exists():
            row=dict(id=video["id"],dataset=video["dataset"],status="FAILED",error=read(dest/"error.json"))
        else:
            row=dict(id=video["id"],dataset=video["dataset"],status="PENDING")
        rows.append(row)
    counts={s:sum(r["status"]==s for r in rows) for s in {r["status"] for r in rows}}
    state=dict(status=status,updated_utc=now(),pid=os.getpid(),active=active,total=len(rows),
        completed=counts.get("COMPLETE_LOCAL_EXTRACTION",0),failed=counts.get("FAILED",0),
        counts=counts,videos=rows,error=error,config_signature=config["signature"])
    write_json(root/"status.json",state)
    lines=[f'<tr><td>{html.escape(r["id"])}</td><td>{r["status"]}</td><td>{r.get("keyframes", "")}</td>'
      f'<td>{r.get("captions", "")}</td><td>'+ (f'<a href="{r["dataset"].lower()}/{r["id"]}/index.html">Review</a>'
      if r["status"]=="COMPLETE_LOCAL_EXTRACTION" else '')+'</td></tr>' for r in rows]
    (root/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>FC-LGVSC local automatic extraction</title>'
      '<style>body{font:16px sans-serif;margin:35px}td,th{padding:7px;border-bottom:1px solid #ddd}</style>'
      f'<h1>FC-LGVSC local automatic extraction</h1><p>{status}: {state["completed"]}/{len(rows)} complete; '
      f'{state["failed"]} failed. Active: {html.escape(str(active))}. Updated: {now()}</p>'
      f'<p>{html.escape(config["model_id"])} + TransNetV2 + InternVL2-8B SKEM. Existing assistant outputs are preserved separately. '
      'Equal quality is not established. This page reports extraction only. See per-video progress.json for live stage.</p>'
      '<table><tr><th>Video</th><th>Status</th><th>Keys</th><th>Captions</th><th>Artifacts</th></tr>'+''.join(lines)+'</table>')
    return state


def code_paths():
    return [Path(__file__), REPO/"scripts/run_fc_auto_extraction.sh",
            REPO/"02_semantic_encoder/skem/MLM-keyframe-internvl.py",
            REPO/".local/fc_auto_environment.lock.txt"] + [REPO/"src/semantic_transmission"/f"{n}.py"
        for n in ["artifacts","assisted_captions","hybrid_selection","internvl_memory","exact_reuse","webvid5","cli"]]


def initialize(root,prepared,ids=None,model_id=MODEL_ID,revision=REVISION,quantization=None):
    root.mkdir(parents=True,exist_ok=True)
    if (root/"run_config.json").exists():
        raise ValueError("output already initialized; use run to resume")
    inventory=read(prepared/"inventory.json")
    videos=[]
    for v in inventory["videos"]:
        if ids and v["id"] not in ids: continue
        path=prepared/v["dataset"].lower()/v["id"]/"source_prepared.json"
        videos.append(dict(id=v["id"],dataset=v["dataset"],source_sha256=v["source_sha256"],
                           prepared_sha256=sha256(path)))
    if not videos or (ids and {v["id"] for v in videos} != set(ids)):
        raise ValueError("unknown video IDs")
    # Interleave datasets; first run three existing reference cases for comparison.
    refs=["webvid_008_852d56f2","webvid_011_65f0e6cf","tvsum_011_0e11bb5b"]
    ordered=[v for name in refs for v in videos if v["id"]==name]
    groups=[[v for v in videos if v["dataset"]==d and v["id"] not in refs] for d in ("WebVid","TVSum")]
    for n in range(max(map(len,groups))):
        for group in groups:
            if n<len(group):ordered.append(group[n])
    model_path=Path.home()/".cache/huggingface/hub"/("models--"+model_id.replace("/","--"))/"snapshots"/revision
    if not (model_path/"model.safetensors.index.json").exists():
        raise FileNotFoundError("download pinned Qwen model first")
    template=read(prepared/"webvid/webvid_011_65f0e6cf/extraction/protocol.json")
    model_files={str(p):dict(sha256=sha256(p),size=p.stat().st_size)
        for p in sorted(model_path.iterdir()) if p.is_file()}
    # Bind existing SKEM snapshot bytes to their audited manifest; validate once at launch.
    skem_files={p:{"sha256":info["sha256"],"size":info["size"]} for p,info in template["model_files"].items()}
    skem_config={k:template["config"][k] for k in ["seed","flash_attn","internvl_gpu_head",
        "internvl_offload_layers","internvl_compact_kv_cache","max_tiles","max_new_tokens"]}
    skem_config["models"]={"internvl":template["config"]["models"]["internvl"]}
    config=dict(version=1,created_utc=now(),prepared_root=str(prepared),videos=ordered,
        model_id=model_id,model_revision=revision,model_path=str(model_path),model_files=model_files,
        dtype="bfloat16",quantization=quantization,attention="sdpa",generation="greedy",seed=2025,
        candidate_stride=4,candidate_window_frames=48,max_gap_frames=24,psss_threshold=.35,
        pixel_guard=dict(minimum=.05,ratio=4.,history=24),
        skem=dict(config=skem_config,prompts=template["prompts"],model_files=skem_files),
        transnet={str(TRANSNET/n):sha256(TRANSNET/n) for n in ["weights.pth","transnetv2_pytorch.py","provenance.json"]},
        code={str(p):sha256(p) for p in code_paths()},
        prompts=dict(proposals=PROPOSAL_PROMPT,caption=CAPTION_PROMPT,review=REVIEW_PROMPT),
        local_python=str(REPO/".local/fc_auto_env/bin/python"),
        skem_python=read(REPO/".local/settings.json")["internvl_python"],
        source_inventory_sha256=sha256(prepared/"inventory.json"),
        scope="full original duration, reuse normalized full 24FPS 576x320 frames",
        quality_equivalence="NOT_ESTABLISHED",inference_provider="LOCAL_ONLY")
    config["signature"]=digest(config)
    write_json(root/"run_config.json",config)
    report(root,config,status="READY")
    print(json.dumps(dict(status="READY",videos=len(ordered),root=str(root))),flush=True)


def verify_config(config,models=False):
    if digest({k:v for k,v in config.items() if k!="signature"}) != config["signature"]:
        raise ValueError("configuration signature mismatch")
    for p,h in {**config["code"],**config["transnet"]}.items():
        if sha256(p)!=h:raise ValueError(f"frozen code/TransNet changed: {p}")
    if models:
        for p,info in {**config["model_files"],**config["skem"]["model_files"]}.items():
            if sha256(p)!=info["sha256"]:raise ValueError(f"model weights/config changed: {p}")


def worker_env():
    env=os.environ.copy()
    for k in ["LD_LIBRARY_PATH","PYTHONPATH"]:env.pop(k,None)
    env.update(PYTHONPATH=str(REPO/"src"),PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="8",TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env


def run(root,config):
    started=now()
    lock_path=REPO/".local/etri_60s_check.lock"
    with lock_path.open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU extraction lock already held")
        write_json(root/"launch.json",dict(pid=os.getpid(),started_utc=started,root=str(root)))
        verify_config(config,models=True)
        env=worker_env()
        runtime=subprocess.check_output([config["local_python"],"-c",
          "import json,sys,torch,transformers;print(json.dumps(dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.cuda.is_available())))"],env=env,text=True)
        write_json(root/"runtime.json",json.loads(runtime))
        write_json(root/"runner_ready.json",dict(pid=os.getpid(),started_utc=started,ready_utc=now()))
        for video in config["videos"]:
            dest=root/video["dataset"].lower()/video["id"]
            dest.mkdir(parents=True,exist_ok=True)
            report(root,config,active=video["id"])
            try:
                source=source_for(config,video["id"])
                if (dest/"result.json").exists():
                    verify_video(root,config,source)
                    continue
                (dest/"error.json").unlink(missing_ok=True)
                for phase in ("propose","select","caption","verify"):
                    interpreter=config["skem_python"] if phase=="select" else config["local_python"]
                    command=[interpreter,"-m","semantic_transmission.auto_extraction",phase,"--root",str(root),"--video-id",video["id"]]
                    with (dest/f"{phase}.log").open("a") as log:
                        for attempt in range(2):
                            try:
                                # Each subprocess is allowed 12h for the longest full TVSum source.
                                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                                try:
                                    returncode=child.wait(timeout=43200)
                                    if returncode:
                                        raise subprocess.CalledProcessError(returncode,command)
                                except BaseException:
                                    if child.poll() is None:
                                        child.terminate()
                                        try:child.wait(timeout=10)
                                        except subprocess.TimeoutExpired:
                                            child.kill();child.wait()
                                    raise
                                break
                            except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                                if attempt:raise
                print(json.dumps(dict(id=video["id"],status="COMPLETE",utc=now())),flush=True)
            except Exception as exc:
                write_json(dest/"error.json",dict(error=str(exc),traceback=traceback.format_exc(),utc=now()))
                print(json.dumps(dict(id=video["id"],status="FAILED",error=str(exc))),flush=True)
            report(root,config)
        state=report(root,config)
        final="COMPLETE" if state["completed"]==len(config["videos"]) else "PARTIAL_FAILED"
        report(root,config,status=final)
        print(json.dumps(dict(status=final,completed=state["completed"],failed=state["failed"])),flush=True)


def start_detached(root,config):
    """Child survives the caller/tool session; all inference output stays on disk."""
    # Probe global lock before spawning; the actual runner acquires it atomically.
    with (REPO/".local/etri_60s_check.lock").open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("another extraction/reconstruction already holds the GPU lock")
    with (root/"batch.log").open("ab",buffering=0) as log:
        child=subprocess.Popen([config["local_python"],"-m","semantic_transmission.auto_extraction",
            "run","--root",str(root)],cwd=REPO,env=worker_env(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    write_json(root/"start_request.json",dict(pid=child.pid,requested_utc=now(),log=str(root/"batch.log")))
    print(json.dumps(dict(status="START_REQUESTED",pid=child.pid,root=str(root),
                         log=str(root/"batch.log"))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["init","start","run","propose","select","caption","verify","status"])
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--prepared",type=Path,default=DEFAULT_PREPARED)
    parser.add_argument("--video-id",action="append")
    parser.add_argument("--model-id",default=MODEL_ID)
    parser.add_argument("--revision",default=REVISION)
    parser.add_argument("--quantization",choices=["int8","nf4"])
    args=parser.parse_args()
    root=args.root.resolve()
    if args.action=="init":return initialize(root,args.prepared.resolve(),args.video_id,
                                            args.model_id,args.revision,args.quantization)
    config=read(root/"run_config.json")
    if args.action=="status":
        state=read(root/"status.json");state.pop("videos",None);print(json.dumps(state,indent=2));return
    verify_config(config)
    if args.action=="start":return start_detached(root,config)
    if args.action=="run":
        def stop(signum,frame):
            raise KeyboardInterrupt(f"stop requested by signal {signum}")
        signal.signal(signal.SIGTERM,stop)
        try:return run(root,config)
        except BaseException as exc:
            report(root,config,status="STOPPED",error=str(exc))
            raise
    if not args.video_id or len(args.video_id)!=1:raise ValueError("worker requires one video-id")
    source=source_for(config,args.video_id[0])
    {"propose":propose,"select":select,"caption":caption,"verify":verify_video}[args.action](root,config,source)


if __name__=="__main__":main()
