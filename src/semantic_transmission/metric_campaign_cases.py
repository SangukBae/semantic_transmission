"""Follow-up corpora: crossed errors, RX counterfactuals and annotated footage.

Truth comes from renderer state, edit index maps, insertion masks and existing
DAVIS masks. Candidate observers receive only RGB and receiver decode status.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .artifacts import sha256
from .metric_campaign import REPO, read, save
from .metric_v2_cases import temporal_variant
from .metric_v3_cases import annotation_events, public_truth
from .metric_v5_cases import FAMILIES, TARGETS, draw, scene, variants, truth_order_inversions
from .forbidden_state_metric import truth_occupancy
from .sta_video_validation import encode_video, receive

PRIMARY = ("fso_presence", "fso_premature", "fso_motion", "fso_heading", "eoi", "uep")


def appearance(frames, style):
    if style == "identity":
        return frames.copy()
    if style == "brightness20":
        return np.clip(frames.astype(float) * 1.15 + 20, 0, 255).astype(np.uint8)
    out = []
    h, w = frames.shape[1:3]
    yy, xx = np.mgrid[:h, :w]
    for t, frame in enumerate(frames):
        if style == "texture":
            value = np.clip(frame.astype(float) + (20 * np.sin(xx * .4) * np.cos(yy * .29))[..., None], 0, 255).astype(np.uint8)
        elif style == "palette":
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int) + 45) % 180
            value = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif style == "camera":
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5 * np.sin(t * .4), 1.)
            matrix[:, 2] += [2 * np.sin(t * .8), 2 * np.cos(t * .6)]
            value = cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        else:
            raise ValueError(style)
        out.append(value)
    return np.stack(out)


def insert_object(frames, seed, length):
    """A new, code-labelled polygon with varying location, size, colour and time."""
    rng = np.random.default_rng(seed)
    n, h, w = frames.shape[:3]
    start = int(rng.integers(3, n - length - 2))
    cx, cy = rng.uniform(.10, .22) * w, rng.uniform(.66, .85) * h
    radius = rng.uniform(.065, .095) * min(w, h)
    angles = np.arange(10) * np.pi / 5
    radii = np.array([radius, radius * .55] * 5)
    points = np.rint(np.c_[cx + radii * np.cos(angles), cy + radii * np.sin(angles)]).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [points], 1)
    color = rng.integers(100, 240, 3)
    rec = frames.copy()
    masks = np.zeros((n, h, w), bool)
    yy, xx = np.mgrid[:h, :w]
    pixels = np.clip(color + (12 * np.sin(xx * .3) * np.cos(yy * .4))[..., None], 0, 255).astype(np.uint8)
    for t in range(start, start + length):
        rec[t, mask > 0] = pixels[mask > 0]
        masks[t] = mask > 0
    return rec, masks


class Corpus:
    def __init__(self, root, phase):
        self.root, self.phase = Path(root), phase
        self.rows, self.artifacts = [], set()

    def put(self, frames):
        frames = np.ascontiguousarray(frames)
        key = hashlib.sha256(frames.tobytes()).hexdigest()
        path = self.root / "arrays" / (key + ".npz")
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            with np.load(path) as f:
                if not np.array_equal(f["frames"], frames):
                    raise ValueError("array hash collision or modified artifact: " + str(path))
        else:
            temp = path.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, frames=frames)
            temp.replace(path)
        self.artifacts.add(path)
        return {"path": str(path), "pixel_sha256": key, "file_sha256": sha256(path)}

    def add(self, sid, name, a, b, truth, *, style="identity", family="natural", rx=None, status=None, **extra):
        row = {"case_id": sid.replace("/", "__") + "__" + name, "source_id": sid,
               "family": family, "style": style, "source": self.put(a), "reconstruction": self.put(b),
               "truth": truth, **extra}
        if rx is not None:
            row.update(rx=self.put(rx), rx_status=status)
        self.rows.append(row)

    def finish(self, **metadata):
        path = self.root / self.phase / "cases.json"
        save(path, {"rows": self.rows, "count": len(self.rows), **metadata})
        self.artifacts.add(path)
        return sorted(self.artifacts)


def occupancy_truth(source, reconstruction):
    """Target presence and magnitude from independent annotated kinematics.

    Null denotes no eligible annotated query, not zero error. DAVIS annotates
    selected objects only, so unknown new objects are never labelled absent.
    """
    values = {name: [] for name in PRIMARY if name.startswith("fso_")}
    events = annotation_events(source["centers"], source["visible"])
    for event in events:
        for instance, (kind, _) in TARGETS.items():
            if event["type"] != kind:
                continue
            frame = round(event["time_s"] * 8)
            args = (event["object_id"], frame)
            a = truth_occupancy(instance, source["centers"], source["visible"], *args)["occupancy"]
            b = truth_occupancy(instance, reconstruction["centers"], reconstruction["visible"], *args)["occupancy"]
            if a is not None and b is not None:
                values["fso_" + instance].append(max(0., b - a))
    out = {name: max(v) if v else None for name, v in values.items()}
    out["eoi"] = truth_order_inversions(source, reconstruction)["truth_eoi"]
    return out


def cross_corpus(root, config):
    corpus = Corpus(root, "01_cross")
    previous = set()
    for p in (REPO / "outputs").glob("*/cases.jsonl"):
        for line in p.read_text().splitlines():
            import json
            row = json.loads(line)
            if row.get("source_pixel_sha256"):
                previous.add(row["source_pixel_sha256"])
    for fi, family in enumerate(FAMILIES):
        for k in range(config["synthetic_sources_per_family"]):
            seed = config["synthetic_seed"] + fi * 100 + k
            state = scene(seed, family)
            a, source_truth = draw(state)
            if hashlib.sha256(a.tobytes()).hexdigest() in previous:
                raise ValueError("synthetic source already used")
            sid = family + "/" + str(seed)
            for name, b, gt, meta in variants(state):
                # Identity is the common source; all nuisances are crossed with
                # every error, rather than applied exclusively to normal cases.
                if meta["target"] == "control" and name != "identity":
                    continue
                truth = occupancy_truth(source_truth, gt)
                truth["uep"] = 0. if meta["target"] == "control" else None
                for style in config["styles"]:
                    if style == "texture":
                        rec, _ = draw(state, gt["centers"], gt["visible"], style="texture")
                    else:
                        rec = appearance(b, style)
                    corpus.add(sid, name + "_" + style, a, rec, truth, family=family, style=style,
                               rx=a, status=["ok"] * len(a), kind=meta["kind"],
                               truth_provenance="renderer kinematics; nuisance transformations preserve world state")
            for length in (2, 4):
                b, mask = insert_object(a, seed, length)
                truth = {m: None for m in PRIMARY}
                truth["uep"] = min(1., float(mask.any((1, 2)).sum()) / 4)
                for style in config["styles"]:
                    corpus.add(sid, f"addition_{length}_{style}", a, appearance(b, style), truth,
                               family=family, style=style, rx=a, status=["ok"] * len(a), kind="addition",
                               truth_provenance="code-recorded insertion mask")
            print("prepared cross", sid, len(corpus.rows), flush=True)
    return corpus.finish(scope="new seeds in the existing three procedural families; not unseen-family generalisation")


def rx_corpus(root, config):
    corpus = Corpus(root, "02_components")
    for fi, family in enumerate(FAMILIES):
        for k in range(config["rx_sources_per_family"]):
            seed = config["rx_seed"] + fi * 100 + k
            a, _ = draw(scene(seed, family))
            sid = "rx_" + family + "/" + str(seed)
            for length in (2, 4, 8):
                b, masks = insert_object(a, seed, length)
                occupied = np.flatnonzero(masks.any((1, 2)))
                for mode in ("absent", "supported", "partial", "corrupt", "sparse"):
                    supplied = a.copy()
                    support = np.zeros(len(a), bool)
                    if mode in ("supported", "corrupt", "sparse"):
                        supplied = b.copy()
                        support[occupied] = True
                    elif mode == "partial":
                        chosen = occupied[:max(1, len(occupied) // 2)]
                        supplied[chosen] = b[chosen]
                        support[chosen] = True
                    packets = encode_video(supplied)
                    if mode == "corrupt":
                        for t in occupied:
                            raw = bytearray(packets[t]["payload"])
                            raw[len(raw) // 2] ^= 1
                            packets[t] = {**packets[t], "payload": bytes(raw)}
                    elif mode == "sparse":
                        packets = [p if t % 4 == 0 else None for t, p in enumerate(packets)]
                    rx, status, indices = receive(packets)
                    supported = {t for t in occupied if support[t] and status[t] == "ok"}
                    oracle = min(1., len(set(occupied) - supported) / 4)
                    corpus.add(sid, f"{length}_{mode}", a, b, {"uep": oracle}, family=family,
                               rx=rx, status=status, kind=mode, counterfactual_group=sid + "/" + str(length),
                               truth_provenance="insertion mask intersected with successfully decoded PNG support slots",
                               decoded_source_indices=indices)
            print("prepared RX", sid, len(corpus.rows), flush=True)
    return corpus.finish(scope="fixed source/reconstruction, changed receiver evidence; dense and sparse PNG surrogates")


def natural_corpus(root, config, inventory):
    corpus = Corpus(root, "03_natural")
    source_rows = []
    for index, item in enumerate(inventory):
        frames, masks = [], []
        for image, label in zip(item["images"], item["labels"]):
            rgb = cv2.imread(image)
            # DAVIS PNGs use indexed palettes. OpenCV expands those to BGR
            # colours, which would destroy the annotated object identities.
            with Image.open(label) as annotated:
                mask = np.asarray(annotated).copy()
            if rgb is None or mask.ndim != 2:
                raise ValueError("unreadable DAVIS input")
            frames.append(cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (320, 192), interpolation=cv2.INTER_AREA))
            masks.append(cv2.resize(mask, (320, 192), interpolation=cv2.INTER_NEAREST))
        a, labels = np.stack(frames), np.stack(masks)
        ids = [i for i in np.unique(labels) if 0 < i < 255]
        relabelled = np.zeros_like(labels)
        for j, identity in enumerate(ids, 1):
            relabelled[labels == identity] = j
        gt = public_truth(relabelled)
        sid = item["source_id"]
        source_rows.append({"source_id": sid, "array": corpus.put(a), "source_annotations": item["labels"]})
        truth = occupancy_truth(gt, gt)
        truth["uep"] = 0.
        for style in config["styles"]:
            corpus.add(sid, "normal_" + style, a, appearance(a, style), truth, style=style,
                       rx=a, status=["ok"] * len(a), kind="control", truth_provenance="unchanged original and annotation state")
        for kind in ("reverse", "freeze", "lag", "swap"):
            for severity in (.125, .25, .5):
                b, meta = temporal_variant(a, kind, severity)
                indices = meta["source_index_map"]
                changed = {key: value[indices] for key, value in gt.items()}
                truth = occupancy_truth(gt, changed)
                # These edits are not certified negative examples for UEP:
                # selected DAVIS masks do not exhaustively annotate every object.
                truth["uep"] = None
                for style in config["styles"]:
                    corpus.add(sid, f"{kind}_{severity}_{style}", a, appearance(b, style), truth,
                               style=style, rx=a, status=["ok"] * len(a), kind=kind,
                               truth_provenance="existing DAVIS masks transformed by the known edit index map")
        for length in (2, 4):
            b, mask = insert_object(a, config["natural_seed"] + index, length)
            for style in config["styles"]:
                corpus.add(sid, f"addition_{length}_{style}", a, appearance(b, style),
                           {"uep": min(1., float(mask.any((1, 2)).sum()) / 4)}, style=style,
                           rx=a, status=["ok"] * len(a), kind="addition", truth_provenance="known inserted polygon, not an exhaustive natural-object annotation")
        print("prepared natural", sid, len(corpus.rows), flush=True)
    sources_path = root / "03_natural" / "sources.json"
    save(sources_path, source_rows)
    return corpus.finish(timeline=f"every {config['natural_stride']} DAVIS JPEG, assigned 8 Hz benchmark playback; native capture FPS unknown",
                         scope="controlled edits of unused natural footage; separate from generated reconstruction validity") + [sources_path]
