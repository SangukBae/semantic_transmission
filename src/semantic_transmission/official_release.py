"""Hash-pinned author scripts with a closed list of memory-only adaptations.

No endpoint padding, schedule repair, mask repair, tail17 or model reuse is
installed here. The generated scripts and exact patches are retained per run.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

from .artifacts import sha256, write_json

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs/lgvsc_official_sources.json"
SELECTOR = "02_semantic_encoder/skem/MLM-keyframe-internvl.py"
DECODER = "04_semantic_decoder/scripts/mydemo_new_align_sh.py"


def read(path):
    return json.loads(Path(path).read_text())


def source_path(name, repo=REPO):
    manifest = read(repo / "configs/lgvsc_official_sources.json")
    expected = manifest["files"][name]["sha256"]
    path = repo / ".local/vendor/LGVSC-official" / manifest["commit"] / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned source {name}; run scripts/run_lgvsc_official.sh fetch")
    if sha256(path) != expected:
        raise ValueError(f"Pinned author source changed: {path}")
    return path


def fetch(repo=REPO):
    manifest = read(repo / "configs/lgvsc_official_sources.json")
    for name, item in manifest["files"].items():
        path = repo / ".local/vendor/LGVSC-official" / manifest["commit"] / name
        if path.exists():
            source_path(name, repo)
            continue
        with urllib.request.urlopen(item["url"], timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Author download hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    return manifest


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Unsupported upstream source at patch anchor: {before[:90]!r}")
    return text.replace(before, after, 1)


def selector_source(text, cpu_layers=3):
    # Stage BF16 layers without changing prompts, candidate list, probability
    # arithmetic, generation settings or source ordering. Even the original
    # zero-probability division behaviour is retained.
    text = replace_once(text, "    model.cuda()", "\n".join([
        "    from semantic_transmission.internvl_memory import place_internvl",
        f"    model = place_internvl(model, gpu_head=True, cpu_layers={cpu_layers}, compact_cache=True)"]))
    text = replace_once(text, "            qbar.update(1)", "\n".join([
        "            qbar.update(1)",
        "            from semantic_transmission.validation_progress import emit",
        "            emit(qbar.n, len(all_frames))"]))
    return text


def decoder_source(text):
    text = replace_once(text,
        "    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)",
        "\n".join([
            "    from semantic_transmission.official_release import cpu_text_encoder",
            "    text_encoder = cpu_text_encoder(build_module(cfg.text_encoder, MODELS, device='cpu'), device)",
            "    from semantic_transmission.decoder_memory import install",
            "    memory_record = install()",
            "    from semantic_transmission.artifacts import write_json",
            "    from semantic_transmission.official_release import trace_references",
            "    append_generated = trace_references(append_generated, os.environ['LGVSC_OFFICIAL_TRACE'])"]))
    # The original rebuilds the diffusion model at every interval. Keep that
    # and its RNG consumption, but free the previous model before constructing
    # its replacement instead of temporarily holding two complete models.
    text = replace_once(text, "                    model = (", "\n".join([
        "                    if 'model' in locals():",
        "                        text_encoder.y_embedder = None",
        "                        del model",
        "                    model = ("]))
    text = replace_once(text, "vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit",
        "vae, video_clips[-1].to(device=device, dtype=dtype), refs, ms, loop_i, condition_frame_length, condition_frame_edit")
    text = replace_once(text, "                    video_clips.append(samples)", "\n".join([
        "                    video_clips.append(samples.cpu())",
        "                    del samples",
        "                    write_json(os.environ['LGVSC_OFFICIAL_MEMORY'], memory_record)",
        "                    from semantic_transmission.validation_progress import emit",
        "                    emit(loop_i + 1, loop)"]))
    # An observer retains lossless pixels without changing the official
    # concatenation or save_sample input. No boundary frames are removed.
    text = replace_once(text, "                        video = torch.cat(video, dim=1)", "\n".join([
        "                        video = torch.cat(video, dim=1)",
        "                        from semantic_transmission.official_release import save_frames",
        "                        save_frames(video, save_path + '_frames')"]))
    return text


def cpu_text_encoder(encoder, device):
    original = encoder.encode

    def encode(text):
        # Preserve FP32 T5 outputs; the original STDiT handles its own cast.
        return {name: value.to(device=device) for name, value in original(text).items()}

    encoder.encode = encode
    return encoder


def trace_references(original, path):
    records = []
    write_json(path, records)

    def append(vae, video, refs, strategies, loop, length, edit):
        result = original(vae, video, refs, strategies, loop, length, edit)
        records.append(dict(loop=loop, input_frames=video.shape[2],
                            reference_shapes=[list(row[-1].shape) for row in result[0]],
                            policy="whole_previous_segment_before_vae"))
        write_json(path, records)
        return result

    return append


def save_frames(video, path):
    import torch
    from PIL import Image
    path = Path(path)
    path.mkdir(exist_ok=False)
    for i in range(video.shape[1]):
        pixels = (video[:, i].clamp(-1, 1).add(1).div(2).mul(255).add(.5)
                  .clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8))
        Image.fromarray(pixels.numpy()).save(path / f"{i:05d}.png")


def materialize(directory, repo=REPO):
    import difflib
    directory.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, transform in [(SELECTOR, selector_source), (DECODER, decoder_source)]:
        original = source_path(name, repo).read_text()
        modified = transform(original)
        compile(modified, name, 'exec')
        target = directory / Path(name).name
        target.write_text(modified)
        patch = ''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                            fromfile=name, tofile=target.name))
        target.with_suffix('.patch').write_text(patch)
        records[name] = dict(upstream_sha256=sha256(source_path(name, repo)),
                             executed_sha256=sha256(target))
    config_utils = source_path('04_semantic_decoder/scripts/config_utils.py', repo)
    (directory / 'config_utils.py').write_bytes(config_utils.read_bytes())
    write_json(directory / 'patches.json', dict(scripts=records,
        allowed=['BF16 layer staging', 'FP32 T5 CPU placement', 'CPU video storage',
                 'release previous model before rebuilding', 'independent attention/MLP chunking',
                 'progress/reference/lossless-frame observers'],
        disabled=['tail17', 'short_reference_padding', 'condition_collision_repair',
                  'short_schedule_repair', 'endpoint_exact', 'diffusion_model_reuse']))
