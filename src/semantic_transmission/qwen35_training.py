"""Resumable stage-wise and joint training for the reviewed local pilot."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

from .qwen35_training_data import ROOT, DEFAULT_DATA, read_json, write_json, sha256, verify

CONFIG = ROOT / "configs/qwen35_multitask_v1.json"
RUN = ROOT / "outputs/qwen35_multitask_v1"


def engine():
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen35_advanced_caption
    return qwen35_advanced_caption


def rows_for(data, split):
    return read_json(Path(data) / f"{split}.json")["records"]


def images_for(row, indices):
    from PIL import Image
    return [Image.open(row["frames"][int(i)]["path"]).convert("RGB") for i in indices]


def labels_for(row, indices):
    return [f"i{int(i)}, source frame {row['frames'][int(i)]['index']}, {row['frames'][int(i)]['index']/row['fps']:.3f} seconds" for i in indices]


def parsed_selection(text, n=8, k=4):
    match = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(match.group(0) if match else text)
    keys = obj["keys"]
    if (len(keys) != k or any(type(x) is not int for x in keys) or len(set(keys)) != k
            or keys != sorted(keys) or keys[0] != 0 or keys[-1] != n - 1):
        raise ValueError("Invalid key indices")
    if not all(0 <= x < n for x in keys):
        raise ValueError("Out of range")
    return obj


def baseline(args, config):
    from .qwen35_multitask import coverage
    out = args.run / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists():
        from .qwen35_training_eval import verify_baseline
        frozen = verify_baseline(args.data)
        if frozen["config"] != config:
            raise ValueError("Baseline is frozen with different conditions")
        print("Frozen baseline verified; no generation or overwrite performed", flush=True)
        return
    protocol = {"data_freeze_sha256": sha256(args.data / "freeze.json"), "config": config,
                "scope": "Untrained pinned Qwen proposes and selects from eight images, then captions four selected images; source-disjoint AI-reviewed pilot",
                "selection_failure_policy": "record_failure_and_uniform_0_2_5_7_no_retry",
                "weights_trained": False}
    if (out / "protocol.json").exists():
        if read_json(out / "protocol.json") != protocol:
            raise ValueError("Baseline conditions changed; choose a new run directory")
    else:
        write_json(out / "protocol.json", protocol)
    work = rows_for(args.data, "validation") + rows_for(args.data, "test")
    pending = [r for r in work if not (out / "records" / f"{r['id']}.json").exists()]
    eng = engine()
    model, processor, runtime = eng.load("nf4") if pending else (None, None, None)
    for row in pending:
        started = time.monotonic()
        selection = eng.generate(model, processor, images_for(row, range(8)), labels_for(row, range(8)), config["selection_prompt"], config["max_pixels"], 256)
        error = None
        try:
            if selection["truncated"]:
                raise ValueError("Selection response truncated")
            proposal = parsed_selection(selection["caption"])
        except (ValueError, KeyError, TypeError) as e:
            error = str(e)
            proposal = {"events": [], "keys": [0, 2, 5, 7]}
        keys = proposal["keys"]
        caption = eng.generate(model, processor, images_for(row, keys), labels_for(row, keys), config["caption_prompt"], config["max_pixels"], config["max_caption_tokens"])
        result = {"id": row["id"], "split": row["split"], "keys": keys,
                  "source_key_indices": [row["frames"][i]["index"] for i in keys],
                  "proposal": proposal, "selection_error": error, "selection_call": selection,
                  "caption": caption["caption"], "caption_call": caption,
                  "total_seconds": time.monotonic() - started, "coverage": coverage(row, keys),
                  "model_runtime": runtime, "protocol_sha256": sha256(out / "protocol.json")}
        write_json(out / "records" / f"{row['id']}.json", result)
        print(json.dumps({"baseline": row["id"], "seconds": result["total_seconds"], "selection_error": error}), flush=True)
    records = [read_json(out / "records" / f"{r['id']}.json") for r in work]
    for r in records:
        if r["protocol_sha256"] != sha256(out / "protocol.json"):
            raise ValueError("Mismatched cached baseline")
    result = {"status": "GENERATED_CAPTION_REVIEW_PENDING", "splits": {}}
    for split in ["validation", "test"]:
        part = [r for r in records if r["split"] == split]
        result["splits"][split] = {"windows": len(part), "keys": sum(len(r["keys"]) for r in part),
            "event_hits": sum(r["coverage"]["event_hits"] for r in part), "event_count": sum(r["coverage"]["event_count"] for r in part),
            "generation_seconds": sum(r["total_seconds"] for r in part),
            "selection_failures": sum(r["selection_error"] is not None for r in part),
            "caption_failures": sum(r["caption_call"]["status"] != "CAPTION_COMPLETE" for r in part)}
    write_json(out / "RESULT.json", result)
    write_json(out / "generation_freeze.json", {"protocol_sha256": sha256(out / "protocol.json"),
        "records": {f"records/{r['id']}.json": sha256(out / "records" / f"{r['id']}.json") for r in records}})


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_gpu():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def tensor_save(path, obj):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def cache_protocol(args, config):
    from .qwen35_backend import SNAPSHOT
    return {"freeze_sha256": sha256(args.data / "freeze.json"), "revision": config["revision"],
        "max_pixels": config["max_pixels"], "dtype": "bfloat16", "encoder": "frozen_Qwen_visual_pooler_output",
        "processor_config_sha256": sha256(SNAPSHOT / "preprocessor_config.json")}


def ensure_cache(args, config, splits=("train", "validation")):
    import torch
    cache = args.data / "features"
    expected = cache_protocol(args, config)
    protocol_path = cache / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != expected:
        raise ValueError("Feature cache conditions differ")
    write_json(protocol_path, expected)
    rows = [r for split in splits for r in rows_for(args.data, split)]
    pending = []
    for row in rows:
        path = cache / f"{row['id']}.pt"
        meta = path.with_suffix(".json")
        if not (path.exists() and meta.exists()):
            pending.append(row)
        elif read_json(meta)["sha256"] != sha256(path) or read_json(meta)["protocol_sha256"] != sha256(protocol_path):
            raise ValueError(f"Corrupt/stale cached features: {row['id']}")
    if not pending:
        return
    eng = engine()
    model, processor, runtime = eng.load("nf4")
    for row in pending:
        start = time.monotonic()
        # No captions, facts, events or split labels enter feature extraction.
        inputs = eng.base.preprocess(processor, images_for(row, range(8)), labels_for(row, range(8)), "Describe the visible frames.", config["max_pixels"]).to("cuda")
        with torch.no_grad():
            visual = model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
            tokens = torch.stack(visual.pooler_output).detach().cpu().to(torch.bfloat16)
        if tokens.ndim != 3 or tokens.shape[0] != 8 or not torch.isfinite(tokens).all():
            raise ValueError("Unexpected/nonfinite visual features")
        path = cache / f"{row['id']}.pt"
        tensor_save(path, {"tokens": tokens, "grid": inputs.image_grid_thw.cpu(),
            "pooled": tokens.float().mean(1), "frame_hashes": [f["sha256"] for f in row["frames"]]})
        write_json(path.with_suffix(".json"), {"sha256": sha256(path), "protocol_sha256": sha256(protocol_path),
            "shape": list(tokens.shape), "seconds": time.monotonic() - start, "runtime": runtime})
        print(json.dumps({"cache": row["id"], "shape": list(tokens.shape)}), flush=True)
        del inputs, visual, tokens
    del model, processor
    release_gpu()


def load_features(args, row):
    import torch
    obj = torch.load(args.data / "features" / f"{row['id']}.pt", map_location="cpu", weights_only=True)
    if obj["frame_hashes"] != [f["sha256"] for f in row["frames"]]:
        raise ValueError("Feature order differs from frozen frames")
    return obj


def load_lora(config, adapter=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    model, processor, runtime = engine().load("nf4")
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        # Avoid PEFT's default promotion of frozen vision/embedding/head to FP32.
        if "norm" in name and "visual" not in name and p.dtype in (torch.bfloat16, torch.float16):
            p.data = p.data.float()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    else:
        targets = [n for n, m in model.named_modules() if "language_model.layers." in n and m.__class__.__name__ == "Linear4bit"]
        if not targets:
            raise ValueError("No pinned Qwen LoRA targets found")
        model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=0., target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    model.base_model.model.model.visual.eval()
    return model, processor, runtime


def caption_loss(model, processor, head, prediction, features, row, config):
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    keys = prediction["indices"].tolist()
    images = images_for(row, keys)
    messages = engine().base.messages_for(images, labels_for(row, keys), config["caption_prompt"])
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    kwargs = {"images": images, "return_tensors": "pt", "images_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": config["max_pixels"]}}}
    inputs = processor(text=[prefix + row["caption"] + "<|im_end|>"], **kwargs)
    prefix_inputs = processor(text=[prefix], **kwargs)
    prefix_len = prefix_inputs.input_ids.shape[1]
    if not torch.equal(inputs.input_ids[:, :prefix_len], prefix_inputs.input_ids):
        raise ValueError("Caption prefix token boundary changed")
    if inputs.input_ids.shape[1] > config["max_input_tokens"]:
        raise ValueError("Sample exceeds validated training token budget")
    if not torch.equal(inputs.image_grid_thw, features["grid"][keys]):
        raise ValueError("Cached vision grid differs from current processor")
    inputs.pop("pixel_values")
    inputs = inputs.to("cuda")
    mixed = head.select_tokens(features["tokens"].to("cuda"), prediction)
    positions = torch.arange(prefix_len - 1, inputs.input_ids.shape[1] - 1, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**inputs, use_cache=False, logits_to_keep=positions,
            mm_encoder_outputs={"image": BaseModelOutputWithPooling(pooler_output=tuple(mixed.unbind(0)))})
        loss = torch.nn.functional.cross_entropy(output.logits.float().flatten(0, 1), inputs.input_ids[:, prefix_len:].flatten())
    return loss


def checkpoint_path(stage_dir, pointer="best"):
    record = read_json(stage_dir / f"{pointer}.json")
    path = stage_dir / record["directory"]
    if sha256(path / "complete.json") != record["complete_sha256"]:
        raise ValueError("Checkpoint manifest changed")
    for name, digest in read_json(path / "complete.json")["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Checkpoint changed: {path / name}")
    return path


def save_checkpoint(stage_dir, head, model, optimizer, state, best=False):
    import torch
    name = f"update-{state['updates']:06d}-e{state['next_epoch']:03d}-n{state['next_offset']:04d}"
    dest = stage_dir / name
    temp = stage_dir / (name + ".partial")
    if temp.exists():
        # A prior interrupted write is never a resumable checkpoint.
        import shutil
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    tensor_save(temp / "head.pt", head.state_dict())
    if model is not None:
        model.save_pretrained(temp / "adapter", safe_serialization=True)
    tensor_save(temp / "optimizer.pt", optimizer.state_dict())
    tensor_save(temp / "rng.pt", {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "python": random.getstate()})
    write_json(temp / "state.json", state)
    write_json(temp / "complete.json", {"files": {str(p.relative_to(temp)): sha256(p) for p in temp.rglob("*") if p.is_file()}})
    if dest.exists():
        raise ValueError(f"Refusing to overwrite a checkpoint: {dest}")
    temp.rename(dest)
    pointer = {"directory": name, "complete_sha256": sha256(dest / "complete.json")}
    write_json(stage_dir / "latest.json", pointer)
    if best:
        write_json(stage_dir / "best.json", pointer)
    return dest


def phase(args, config, stage, smoke=False, stop_after_updates=None):
    import torch
    from .qwen35_multitask import TemporalSelector, supervised_head_loss, coverage
    stage_dir = args.run / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    identity = {"data_freeze_sha256": sha256(args.data / "freeze.json"), "config": config,
                "stage": stage, "smoke_only": smoke, "implementation_sha256": sha256(Path(__file__)),
                "selector_code_sha256": sha256(Path(__file__).with_name("qwen35_multitask.py"))}
    protocol_path = stage_dir / "protocol.json"
    if protocol_path.exists():
        if read_json(protocol_path) != identity:
            raise ValueError("Training conditions/code changed; choose a new run")
        if not args.resume:
            raise ValueError(f"Run exists: {stage_dir}; use --resume or a new --run")
    else:
        write_json(protocol_path, identity)
    if (stage_dir / "RESULT.json").exists() and read_json(stage_dir / "RESULT.json")["status"] == "COMPLETE":
        checkpoint_path(stage_dir)
        print(f"Already complete: {stage}", flush=True)
        return
    train, validation = rows_for(args.data, "train"), rows_for(args.data, "validation")
    if smoke:
        # Include a dynamic source and a static source; no test examples are trained.
        train = [r for r in train if r["video_id"].startswith(("webvid_005", "webvid_007"))]
        validation = [r for r in validation if r["video_id"].startswith("webvid_024")][:2]
    device = "cpu" if stage == "heads" else "cuda"
    seed_all(config["seed"])
    head = TemporalSelector(hidden=config["head_hidden"], candidates=config["candidates"], keys=config["keys"]).to(device)
    adapter = None
    resume_path = checkpoint_path(stage_dir, "latest") if (stage_dir / "latest.json").exists() else None
    source_stage = "heads" if stage == "caption" else "caption" if stage == "joint" else None
    if source_stage and not resume_path:
        previous = args.run / source_stage
        if not (previous / "RESULT.json").exists() or read_json(previous / "RESULT.json")["status"] != "COMPLETE":
            raise ValueError(f"Complete {source_stage} training first")
        previous_protocol = read_json(previous / "protocol.json")
        if previous_protocol["data_freeze_sha256"] != identity["data_freeze_sha256"] or previous_protocol["config"] != config or previous_protocol["smoke_only"] != smoke:
            raise ValueError("Previous training stage used different data/configuration")
    source = resume_path or (checkpoint_path(args.run / source_stage) if source_stage else None)
    if source is not None:
        head.load_state_dict(torch.load(source / "head.pt", weights_only=True, map_location=device))
        adapter = source / "adapter" if (source / "adapter").exists() else None
    head.requires_grad_(stage != "caption")
    model, processor, runtime = (None, None, None) if stage == "heads" else load_lora(config, adapter)
    if model is not None:
        # The loader seeds its own initialization; set the experiment RNG afterward.
        seed_all(config["seed"])
    groups = []
    if stage != "caption":
        groups.append({"params": list(head.parameters()), "lr": config["joint_head_lr"] if stage == "joint" else config["head_lr"]})
    if model is not None:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": config[f"{stage}_lr"]})
    optimizer = torch.optim.AdamW(groups)
    params = [p for g in groups for p in g["params"]]
    epochs = 1 if smoke else config[{"heads": "head_epochs", "caption": "caption_epochs", "joint": "joint_epochs"}[stage]]
    state = {"stage": stage, "updates": 0, "next_epoch": 0, "next_offset": 0,
             "best_validation_loss": None, "history": [], "protocol_sha256": sha256(protocol_path),
             "caption_gradient_to_selection": None, "caption_gradient_to_candidates": None,
             "weights_changed": False, "resumed": False}
    if resume_path:
        state = read_json(resume_path / "state.json")
        if state["protocol_sha256"] != sha256(protocol_path):
            raise ValueError("Resume protocol mismatch")
        optimizer.load_state_dict(torch.load(resume_path / "optimizer.pt", weights_only=True, map_location=device))
        rng = torch.load(resume_path / "rng.pt", weights_only=True, map_location="cpu")
        torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state_all(rng["cuda"]); random.setstate(rng["python"])
        state["resumed"] = True
    pooled = {r["id"]: load_features(args, r)["pooled"].to(device) for r in train + validation}
    start = time.monotonic()
    start_update = state["updates"]
    write_json(stage_dir / "status.json", {"status": "RUNNING", "stage": stage, "updates": state["updates"], "smoke_only": smoke})
    if model is not None:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(state["next_epoch"], epochs):
        order = list(train)
        random.Random(config["seed"] + epoch).shuffle(order)
        offset = state["next_offset"] if epoch == state["next_epoch"] else 0
        while offset < len(order):
            batch = order[offset:offset + config["gradient_accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row in batch:
                prediction = head(pooled[row["id"]], config["temperature"])
                aux, components = supervised_head_loss(head, prediction, row)
                if model is None:
                    loss = aux
                else:
                    ce = caption_loss(model, processor, head, prediction, load_features(args, row), row, config)
                    if stage == "joint" and state["caption_gradient_to_selection"] is None:
                        grads = torch.autograd.grad(ce, [head.selection.weight, head.candidate.weight], retain_graph=True)
                        norms = [float(g.norm()) for g in grads]
                        if not all(math.isfinite(n) and n > 0 for n in norms):
                            raise RuntimeError("Caption loss is disconnected from candidate/selection heads")
                        state["caption_gradient_to_selection"], state["caption_gradient_to_candidates"] = norms
                    loss = ce + config["head_loss_weight"] * aux if stage == "joint" else ce
                    components["caption_ce"] = float(ce.detach())
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
                del loss, prediction, aux
                if model is not None:
                    del ce
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Nonfinite/zero optimizer gradient")
            changed_param = next(p for p in params if p.grad is not None and p.grad.abs().max() > 0)
            before = changed_param.detach().clone()
            optimizer.step()
            state["weights_changed"] = state["weights_changed"] or not torch.equal(before, changed_param)
            del before
            offset += len(batch)
            state.update(updates=state["updates"] + 1, next_epoch=epoch, next_offset=offset)
            record = {"epoch": epoch, "update": state["updates"], "loss": sum(losses) / len(losses),
                      "gradient_norm": float(norm), **components}
            state["history"].append(record)
            write_json(stage_dir / "status.json", {"status": "RUNNING", "stage": stage, "updates": state["updates"], "epoch": epoch, "smoke_only": smoke})
            print(json.dumps({stage: record}), flush=True)
            # A controlled smoke interruption proves the actual checkpoint resume path.
            if stop_after_updates and state["updates"] - start_update >= stop_after_updates and offset < len(order):
                save_checkpoint(stage_dir, head, model, optimizer, state)
                write_json(stage_dir / "RESULT.json", {"status": "INTERRUPTED_FOR_RESUME_TEST", "updates": state["updates"]})
                write_json(stage_dir / "status.json", {"status": "CHECKPOINTED_RESUME_TEST", "stage": stage, "updates": state["updates"]})
                del optimizer, params, groups, model, processor, head, pooled, changed_param
                release_gpu()
                return
            if offset < len(order) and state["updates"] % config["checkpoint_every_updates"] == 0:
                save_checkpoint(stage_dir, head, model, optimizer, state)
        head.eval()
        if model is not None:
            model.eval()
        val_losses, event_hits, event_count = [], 0, 0
        with torch.no_grad():
            for row in validation:
                pred = head(pooled[row["id"]], config["temperature"])
                aux, _ = supervised_head_loss(head, pred, row)
                loss = aux if model is None else caption_loss(model, processor, head, pred, load_features(args, row), row, config)
                if stage == "joint":
                    loss = loss + config["head_loss_weight"] * aux
                val_losses.append(float(loss))
                cov = coverage(row, pred["indices"].tolist())
                event_hits += cov["event_hits"]; event_count += cov["event_count"]
        val_loss = sum(val_losses) / len(val_losses)
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        best = state["best_validation_loss"] is None or val_loss < state["best_validation_loss"]
        state.update(next_epoch=epoch + 1, next_offset=0)
        state["history"].append({"epoch": epoch, "validation_loss": val_loss, "event_hits": event_hits, "event_count": event_count})
        if best:
            state["best_validation_loss"] = val_loss
        save_checkpoint(stage_dir, head, model, optimizer, state, best=best)
        head.train()
        if model is not None:
            model.train(); model.base_model.model.model.visual.eval()
    write_json(stage_dir / "RESULT.json", {"status": "COMPLETE", "smoke_only": smoke, "quality_improvement_claimed": False,
        "updates": state["updates"], "epochs": epochs, "weights_changed": state["weights_changed"],
        "resumed": state["resumed"], "caption_gradient_to_selection": state["caption_gradient_to_selection"],
        "caption_gradient_to_candidates": state["caption_gradient_to_candidates"],
        "best_validation_loss": state["best_validation_loss"], "session_seconds": time.monotonic() - start,
        "training_sources": sorted({r["source_sha256"] for r in train}),
        "validation_sources": sorted({r["source_sha256"] for r in validation}), "test_used": False,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if model is not None else None, "runtime": runtime})
    write_json(stage_dir / "status.json", {"status": "COMPLETE", "stage": stage, "updates": state["updates"], "smoke_only": smoke})
    del optimizer, params, groups, model, processor, head, pooled, changed_param
    release_gpu()


def evaluate(args, config, smoke=False):
    import torch
    from .qwen35_multitask import TemporalSelector, coverage
    from peft import PeftModel
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    checkpoint = args.checkpoint or checkpoint_path(args.run / "joint")
    state = read_json(checkpoint / "state.json")
    if state["stage"] not in ["caption", "joint"]:
        raise ValueError("Caption adapter checkpoint required")
    head = TemporalSelector(hidden=config["head_hidden"], candidates=config["candidates"], keys=config["keys"]).cuda().eval()
    head.load_state_dict(torch.load(checkpoint / "head.pt", map_location="cuda", weights_only=True))
    # Ordinary source-image inference: no reference labels or soft frame mixtures.
    eng = engine()
    model, processor, runtime = eng.load("nf4")
    for name, p in model.named_parameters():
        if "norm" in name and "visual" not in name and p.dtype in (torch.bfloat16, torch.float16):
            p.data = p.data.float()
    model = PeftModel.from_pretrained(model, checkpoint / "adapter", is_trainable=False).eval()
    rows = rows_for(args.data, "validation" if smoke else args.split)
    if smoke:
        rows = rows[:1]
    out = args.run / ("smoke_inference" if smoke else f"evaluation_{args.split}")
    protocol = {"checkpoint_state_sha256": sha256(checkpoint / "state.json"), "data_freeze_sha256": sha256(args.data / "freeze.json"), "config": config}
    if (out / "protocol.json").exists() and read_json(out / "protocol.json") != protocol:
        raise ValueError("Inference output exists with different conditions")
    write_json(out / "protocol.json", protocol)
    for row in rows:
        path = out / f"{row['id']}.json"
        if path.exists():
            continue
        start = time.monotonic()
        inputs = eng.base.preprocess(processor, images_for(row, range(8)), labels_for(row, range(8)), "Describe the visible frames.", config["max_pixels"]).to("cuda")
        with torch.no_grad():
            visual = model.base_model.model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
            pooled = torch.stack([x.float().mean(0) for x in visual.pooler_output])
            pred = head(pooled, config["temperature"])
            keys = pred["indices"].tolist()
        selection_seconds = time.monotonic() - start
        selected_tokens = tuple(visual.pooler_output[i] for i in keys)
        del inputs, visual, pooled, pred
        caption_inputs = eng.base.preprocess(processor, images_for(row, keys), labels_for(row, keys), config["caption_prompt"], config["max_pixels"]).to("cuda")
        caption_inputs.pop("pixel_values")
        generation_start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**caption_inputs,
                mm_encoder_outputs={"image": BaseModelOutputWithPooling(pooler_output=selected_tokens)},
                max_new_tokens=config["max_caption_tokens"], do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        tokens = output[0, caption_inputs.input_ids.shape[1]:]
        text = processor.decode(tokens, skip_special_tokens=True).strip()
        result = {"caption": text, "generated_tokens": len(tokens), "generation_seconds": time.monotonic() - generation_start,
            "status": "CAPTION_COMPLETE" if text and len(tokens) < config["max_caption_tokens"] else "EMPTY_OR_TRUNCATED",
            "visual_encoding_reused": True}
        write_json(path, {"id": row["id"], "keys": keys, "source_key_indices": [row["frames"][i]["index"] for i in keys],
            "caption": result["caption"], "caption_call": result, "selection_seconds": selection_seconds,
            "total_seconds": time.monotonic() - start, "coverage": coverage(row, keys),
            "runtime": runtime, "smoke_only": smoke, "protocol_sha256": sha256(out / "protocol.json")})
        del selected_tokens, caption_inputs, output, tokens
        print(json.dumps({"inference": row["id"], "keys": keys, "caption_status": result["status"]}), flush=True)
    del model, head, processor
    release_gpu()


def training_command(args, config):
    from .qwen35_backend import MODEL_ID, REVISION
    if (config["model_id"], config["revision"], config["quantization"], config["candidates"], config["keys"]) != (MODEL_ID, REVISION, "nf4", 8, 4):
        raise ValueError("Use the validated pinned NF4 eight-candidate/four-key configuration")
    if not 1 <= config["gradient_accumulation"] <= 64 or not 4096 <= config["max_pixels"] <= 262144:
        raise ValueError("Invalid memory/accumulation configuration")
    if args.command in ["staged", "joint", "smoke"]:
        from .qwen35_training_eval import verify_baseline
        baseline_protocol = verify_baseline(args.data)
        for key in ["model_id", "revision", "quantization", "max_pixels", "candidates", "keys", "caption_prompt", "max_caption_tokens"]:
            if baseline_protocol["config"][key] != config[key]:
                raise ValueError(f"Training/inference conditions differ from frozen baseline: {key}")
    if args.command == "evaluate":
        evaluate(args, config); return
    ensure_cache(args, config)
    if args.command == "cache":
        return
    if args.command == "smoke":
        args.run = args.run / "smoke"
        for stage in ["heads", "caption", "joint"]:
            phase(args, config, stage, smoke=True, stop_after_updates=1)
            args.resume = True
            phase(args, config, stage, smoke=True)
        evaluate(args, config, smoke=True)
        write_json(args.run / "RESULT.json", {"status": "PASS_TRAIN_RESUME_INFERENCE_SMOKE", "full_training_started": False,
            "phases": {stage: read_json(args.run / stage / "RESULT.json") for stage in ["heads", "caption", "joint"]}})
    elif args.command == "staged":
        phase(args, config, "heads")
        phase(args, config, "caption")
    elif args.command == "joint":
        phase(args, config, "joint")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["baseline", "cache", "staged", "joint", "smoke", "evaluate", "verify"])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    args = p.parse_args()
    audit = verify(args.data)
    config = read_json(args.config)
    if args.command == "verify":
        print(json.dumps(audit, indent=2)); return
    args.run.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / ".local/etri_60s_check.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("GPU work is already running; retry when that job finishes")
    if args.command == "baseline":
        baseline(args, config)
    else:
        training_command(args, config)


if __name__ == "__main__":
    main()
