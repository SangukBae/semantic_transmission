"""Pinned, inference-only caption repair adapter for new FC-LGVSC runs.

The caption-only arm is intentionally independent of candidate generation and
visual review. Historical configs without this field continue to use base Qwen.
"""
import json
from pathlib import Path

from .artifacts import sha256

REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "outputs/qwen35_evidence_repair_v3/caption/update-000060-e002-n0000/adapter"
FILES = {
    "adapter_config.json": "2f27e042e426fc291a644faf284ac2753ba192d0c4538d2964721cfaeaec3f75",
    "adapter_model.safetensors": "2115bb4ab511b73e1b2781f2cb06a6288492ac8291e7bc84ee1157c29911af9d",
}
NAME = "fc_caption_repair"


def default_adapter():
    value = dict(id="qwen35_evidence_repair_v3_caption_e002_u000060", path=str(ADAPTER),
        files=dict(FILES), rank=32, alpha=64, active_tasks=["draft"],
        language_norm_dtype="float32", output_format="json", inference_only=True)
    verify_adapter(value)
    return value


def verify_adapter(value):
    """Fail closed on missing/replaced weights; never silently use the base model."""
    expected = dict(id="qwen35_evidence_repair_v3_caption_e002_u000060", path=str(ADAPTER),
        files=dict(FILES), rank=32, alpha=64, active_tasks=["draft"],
        language_norm_dtype="float32", output_format="json", inference_only=True)
    if value != expected:
        raise ValueError("caption adapter policy differs from pinned default")
    for name, digest in FILES.items():
        if sha256(ADAPTER / name) != digest:
            raise ValueError(f"caption adapter bytes changed: {name}")
    cfg = json.loads((ADAPTER / "adapter_config.json").read_text())
    if (cfg["r"], cfg["lora_alpha"], cfg["peft_type"], cfg["bias"], cfg["modules_to_save"]) != (32, 64, "LORA", "none", None):
        raise ValueError("unsupported caption adapter architecture")


def task_runtime(config, task):
    if task not in {"draft", "proposals", "visual_review"}:
        raise ValueError("caption adapter requires an explicit inference task")
    active = task == "draft"
    return dict(inference_task=task,
        caption_adapter=config["caption_adapter"] if active else None,
        language_norm_dtype="float32" if active else "bfloat16")


def verify_record(config, record):
    if config.get("caption_adapter"):
        expected = task_runtime(config, record["identity"]["task"])
        runtime = record.get("model_runtime", {})
        if any(k not in runtime or runtime[k] != v for k, v in expected.items()):
            raise ValueError("caption adapter/task provenance mismatch")
