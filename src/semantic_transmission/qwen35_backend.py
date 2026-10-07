"""Pinned local Qwen3.5-9B for FC-LGVSC event proposals and captions."""
from pathlib import Path
from contextlib import contextmanager, nullcontext
import sys
import time

REPO = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SNAPSHOT = REPO / ".local/qwen35_hf" / "models--Qwen--Qwen3.5-9B" / "snapshots" / REVISION


class LocalQwen35:
    def __init__(self, config):
        if (config["model_id"], config["model_revision"], config["quantization"]) != (MODEL_ID, REVISION, "nf4"):
            raise ValueError("expected pinned Qwen3.5-9B NF4, not assistant-authored or Qwen3-VL data")
        if Path(config["model_path"]).resolve() != SNAPSHOT.resolve():
            raise ValueError("Qwen model path differs from pinned snapshot")
        self.config = config
        if config.get("caption_adapter"):
            from .qwen35_caption_adapter import verify_adapter
            verify_adapter(config["caption_adapter"])
        # Reuse the already validated offline loader and its GPU memory guard.
        sys.path.insert(0, str(REPO / "scripts"))
        import qwen35_advanced_caption as engine
        self.engine = engine
        self.model, self.processor, self.runtime = engine.load("nf4")
        self.adapter_loaded = False

    @contextmanager
    def inference_role(self, task):
        """One backbone: LoRA/FP32 language norms only while drafting captions."""
        if not self.config.get("caption_adapter"):
            yield {}
            return
        import torch
        from .qwen35_caption_adapter import NAME, task_runtime, verify_adapter
        runtime = task_runtime(self.config, task)
        if task == "draft" and not self.adapter_loaded:
            verify_adapter(self.config["caption_adapter"])
            # Same locally installed PEFT as training; no downloads/installations.
            sys.path.insert(0, str(REPO / ".local/qwen35_training_probe_deps"))
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.config["caption_adapter"]["path"],
                adapter_name=NAME, is_trainable=False, local_files_only=True)
            self.model.requires_grad_(False)
            self.model.eval()
            self.adapter_loaded = True
        norm_dtypes = []
        try:
            if task == "draft":
                self.model.set_adapter(NAME)
                # set_adapter can mark parameters trainable; inference never does.
                self.model.requires_grad_(False)
                for name, p in self.model.named_parameters():
                    if "norm" in name and "visual" not in name and p.dtype in (torch.float16, torch.bfloat16):
                        norm_dtypes.append((p, p.dtype))
                        p.data = p.data.float()
            disabled = self.model.disable_adapter() if self.adapter_loaded and task != "draft" else nullcontext()
            with disabled:
                yield runtime
        finally:
            for p, dtype in norm_dtypes:
                p.data = p.data.to(dtype)
            if self.adapter_loaded:
                # PEFT re-enables adapter gradients on leaving disable_adapter().
                self.model.requires_grad_(False)

    def generate(self, images, labels, prompt, max_tokens, *, task=None):
        import torch
        if not 1 <= len(images) <= 16 or len(images) != len(labels):
            raise ValueError("expected 1..16 labeled chronological images")
        if not 1 <= max_tokens <= 1152:
            raise ValueError("structured output budget exceeds 1152 tokens")
        inputs = self.engine.base.preprocess(self.processor, images, labels, prompt,
                                             self.config["max_pixels"])
        if inputs.input_ids.shape[1] > 8192:
            raise ValueError("input exceeds the validated 8192-token budget")
        inputs = inputs.to("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with self.inference_role(task) as role_runtime, torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                         do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        tokens = output[0, inputs.input_ids.shape[1]:]
        return dict(raw_text=self.processor.decode(tokens, skip_special_tokens=True).strip(),
            generated_tokens=len(tokens), input_tokens=inputs.input_ids.shape[1],
            image_grid_thw=inputs.image_grid_thw.tolist(), image_count=len(images),
            max_pixels=self.config["max_pixels"], max_new_tokens=max_tokens,
            seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            truncated=len(tokens) >= max_tokens, model_runtime=dict(self.runtime, **role_runtime))
