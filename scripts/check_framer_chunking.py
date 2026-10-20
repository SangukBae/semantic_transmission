"""Verify chunked/nonchunked temporal blocks, including their input gradients."""
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(".local/framer_vendor").resolve()))
from semantic_transmission.framer_compat import patch_temporal_feed_forward
patch_temporal_feed_forward()
# Test the class actually imported by Framer, not its unused local copy.
from models_diffusers.transformer_temporal import TemporalBasicTransformerBlock

torch.manual_seed(7)
torch.set_num_threads(2)
block = TemporalBasicTransformerBlock(32, 32, 4, 8, cross_attention_dim=24).eval()
x = torch.randn(14, 9, 32, requires_grad=True)
context = torch.randn(18, 2, 24)
y = block(x, num_frames=7, encoder_hidden_states=context)
gradient = torch.autograd.grad(y.square().mean(), x)[0]
block.set_chunk_feed_forward(1)
chunked = block(x, num_frames=7, encoder_hidden_states=context)
chunk_gradient = torch.autograd.grad(chunked.square().mean(), x)[0]
torch.testing.assert_close(chunked, y, atol=1e-6, rtol=1e-5)
torch.testing.assert_close(chunk_gradient, gradient, atol=1e-7, rtol=1e-5)
print(json.dumps(dict(status="PASSED", max_output_difference=float((chunked-y).abs().max()),
                      max_gradient_difference=float((chunk_gradient-gradient).abs().max()))))
