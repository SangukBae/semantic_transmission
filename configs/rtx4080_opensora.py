# Local compatibility profile; this is not the paper's full-resolution experiment.
resolution = "240p"
aspect_ratio = "9:16"
image_size = (256, 256)
num_frames = 17
fps = 24
frame_interval = 1
save_fps = 24
seed = 1024
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
# Aligning a 5-latent clip to blocks of five moves its last keyframe to position 0.
align = None
deterministic = False
text_encoder_device = "cpu"
model = dict(type="STDiT3-XL/2", from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
             qk_norm=True, force_huggingface=True, enable_flash_attn=False, enable_layernorm_kernel=False)
vae = dict(type="OpenSoraVAE_V1_2", from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
           micro_frame_size=17, micro_batch_size=1, force_huggingface=True)
text_encoder = dict(type="t5", from_pretrained="DeepFloyd/t5-v1_1-xxl",
                    model_max_length=300, dtype=__import__("torch").float32)
scheduler = dict(type="rflow", use_timestep_transform=True, num_sampling_steps=10, cfg_scale=7.0)
aes = 6.5
flow = None
