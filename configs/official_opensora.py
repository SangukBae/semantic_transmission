# Published Open-Sora sample.py + LGVSC run.bash settings, with explicit size
# because the pinned Open-Sora aspect table does not contain resolution "576".
image_size = (320, 576)
num_frames = "4s"
fps = 24
save_fps = 24
frame_interval = 1
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5
deterministic = True
decoder_policy = "official_release"
# T5Encoder's upstream default is FP32. CPU placement bounds GPU memory.
text_encoder_device = "cpu"
model = dict(type="STDiT3-XL/2", from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
             qk_norm=True, force_huggingface=True,
             enable_flash_attn=True, enable_layernorm_kernel=True)
vae = dict(type="OpenSoraVAE_V1_2", from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
           micro_frame_size=17, micro_batch_size=4, force_huggingface=True)
text_encoder = dict(type="t5", from_pretrained="DeepFloyd/t5-v1_1-xxl",
                    model_max_length=300)
scheduler = dict(type="rflow", use_timestep_transform=True,
                 num_sampling_steps=30, cfg_scale=7.0)
aes = 6.5
flow = None
