import os
import faulthandler
from semantic_transmission.temporal import trim_prefix, segment_lengths, conditioning_indices
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
from PIL import Image

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.datasets.utils import get_transforms_image
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    # append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
import cv2
import subprocess
import pandas as pd
import argparse
from typing import Optional
import random
import numpy as np

def fully_control_random_seed(seed: Optional[int] = None,
                              deterministic: bool = True,
                              diff_rank_seed: bool = False) -> int:
    """Comprehensively control random seed for reproducibility.
    
    This function controls all known sources of randomness, including Python
    random module, NumPy, and PyTorch (both CPU and GPU). Additionally, it
    ensures deterministic algorithms are used when possible for reproducibility.
    
    Args:
        seed (int, optional): The random seed to use. If None, a random seed is generated.
        deterministic (bool): Whether to set the deterministic option for CUDNN/MUSA.
                              Defaults to True.
        diff_rank_seed (bool): Whether to add rank number to the random seed to ensure
                               different random seeds in different processes/threads. 
                               Defaults to False.
                               
    Returns:
        int: The seed being used, either provided or generated.
    """
    # If seed is not provided, generate a random seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    # Modify the seed if using multi-threading/multi-processing
    if diff_rank_seed:
        rank = 0 # You would need a function that gets the current process/thread rank
        seed += rank

    # Set Python's built-in random module seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch CPU and GPU (if available) random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set for all CUDA devices
    if hasattr(torch, 'musa') and torch.musa.is_available():
        torch.musa.manual_seed_all(seed)  # Set for all MUSA devices if available
    
    # Set PYTHONHASHSEED to control hash-based randomness
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置 CuBLAS 工作区配置以启用确定性行为
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Ensure deterministic behavior if specified
    if deterministic:
        torch.backends.cudnn.deterministic = True  # Force cuDNN to use deterministic algorithms
        torch.backends.cudnn.benchmark = False  # Disable the benchmark feature for cuDNN
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)  # PyTorch >= 1.10.0 has this setting
    
    return seed

def compress_video(video_path, output_path):
    # 有可能不需要先对视频进行压缩，而对提取出的关键帧进行压缩
    # Compress video to 240p using ffmpeg
    subprocess.run(['ffmpeg', '-i', video_path, '-vf', 'scale=426:240', '-c:a', 'copy', output_path])

def extract_frames(video_path, output_dir, interval):
    # 每隔 interval 秒截取一帧，并保存为图片文件
    # Extract frames from video at specified interval
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    key_frames = []
    # Get video frame rate
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(f"log: read video fps: {fps}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (interval * fps) == 0:
            frame_path = os.path.join(output_dir, f'frame_{frame_count}.png')
            cv2.imwrite(frame_path, frame)
            #将frame从ndarray转换为PIL.Image
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            key_frames.append(frame)
        frame_count += 1
    cap.release()
    print (key_frames)
    return key_frames #返回关键帧的PIL.Image格式

def apply_caption(video_path, caption_path):
    # Apply caption to video using PLLAVA
    # subprocess.run(['pllava', '-i', video_path, '-o', caption_path])
    caption=' A car was travelling fast on a winding mountain road, the trees on both sides of the road were rapidly receding backwards.'
    return caption

#直接从key_frames中编码
def encode_from_sender(key_frames, vae, image_size):
    # 默认and要求batch_size=1
    # Encode key frames to latent space using VAE
    refs_x = [] #refsx: [batch, ref_num, C, T, H, W]
    ref = []
    transform_name="resize_crop"
    for img in key_frames:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
        img = transform(img)
        video = img.unsqueeze(0).repeat(1, 1, 1, 1)
        r = video.permute(1, 0, 2, 3)
        r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
        r_x = r_x.squeeze(0)
        ref.append(r_x)
    refs_x.append(ref)
    return refs_x

#自动生成mask策略
def get_mask_from_sender(key_frames):
    # Generate mask for key frames
    # 默认batch_size=1
    mask_strategy = []
    for i in range(len(key_frames)):
        if i == 0:
            mask ="0;"
        else:
            mask += f"{i-1},{i},0,-1,1"
            if i != len(key_frames)-1:
                mask += ";"
    mask_strategy.append(mask)
    return mask_strategy, i

def get_keyframes(folder):
    """获取指定文件夹中的所有关键帧文件名"""
    key_frame_path = folder
    if os.path.exists(key_frame_path):
        all_frames = os.listdir(key_frame_path)
        #all_frames应该以.png结尾
        all_frames = [frame for frame in all_frames if frame.endswith('.png')]
        #将all_frames按照数字从小到大排序
        all_frames = sorted(all_frames, key=lambda x: int(x.split('.')[0]))
        all_frames = [key_frame_path + '/' + frame for frame in all_frames]
        print(f"all_frames: {all_frames}")
        return all_frames
    else:
        return []
    
def append_multi_score_to_prompts(prompts, aes=None, flow=None, camera_motion=None):
    new_prompts = []
    for i, prompt in enumerate(prompts):
        new_prompt = prompt
        if aes is not None and "aesthetic score:" not in prompt:
            new_prompt = f"{new_prompt} aesthetic score: {aes:.1f}."
        if flow is not None and "motion score:" not in prompt:
            new_prompt = f"{new_prompt} motion score: {flow[i]:.1f}."
        if camera_motion is not None and "camera motion:" not in prompt:
            new_prompt = f"{new_prompt} camera motion: {camera_motion}."
        new_prompts.append(new_prompt)
    return new_prompts

if __name__ == "__main__":
    # Slow first-time model loading/kernels remain diagnosable in batch logs.
    faulthandler.dump_traceback_later(180, repeat=True)
#发送端视频信息批量提取
    cfg = parse_configs(training=False)

    root_path = cfg.root_path
    csv_path = cfg.csv_path
    save_dir = cfg.save_dir
    method = cfg.method #应当为形如“key_framesinternvl_diff_0.35”的字符串
    print("log: root_path ",root_path, "csv_path ",csv_path, "save_dir ",save_dir)
    # csv文件中列分别为：path,text,flow
    df = pd.read_csv(csv_path)
    videos = []
    texts = []
    flows = []
    cur_video_name = ""
    cur_video = []
    cur_text = []
    cur_flow = []
    for index, row in df.iterrows():
        video_path = row['path']
        text = row['text']
        flow = row['flow']
        #提取路径中的视频名字
        video_name = video_path.split('/')[-2] #取名称
        # 因为是从生成的csv文件中读取，所以顺序i一定是相近的
        if video_name != cur_video_name:
            if cur_video_name != "":
                videos.append(cur_video)
                texts.append(cur_text)
                flows.append(cur_flow)
            cur_video_name = video_name
            cur_video = []
            cur_text = []
            cur_flow = []
        cur_video.append(video_path)
        cur_text.append(text)
        cur_flow.append(flow)
    videos.append(cur_video)
    texts.append(cur_text)
    flows.append(cur_flow)
    
    print("log: videos ",videos)
    print("log: texts ",texts)
    print("log: flows ",flows)
    # compress_video(video_path, save_dir)

#以下是接收端推理准备部分
    torch.set_grad_enabled(False)#关闭梯度计算
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==

    # 设置随机数
    set_random_seed(seed=cfg.get("seed", 1024))
    seed = fully_control_random_seed(seed=cfg.get("seed", 1024), deterministic=cfg.get("deterministic", True), diff_rank_seed=False)
    

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True #允许使用tf32
    torch.backends.cudnn.allow_tf32 = True #允许使用tf32

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    # set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=cfg.get("text_encoder_device", device))
    if cfg.get("text_encoder_device", device) != device:
        # Keep the 4.7B T5 on CPU; transfer only its small conditioning tensors.
        original_encode = text_encoder.encode
        def encode_to_sampling_device(text):
            return {name: tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
                    for name, tensor in original_encode(text).items()}
        text_encoder.encode = encode_to_sampling_device
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    
    


#以下是接收端推理部分，循环处理每个视频
    for i in range(len(videos)):
#接受关键帧
        frams_dir = root_path + "/frames/"+videos[i][0].split('/')[-2] + "/"+method
        print("log: frams_dir ",frams_dir)
        key_frames = get_keyframes(frams_dir)
        if len(key_frames) < 2 or len(texts[i]) != len(key_frames) - 1:
            raise ValueError("Each video requires >=2 keyframes and one caption per adjacent pair")
        print("log: key_frames ",key_frames)
        num_frames_ls = segment_lengths(
            [int(os.path.basename(frame).split('.')[0]) for frame in key_frames],
            dframe_to_frame(cfg.get("condition_frame_length", 5)),
        )

        print("log: num_frames_ls ",num_frames_ls)
        
#将关键帧读取为PIL.Image格式
        key_frames = [Image.open(frame) for frame in key_frames]
        
        
#处理每个looop的caption
        caption = ""
        for k, text in enumerate(texts[i]):
            caption += f'|{k}|' + text
            

        flow_ls = flows[i]
        print("log: flow_ls ",flow_ls)
        print("log: caption ",caption)

# ======================================================
# inference
# ======================================================
        # == load prompts ==
        prompts = [caption]
        print("----------prompts:", prompts)
        start_idx = cfg.get("start_index", 0)
        if prompts is None:
            if cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
            else:
                prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop
        print("----------prompts:", prompts)

        # == prepare reference ==
        reference_path = cfg.get("reference_path", [""] * len(prompts))
        mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        mask_strategy, loop = get_mask_from_sender(key_frames)

        print(f"----------{i}th video inference start")
        print("----------mask_strategy:", mask_strategy)
        print("----------has loops:", loop)

        # == prepare arguments ==
        fps = cfg.fps
        save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
        multi_resolution = cfg.get("multi_resolution", None)
        batch_size = cfg.get("batch_size", 1)
        num_sample = cfg.get("num_sample", 1)
        # loop = cfg.get("loop", 1)
        condition_frame_length = cfg.get("condition_frame_length", 5)
        condition_frame_edit = cfg.get("condition_frame_edit", 0)
        align = cfg.get("align", None)

        os.makedirs(save_dir, exist_ok=True)
        # sample_name = cfg.get("sample_name", None)
        sample_name = videos[i][0].split('/')[-2]
        prompt_as_path = cfg.get("prompt_as_path", False)

        # == Iter over all samples ==
        for j in progress_wrap(range(0, len(prompts), batch_size)):
            # == prepare batch prompts ==
            batch_prompts = prompts[j : j + batch_size]
            ms = mask_strategy[j : j + batch_size]
            refs = reference_path[j : j + batch_size]
            print("-------------batch_prompts:", batch_prompts)

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            print("-------------batch_prompts:", batch_prompts)
            print("-------------reference paths:", refs)
            print("-------------ms:", ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==

            refs = encode_from_sender(key_frames, vae, image_size) # 按顺序编码关键帧
            print("-------------reference latent shapes:", [[tuple(x.shape) for x in row] for row in refs])


            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):
                # == prepare save paths ==
                save_paths = [
                    get_save_path_name(
                        save_dir,
                        sample_name=sample_name,
                        sample_idx=start_idx + idx,
                        prompt=original_batch_prompts[idx],
                        prompt_as_path=prompt_as_path,
                        num_sample=num_sample,
                        k=k,
                    )
                    for idx in range(len(batch_prompts))
                ]

                # NOTE: Skip if the sample already exists
                # This is useful for resuming sampling VBench
                if prompt_as_path and all_exists(save_paths):
                    continue

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)
                print("---------batched_prompt_segment_list", batched_prompt_segment_list)
                print("---------batched_loop_idx_list", batched_loop_idx_list)

                # 1. refine prompt by openai
                if cfg.get("llm_refine", False):
                    # only call openai API when
                    # 1. seq parallel is not enabled
                    # 2. seq parallel is enabled and the process is rank 0
                    if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                            batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                    # sync the prompt if using seq parallel
                    if enable_sequence_parallelism:
                        coordinator.block_all()
                        prompt_segment_length = [
                            len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                        ]

                        # flatten the prompt segment list
                        batched_prompt_segment_list = [
                            prompt_segment
                            for prompt_segment_list in batched_prompt_segment_list
                            for prompt_segment in prompt_segment_list
                        ]

                        # create a list of size equal to world size
                        broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                        dist.broadcast_object_list(broadcast_obj_list, 0)

                        # recover the prompt list
                        batched_prompt_segment_list = []
                        segment_start_idx = 0
                        all_prompts = broadcast_obj_list[0]
                        for num_segment in prompt_segment_length:
                            batched_prompt_segment_list.append(
                                all_prompts[segment_start_idx : segment_start_idx + num_segment]
                            )
                            segment_start_idx += num_segment

                # 2. append score
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    print("---------prompt_segment_list", prompt_segment_list)
                    batched_prompt_segment_list[idx] = append_multi_score_to_prompts(
                        prompt_segment_list,
                        aes=cfg.get("aes", None),
                        # flow=cfg.get("flow", None),
                        flow = flow_ls,
                        camera_motion=cfg.get("camera_motion", None),
                    )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))
                print("---------batch_prompts", batch_prompts)

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    #这部分应该放在循环中，更改为输入视频的长度 =======================
                    num_frames = num_frames_ls[loop_i]
                    # == multi-resolution info ==
                    model_args = prepare_multi_resolution_info(
                        multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
                    )
                    # == build diffusion model ==
                    input_size = (num_frames, *image_size)
                    latent_size = vae.get_latent_size(input_size)
                    model = (
                        build_module(
                            cfg.model,
                            MODELS,
                            input_size=latent_size,
                            in_channels=vae.out_channels,
                            caption_channels=text_encoder.output_dim,
                            model_max_length=text_encoder.model_max_length,
                            enable_sequence_parallelism=enable_sequence_parallelism,
                        )
                        .to(device, dtype)
                        .eval()
                    )
                    text_encoder.y_embedder = model.y_embedder  # required for classifier-free guidance

                    # == build scheduler ==
                    scheduler = build_module(cfg.scheduler, SCHEDULERS)
                    # ======================================================

                    # == get prompt for loop i ==
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        # Encode exactly one overlap block, including for dense SKEM cuts.
                        previous = video_clips[-1]
                        if cfg.get("decoder_policy") != "official_release":
                            overlap = conditioning_indices(previous.shape[2], dframe_to_frame(condition_frame_length))
                            previous = previous[:, :, overlap]
                        refs, ms = append_generated(
                            vae, previous, refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )
                    print("loop_i", loop_i, "batch_prompts_loop", batch_prompts_loop, "refs", len(refs), "ms", ms)
                    # == sampling ==
                    z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    print(masks)
                    
                    samples = scheduler.sample(
                        model,
                        text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                    )
                    
                    samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                    video_clips.append(samples)

                # == save samples ==
                if is_main_process():
                    for idx, batch_prompt in enumerate(batch_prompts):
                        if verbose >= 2:
                            logger.info("Prompt: %s", batch_prompt)
                        save_path = save_paths[idx]
                        video = [video_clips[i][idx] for i in range(loop)]
                        for i in range(1, loop):
                            video[i] = video[i][:, trim_prefix(i, dframe_to_frame(condition_frame_length),
                                                              cfg.get("decoder_policy", "endpoint_exact")) :]
                        video = torch.cat(video, dim=1)
                        if cfg.get("save_frames", False):
                            # Preserve pre-MP4 uint8 pixels for a common evaluation boundary.
                            frames_dir = save_path + "_frames"
                            os.makedirs(frames_dir, exist_ok=False)
                            frames_uint8 = (video.clamp(-1, 1).add(1).div(2).mul(255).add(0.5)
                                            .clamp(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8))
                            for frame_number, frame in enumerate(frames_uint8):
                                Image.fromarray(frame.numpy()).save(os.path.join(frames_dir, f"{frame_number:05d}.png"))
                        save_path = save_sample(
                            video,
                            fps=save_fps,
                            save_path=save_path,
                            verbose=verbose >= 2,
                        )
                        if save_path.endswith(".mp4") and cfg.get("watermark", False):
                            time.sleep(1)  # prevent loading previous generated video
                            add_watermark(save_path)
            start_idx += len(batch_prompts)
        logger.info("Inference finished.")
        logger.info("Saved %s samples to %s", start_idx, save_dir)
