#!/bin/bash

# 这个脚本接受一些输入,完成从关键帧提取被关键帧切分的视频片段,并进行视频片段的描述和flow特征的提取

# 用法: ./after_extract.sh "$DATA_ROOT" 16x24 frames key_framesinternvl_diff_0.35
#   ($DATA_ROOT and $OPENSORA_DIR come from env.sh at the repo root; source it first)

source ~/anaconda3/etc/profile.d/conda.sh

root_dir="$1"
video_dir="$2"
frame_dir="$3"
method="$4"

# Check if args is provided
if [ -z "$root_dir" ]; then
    echo "Error: Root directory not provided."
    exit 1
fi
if [ -z "$video_dir" ]; then
    echo "Error: Video directory not provided."
    exit 1
fi
if [ -z "$frame_dir" ]; then
    echo "Error: Frame directory not provided."
    exit 1
fi
if [ -z "$method" ]; then
    echo "Error: Method not provided."
    exit 1
fi

# 将conda环境切换到llava
conda activate llava
# 检查conda环境是不是llava
if [ "$(conda env list | grep \* | cut -d ' ' -f 1)" != "llava" ]; then
    echo "Error: Conda environment is not llava."
    exit 1
fi
# 1. 从关键帧提取被关键帧切分的视频片段
echo "Extracting video clips..."
python get_key_video.py --root_directory "$root_dir" --videos "$video_dir" --frames "$frame_dir" --method "$method" > /dev/null


# 2. 进行视频片段的描述
echo "Describing video clips..."

csv_path="$root_dir/$video_dir/$method""_video_paths.csv"
echo "CSV path: $csv_path"

# 将conda环境切换到pllava
conda activate pllava

# 检查conda环境是不是pllava
if [ "$(conda env list | grep \* | cut -d ' ' -f 1)" != "pllava" ]; then
    echo "Error: Conda environment is not pllava."
    exit 1
fi

: "${OPENSORA_DIR:?set OPENSORA_DIR (source env.sh at repo root)}"
cd "$OPENSORA_DIR/tools/caption/pllava_dir"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PYTHONPATH:$OPENSORA_DIR/tools/caption/pllava_dir/PLLaVA" \
python caption_pllava.py \
  --pretrained_model_name_or_path PLLaVA/MODELS/pllava-7b \
  --use_lora \
  --lora_alpha 4 \
  --num_frames 4 \
  --weight_dir PLLaVA/MODELS/pllava-7b \
  --csv_path "$csv_path" \
  --pooling_shape 4-12-12




# 3. 进行flow特征的提取
cd "$OPENSORA_DIR"

#判断是否存在文件夹pretained_models
if [ ! -d "pretrained_models" ]; then
    echo "Error: Directory pretained_models not found."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth -P ./pretrained_models/unimatch/
fi

csv_path="$root_dir/$video_dir/$method""_video_paths_text.csv"

CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node 1 tools/scoring/optical_flow/inference.py  $csv_path

