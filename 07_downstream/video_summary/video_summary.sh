#!/bin/bash

# 这个脚本接受一些输入,完成从从视频片段提取视频的描述

# 用法: ./video_summary.sh "$DATA_ROOT" 16x24 frames key_framesinternvl_diff_0.35 <target_dir>
#   (source env.sh at repo root first; needs $OPENSORA_DIR)
source "$(conda info --base)/etc/profile.d/conda.sh"

root_dir="$1"
video_dir="$2"
frame_dir="$3"
method="$4"
target_dir="$5"

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
if [ -z "$target_dir" ]; then
    echo "Error: Target directory not provided."
    exit 1
fi

conda activate llava


if [ "$method" = "origin" ] || [ "$method" = "compare" ] || [ "$method" = "compare1" ]; then
    video_path="$root_dir/$video_dir"
    # 调用get_summary_path.py脚本，将路径保存到csv_path中，其中包含了视频的路径，表头为path。输入为包含视频的路径，target_dir，method
    python get_summary_path.py --video_path "$video_path" --target_dir "$target_dir" --method "$method"
else
    video_path="$root_dir/${method}_10_save_dir_individual"
    python get_summary_path.py --video_path "$video_path" --target_dir "$target_dir" --method "$method"
fi

csv_path="$target_dir/${method}_path.csv"

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
python caption_pllava_downstream.py \
  --pretrained_model_name_or_path PLLaVA/MODELS/pllava-7b \
  --use_lora \
  --lora_alpha 4 \
  --num_frames 4 \
  --weight_dir PLLaVA/MODELS/pllava-7b \
  --csv_path "$csv_path" \
  --pooling_shape 4-12-12




