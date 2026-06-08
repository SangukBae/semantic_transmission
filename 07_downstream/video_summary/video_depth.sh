#!/bin/bash

# 这个脚本接受一些输入,完成从从视频到视频路径的标注

# 用法: ./video_depth.sh "$DATA_ROOT" 16x24 frames key_framesinternvl_diff_0.35 "$DEPTH_DIR/Sora"
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





