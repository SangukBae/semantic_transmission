#!/bin/bash

#./get_prepare.sh $DATA_ROOT 16x24
#./get_prepare.sh $DATA_ROOT 16x24
#./get_prepare.sh $DATA_ROOT 16x24
#./get_prepare.sh $DATA_ROOT 16x24
#./get_prepare.sh $DATA_ROOT 16x24
root_dir="$1"
sub_name="$2"

# Check if args is provided
if [ -z "$root_dir" ]; then
    echo "Error: Root directory not provided."
    exit 1
fi
if [ -z "$sub_name" ]; then
    echo "Error: Subname not provided."
    exit 1
fi

echo "start get_prepare.sh"

# 检查conda环境是不是llava
if [ "$(conda env list | grep \* | cut -d ' ' -f 1)" != "llava" ]; then
    echo "Error: Conda environment is not llava."
    echo "Activating conda environment: llava"
    source "$(conda info --base)/etc/profile.d/conda.sh"  # 确保能够使用 conda 命令
    conda activate llava
fi

# 先获取视频
# 在root_dir生成一个videos.csv
echo "Getting videos..."
python get_csv.py --root "$root_dir" --subname ""


# 编辑视频，统一到24fps，不超过16s
# 将视频保存到root_dir/subname中
echo "Editing videos..."
python edit_movie.py --root "$root_dir" --subname "$sub_name"

# 将视频resize到576x320
# 将视频保存到root_dir/subname中
echo "Resizing videos..."
bash resize_videoframe.sh "$root_dir/$sub_name"

# Call 重新获取csv
# 将csv中指向新的视频的路径
echo "Getting csv..."
python get_csv.py --root "$root_dir" --subname "$sub_name"

# Call get_frame.py
# 从处理后的视频中提取关键帧
echo "Getting frames..."
python get_frame.py --root "$root_dir" --subname "$sub_name"

# if [ "$(conda env list | grep \* | cut -d ' ' -f 1)" != "internvl" ]; then
#     echo "Error: Conda environment is not internvl."
#     echo "Activating conda environment: internvl"
#     source "$(conda info --base)/etc/profile.d/conda.sh"  # 确保能够使用 conda 命令
#     conda activate internvl
# fi

# python $INTERNVL_DIR/MLM-keyframe-internvl_0_4.py \
#     --csv-path $DATA_ROOT/16x24/videos.csv \
#     --method key_framesinternvl_diff_0.35
