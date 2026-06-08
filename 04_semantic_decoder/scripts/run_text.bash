#!/bin/bash

# 检查参数
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <csv_name> <save_dir> <method> <root_dir>"
    echo "using default values:"
fi

# 检查conda环境是否激活
if [[ "$CONDA_DEFAULT_ENV" != "opensora" ]]; then
    echo "Error: conda environment 'opensora' is not activated."
    exit 1
fi

# 设置默认参数
default_csv_name="videos_text_flow"
default_save_name="text_only"
default_root_dir="${DATA_ROOT:?set DATA_ROOT (source env.sh)}"
default_method="text_only"

# 获取参数，使用提供的值或默认值
root_dir=${4:-$default_root_dir}
csv_name=${1:-$default_csv_name}
csv_path="$root_dir/16x24/$csv_name.csv"
save_name=${2:-$default_save_name}
save_dir="$root_dir/$save_name"
method=${3:-$default_method}

# 判断method与csv_name去掉_video_paths_text_flow是否一致
# if [[ "$method" != "${csv_name%_video_paths_text_flow}" ]]; then
#     echo "Error: method name '$method' does not match CSV name '$csv_name'."
#     exit 1
# fi

# 输出用于调试的参数值
echo "CSV path: $csv_path"
echo "save directory: $save_dir"
echo "root directory: $root_dir"
echo "method: $method"

# 创建输出文件夹
mkdir -p "$save_dir/temp_csv"

# 初始化变量
cur_video_name=""
file_index=1
temp_file=""

# 遍历CSV文件的每一行（跳过第一行表头）
tail -n +2 "$csv_path" | while IFS= read -r line; do
    # 使用Python脚本正确解析CSV行（包括引号内的逗号）
    parsed_line=$(python3 - <<EOF
import csv
import sys

line = '''$line'''
reader = csv.reader([line], skipinitialspace=True)
for row in reader:
    print('|'.join(row))
EOF
    )

    # 分割解析的行数据
    video_path=$(echo "$parsed_line" | cut -d '|' -f 1)
    text=$(echo "$parsed_line" | cut -d '|' -f 2)
    flow=$(echo "$parsed_line" | cut -d '|' -f 3)

    # 提取视频名称
    video_name=$(basename "$video_path" .mp4)

    # 检查视频名称是否发生变化
    if [[ "$video_name" != "$cur_video_name" ]]; then
        # 如果已经有当前视频的文件，则关闭并保存
        if [[ -n "$temp_file" ]]; then
            file_index=$((file_index + 1))
        fi
        # 新的视频，创建新的文件
        temp_file="$save_dir/temp_csv/$file_index.csv"
        echo "path,text,flow" > "$temp_file"
        cur_video_name="$video_name"
    fi

    # 将当前行写入对应的CSV文件
    echo "$video_path,\"$text\",$flow" >> "$temp_file"

    # 输出日志
    echo "log: video_name: $video_name"
    echo "log: text: $text"
    echo "log: flow: $flow"

    # 调用Python脚本进行推理
    python scripts/inference.py configs/opensora-v1-2/inference/sample.py \
        --num-frames 4s --resolution 576 --aspect-ratio 5:9 --aes 6.5 \
        --prompt "$text" --save-dir "$save_dir" --sample-name "$video_name" --flow "$flow" >> "$save_dir/log.txt" 2>&1
done

# 判断以上是否成功
if [ $? -ne 0 ]; then
    echo "Error: failed to process."
    exit 1
fi

# 清理临时文件
rm -rf "$save_dir/temp_csv"

