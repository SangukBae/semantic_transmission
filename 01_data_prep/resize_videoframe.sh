#!/bin/bash

# 设置路径变量，替换为你的视频文件所在的目录
path="$1"
# 检查路径是否为空
if [ -z "$path" ]; then
  echo "请提供有效的路径参数"
  echo "用法: $0 /path/to/videos"
  exit 1
fi
# 遍历所有子目录中的 .mp4 文件
for video in "$path"/*.mp4; do
  # 检查文件是否存在
  if [ -f "$video" ]; then
    echo "正在处理: $video"

    # 创建一个临时文件,在原来的文件名后面加上temp
    temp_video="${video%.mp4}_temp.mp4"

    # 使用 FFmpeg 进行视频裁剪和缩放，保存到临时文件
    ffmpeg -i "$video" -vf "crop=iw:ih:((iw-576)/2):((ih-320)/2),scale=576:320" \
    -c:v libx264 -crf 18 -preset medium -y "$temp_video"

    # 检查临时文件是否成功创建
    if [ -f "$temp_video" ]; then
      # 如果处理成功，替换原始文件
      # 打印temp_video长宽信息
        
      mv "$temp_video" "$video"
      ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$video"
      echo "已处理并覆盖: $video"
    else
      echo "处理失败: $video"
    fi
  fi
done

echo "所有视频处理完成！"