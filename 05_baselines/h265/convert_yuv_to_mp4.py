import os
import glob

# 设置输入和输出目录
input_dir = "HEVC/ClassD"
output_dir = "HEVC/ClassD/mp4"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有 .yuv 文件
yuv_files = glob.glob(os.path.join(input_dir, "*.yuv"))

# 遍历每个 .yuv 文件并转换为 .mp4
for yuv_file in yuv_files:
    # 获取文件名（不包括扩展名）
    file_name = os.path.splitext(os.path.basename(yuv_file))[0]
    
    # 假设文件名中包含分辨率信息，例如 "video_1920x1080_30.yuv"
    resolution = file_name.split("_")[1]
    width, height = resolution.split("x")
    
    # 设置输出文件路径
    output_file = os.path.join(output_dir, file_name + ".mp4")
    
    # 使用 ffmpeg 进行转换
    command = f"ffmpeg -y -s {width}x{height} -pix_fmt yuv420p -i {yuv_file} -c:v libx264 -crf 0 -preset veryslow {output_file}"
    os.system(command)
    print(f"Converted {yuv_file} to {output_file}")