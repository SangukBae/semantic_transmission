import argparse
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
import subprocess

def resize_video(input_file, output_file):
    """使用ffmpeg调整视频大小，保持比例，短边设为256"""
    try:
        # 构建ffmpeg命令，使用更简洁的缩放表达式
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_file,
            '-vf', 'scale=iw*min(256/iw\\,256/ih):ih*min(256/iw\\,256/ih)',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',
            '-y',  # 覆盖输出文件
            output_file
        ]
        
        # 执行ffmpeg命令
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    
    except Exception as e:
        print(f"处理视频时出错: {input_file}")
        print(f"错误信息: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='调整视频大小并提取标签')
    parser.add_argument('--input_path', type=str, required=True, help='输入路径')
    parser.add_argument('--method', type=str, required=True, help='方法名称')
    parser.add_argument('--csv_path', type=str, required=True, help='测试CSV文件路径')
    
    args = parser.parse_args()
    
    input_dir = os.path.join(args.input_path, args.method)
    output_dir = os.path.join(args.input_path, f"{args.method}_256")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件获取标签
    labels_df = pd.read_csv(args.csv_path, header=None)
    
    # 创建新的CSV用于存储结果
    result_rows = []
    
    # 遍历视频文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.mp4'):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)
            
            # 调整视频大小
            if resize_video(input_file_path, output_file_path):
                # 使用完整文件名作为标识符
                identifier = file_name  # 使用完整文件名
                                
                # 在原始CSV中查找精确匹配项
                matching_rows = labels_df[labels_df.iloc[:, 0] == identifier]
                if not matching_rows.empty:
                    label = matching_rows.iloc[0, 1]  # 获取标签（第二列）
                    
                    # 添加到结果中
                    result_path = os.path.join(output_dir, file_name)
                    result_rows.append([result_path, label])
                else:
                    print(f"警告: 在CSV中未找到匹配项: {file_name}")
    
    # 将结果保存到新的CSV文件
    result_df = pd.DataFrame(result_rows)
    result_csv_path = os.path.join(output_dir, 'test.csv')
    result_df.to_csv(result_csv_path, index=False, header=False)
    
    print(f"处理完成。调整大小的视频已保存到: {output_dir}")
    print(f"标签信息已保存到: {result_csv_path}")

if __name__ == '__main__':
    main()