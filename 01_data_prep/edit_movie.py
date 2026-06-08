import pandas as pd
import os
import argparse
from moviepy.editor import VideoFileClip
from moviepy.video.fx import resize

def process_videos(root, subname):
    # CSV 文件路径
    csv_path = os.path.join(root, 'videos.csv')

    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 创建新的目录
    output_dir = os.path.join(root, subname)
    os.makedirs(output_dir, exist_ok=True)

    # 处理视频文件
    for index, row in df.iterrows():
        video_path = row['path']
        
        # 创建输出路径
        output_path = os.path.join(output_dir, os.path.basename(video_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # 读取视频文件
            video = VideoFileClip(video_path)
            
            # 裁剪视频长度
            if video.duration > 16:
                video = video.subclip(0, 16)
            
            # 调整帧率
            if video.fps > 24:
                video = video.set_duration(video.duration).set_fps(24)
            

            # 保存处理后的视频
            video.write_videofile(output_path, codec='libx264', fps=24)
            
            print(f'Processed: {video_path}')
        
        except Exception as e:
            print(f'Error processing {video_path}: {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理视频文件")
    parser.add_argument('--root', type=str, default=os.environ.get("DATA_ROOT", ""), help='根目录路径')
    parser.add_argument('--subname', type=str, default='16x24_576x320', help='子目录名称')

    args = parser.parse_args()

    process_videos(args.root, args.subname)