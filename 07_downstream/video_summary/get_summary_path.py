#从路径中得到所有视频的路径，将路径保存到csv_path中，其中包含了视频的路径，表头为path。输入为包含视频的路径，target_dir，method
import os
import pandas as pd
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()

    target_dir = args.target_dir
    video_path = args.video_path
    method = args.method

    csv_path = os.path.join(target_dir, f'{method}_path.csv')

    # 遍历video_path目录下所有文件（不包括子文件夹），将.mp4文件的路径保存到csv_path中
    video_list = []
    for filename in os.listdir(video_path):
        if filename.endswith('.mp4'):
            video_list.append(os.path.join(video_path, filename))

    # 将视频路径保存到DataFrame
    df = pd.DataFrame({'path': video_list})
    df.to_csv(csv_path, index=False)
