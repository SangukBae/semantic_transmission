import os
import cv2
import pandas as pd
import csv
import sys

def extract_frames(video_path, output_dir, interval):
    # Extract frames from video at specified interval, 同时在output_dir下保存一个新的frames.csv文件，记录截取的所有帧的位置
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    with open(os.path.join(output_dir, 'frames.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_path'])
        # 获取视频帧率
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        print(f"视频帧率: {fps}")
        #截取间隔为int(interval * fps)
        per_interval = int(interval * fps)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % per_interval == 0:
                frame_path = os.path.join(output_dir, f'{frame_count}.png')
                cv2.imwrite(frame_path, frame)
                writer.writerow([frame_path])
            frame_count += 1
        print(f"total frames: {frame_count+1}")
        cap.release()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument('--root', type=str, default=os.environ.get("DATA_ROOT", ""), help='Root directory')
    parser.add_argument('--subname', type=str, default='16x24', help='Subname directory')
    
    args = parser.parse_args()
    
    root = args.root
    subname = args.subname
    
    #运行之前请预先生成scv文件
    scv_path = os.path.join(root, subname, 'videos.csv')
    save_dir = os.path.join(root, 'frames')
    
    df = pd.read_csv(scv_path)
    for index, row in df.iterrows():
        video = row['path']
        frame_save_dir = os.path.join(save_dir, video.split('/')[-1].split('.')[0])
        # extract_frames(video, frame_save_dir, interval=0.15)
        extract_frames(video, frame_save_dir, interval=0.2)
        #将frame_save_dir写入csv
        df.at[index, 'frame_save_dir'] = frame_save_dir
    df.to_csv(scv_path, index=False)