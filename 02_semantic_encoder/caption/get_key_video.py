
"""
This script processes video files by extracting keyframes and cutting the videos into segments based on these keyframes. 
The keyframes are extracted using a specified method, and the resulting video segments are saved in an organized directory structure. 
Additionally, the paths of the generated video segments are recorded in a CSV file.
Functions:
    - get_video_files(directory): Retrieves all video files from the specified directory.
    - get_keyframes(folder): Retrieves all keyframe filenames from the specified folder.
    - extract_frame_time(frame_file, fps): Extracts the timestamp (in seconds) from the keyframe filename.
    - cut_video(video_path, keyframes_folder, csv_writer, output_folder, video_name, method): Cuts the video into segments based on the keyframes and saves the segments.
    - process_videos(videos_directory, frames_directory, method, csv_file): Processes each video in the specified directory, cutting it into segments based on keyframes.
Data Storage:
    - The generated video segments are saved in a directory structure under the specified output folder, organized by method and video name.
    - The paths of the generated video segments are recorded in a CSV file located in the videos directory, named according to the keyframe extraction method (e.g., 'key_framesinternvl_diff_0.35_video_paths.csv').
Usage:
    The script is executed with command-line arguments specifying the root directory, video folder name, frames folder name, and keyframe extraction method. 
    Example command:
        python get_key_video.py --root_directory /path/to/root --videos video_folder --frames frames_folder --method keyframe_method
"""
import os
import shutil
import cv2
import csv
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def get_video_files(directory):
    """获取目录中的所有视频文件"""
    return [f for f in os.listdir(directory) if f.endswith('.mp4')]

def get_keyframes(folder):
    """获取指定文件夹中的所有关键帧文件名"""
    return sorted([f for f in os.listdir(folder) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

def extract_frame_time(frame_file, fps):
    """从关键帧文件名中提取时间点（秒）"""
    frame_number = int(frame_file.split('.')[0])
    return frame_number / fps

def cut_video(video_path, keyframes_folder, csv_writer, output_folder, video_name, method):
    """根据关键帧序列切分视频"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = get_keyframes(keyframes_folder)
    print("keyframes: ", keyframes)

    out_folder = f"{output_folder}/{method}/{video_name.split('.')[0]}"
    #如果out_folder文件夹已经存在，则删除
    if os.path.exists(f"{out_folder}"):
        shutil.rmtree(out_folder)
    #如果out_folder文件夹不存在，则创建
    if not os.path.exists(f"{out_folder}"):
        os.makedirs(f"{out_folder}")
    start_time = 0
    start_frame = 0
    for i, keyframe in enumerate(keyframes):
        end_time = extract_frame_time(keyframe, fps)
        if end_time - start_time == 0:
            continue # 跳过持续时间为0的片段
        # output_path = f"{video_path.split('.')[0]}/{i}.{start_frame}.{keyframe.split('.')[0]}.mp4"
        output_path = f"{out_folder}/{i}.{start_frame}.{keyframe.split('.')[0]}.mp4"
        # print("output_path: ", output_path)
        # 将output_path添加到csv文件中
        csv_writer.writerow([output_path])

        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)
        start_time = end_time
        start_frame = int(keyframe.split('.')[0])
    
    #处理只有一个关键帧的情况
    if start_time == 0:#那么就是第0帧到最后一帧
        # output_path = f"{video_path.split('.')[0]}/1.0.{cap.get(cv2.CAP_PROP_FRAME_COUNT)}.mp4"
        output_path = f"{out_folder}/1.0.{cap.get(cv2.CAP_PROP_FRAME_COUNT)}.mp4"
        csv_writer.writerow([output_path])
        ffmpeg_extract_subclip(video_path, 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps, targetname=output_path)
    cap.release()

def process_videos(videos_directory, frames_directory, method, csv_file):
    """遍历视频目录，处理每个视频"""
    videos = get_video_files(videos_directory)
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['path'])  # 写入CSV列名
        for video in videos:
            video_path = os.path.join(videos_directory, video)
            keyframes_folder = os.path.join(frames_directory, video.split('.')[0])
            keyframes_folder = os.path.join(keyframes_folder, method)
            print("keyframes_folder: ", keyframes_folder)
            if os.path.exists(keyframes_folder):
                cut_video(video_path, keyframes_folder, csv_writer, videos_directory, video, method)
    print("csv_file: ", csv_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_directory', type=str, help='关键帧所在的根目录', default=os.environ.get("DATA_ROOT", ""))
    parser.add_argument('--videos', type=str, help='视频文件夹名称,如16x24', default='16x24')
    parser.add_argument('--frames', type=str, help='帧文件夹名称,如frames', default='frames')
    parser.add_argument('--method', type=str, help='关键帧提取方法名称', default='key_framesinternvl_diff_0.35')
    args = parser.parse_args()
    root_directory = args.root_directory
    videos = args.videos
    videos_directory = os.path.join(root_directory, videos)
    frames = args.frames
    frames_directory = os.path.join(root_directory, frames)
    method = args.method
    csv_file = os.path.join(videos_directory, f'{method}_video_paths.csv')
    process_videos(videos_directory, frames_directory, method, csv_file)
    # 将csv_file作为输出参数
    
    # 加入参数之前的代码
    # root_directory = '$DATA_ROOT'
    # videos = "16x24"
    # videos_directory = os.path.join(root_directory, videos)
    # frames = "frames"
    # frames_directory = os.path.join(root_directory, frames)
    # method = 'key_frames-gpt-optimization'
    # csv_file = os.path.join(videos_directory, 'video_paths.csv')
    # process_videos(videos_directory, frames_directory, method, csv_file)
