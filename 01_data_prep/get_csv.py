import os
import csv
import argparse

def generate_csv(root, target):
    video_folder = os.path.join(root, target)
    print(f"正在生成 CSV 文件：{video_folder}")
    #判断文件夹是否存在
    if not os.path.exists(video_folder):
        print(f"文件夹不存在：{video_folder}")
        return
    # 列出目录中所有的视频文件（假设后缀为 .mp4）
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

    # CSV 文件路径
    csv_file = os.path.join(video_folder, 'videos.csv')

    # 将路径写入 CSV 文件
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['path'])  # 写入 CSV 文件的表头
        for video_path in video_files:
            writer.writerow([video_path])

    print(f"CSV 文件已生成：{csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成包含视频文件路径的 CSV 文件")
    parser.add_argument('--root', type=str, default=os.environ.get("DATA_ROOT", ""), help='根目录路径')
    parser.add_argument('--subname', type=str, default='16x24', help='子目录名称')

    args = parser.parse_args()

    generate_csv(args.root, args.subname)
