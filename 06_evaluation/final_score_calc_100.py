import os
import csv
import cv2
import torch
import clip
import lpips
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import argparse
from skimage.metrics import structural_similarity as ssim
from DISTS_pytorch import DISTS  # 导入 DISTS 库

# 用于存储各项指标的列表
clip_generateds = []
psnr_generateds = []
ssim_generateds = []
lpips_generateds = []
dists_generateds = []  # DISTS 生成结果列表
psnr_first_100_generateds = []
psnr_last_100_generateds = []
ssim_first_100_generateds = []
ssim_last_100_generateds = []


# 设置设备和模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
cos = torch.nn.CosineSimilarity(dim=1)
lpips_model = lpips.LPIPS(net='vgg').to(device)
dists_model = DISTS().to(device)  # 初始化 DISTS 模型

# 计算 CLIP 相似度
def calculate_clip_similarity(frame_path_1, frame_path_2):
    image1 = Image.open(frame_path_1).convert('RGB')
    image2 = Image.open(frame_path_2).convert('RGB')

    if image1.size != image2.size:
        image2 = image2.resize(image1.size)  # 修改为使用 image1 的尺寸

    image1 = preprocess(image1).unsqueeze(0).to(device)
    image2 = preprocess(image2).unsqueeze(0).to(device)

    with torch.no_grad():
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)

        similarity = cos(image1_features, image2_features).item()
    return (similarity + 1) / 2  # 归一化到 [0, 1]

# 计算 PSNR
def calculate_psnr(frame_path_1, frame_path_2):
    img1 = cv2.imread(frame_path_1)
    img2 = cv2.imread(frame_path_2)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # 修改为使用 img1 的尺寸

    psnr_value = cv2.PSNR(img1, img2)
    return psnr_value

# 计算 SSIM
def calculate_ssim(frame_path_1, frame_path_2):
    img1 = cv2.imread(frame_path_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame_path_2, cv2.IMREAD_GRAYSCALE)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # 修改为使用 img1 的尺寸
    ssim_value = ssim(img1, img2)
    return ssim_value

# 计算 LPIPS
def calculate_lpips(frame_path_1, frame_path_2):
    image1 = lpips.im2tensor(lpips.load_image(frame_path_1)).to(device)
    image2 = lpips.im2tensor(lpips.load_image(frame_path_2)).to(device)

    if image1.shape != image2.shape:
        image2 = T.Resize(image1.shape[-2:])(image2)  # 修改为使用 image1 的尺寸

    with torch.no_grad():
        lpips_value = lpips_model(image1, image2).item()
    return lpips_value

# 计算 DISTS
def calculate_dists(frame_path_1, frame_path_2):
    # 使用 PIL 加载图像并转换为 RGB 格式
    image1 = Image.open(frame_path_1).convert('RGB')
    image2 = Image.open(frame_path_2).convert('RGB')

    # 调整大小以匹配
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)  # 修改为使用 image1 的尺寸

    # 转换为 Tensor 并归一化到 [0,1]
    transform = T.Compose([
        T.ToTensor(),  # 转换为 Tensor 并归一化到 [0,1]
    ])

    image1 = transform(image1).unsqueeze(0).to(device)
    image2 = transform(image2).unsqueeze(0).to(device)

    with torch.no_grad():
        dists_value = dists_model(image1, image2).item()
    return dists_value

# 计算视频的平均分数（PSNR 和 SSIM 分别对前 100 帧和后 100 帧）
def calculate_average_scores(video_folder_1, video_folder_2):
    frame_files = sorted(os.listdir(video_folder_1), key=lambda x: int(os.path.splitext(x)[0]))
    total_frames = len(frame_files)

    # 选择前 100 和后 100 帧索引
    first_100 = frame_files[:100]
    last_100 = frame_files[-100:]

    clip_scores, lpips_scores, dists_scores = [], [], []
    psnr_scores, ssim_scores = [], []
    psnr_scores_first, psnr_scores_last = [], []
    ssim_scores_first, ssim_scores_last = [], []

    for i, frame_file in enumerate(frame_files):
        frame_path_1 = os.path.join(video_folder_1, frame_file)
        frame_path_2 = os.path.join(video_folder_2, frame_file)

        if os.path.exists(frame_path_1) and os.path.exists(frame_path_2):
            clip_score = calculate_clip_similarity(frame_path_1, frame_path_2)
            lpips_score = calculate_lpips(frame_path_1, frame_path_2)
            dists_score = calculate_dists(frame_path_1, frame_path_2)
            psnr_score = calculate_psnr(frame_path_1, frame_path_2)
            ssim_score = calculate_ssim(frame_path_1, frame_path_2)

            clip_scores.append(clip_score)
            lpips_scores.append(lpips_score)
            dists_scores.append(dists_score)
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)

            if frame_file in first_100:
                psnr_scores_first.append(psnr_score)
                ssim_scores_first.append(ssim_score)
            if frame_file in last_100:
                psnr_scores_last.append(psnr_score)
                ssim_scores_last.append(ssim_score)

    if clip_scores and lpips_scores and dists_scores:
        return {
            'clip': np.mean(clip_scores),
            'psnr_first100': np.mean(psnr_scores_first) if psnr_scores_first else 0,
            'psnr_last100': np.mean(psnr_scores_last) if psnr_scores_last else 0,
            'ssim_first100': np.mean(ssim_scores_first) if ssim_scores_first else 0,
            'ssim_last100': np.mean(ssim_scores_last) if ssim_scores_last else 0,
            'psnr': np.mean(psnr_scores),
            'ssim': np.mean(ssim_scores),
            'lpips': np.mean(lpips_scores),
            'dists': np.mean(dists_scores)
        }
    else:
        return None

# 提取视频帧
def extract_frames(video_path, output_dir, interval):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    per_interval = interval
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % per_interval == 0:
            frame_path = os.path.join(output_dir, f'{frame_count}.png')
            cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()

def clean_frame_dirs(generated_video_folder, video_name):
    """删除指定视频的帧文件夹"""
    ref_frames_dir = os.path.join(generated_video_folder, 'ref_frames', video_name)
    gen_frames_dir = os.path.join(generated_video_folder, 'gen_frames', video_name)
    
    # 使用 os.system 删除文件夹
    os.system(f'rm -rf "{ref_frames_dir}"')
    os.system(f'rm -rf "{gen_frames_dir}"')
    print(f"Cleaned frame directories for video: {video_name}")


# 主函数
def main(reference_video_folder, generated_video_folder, output_csv, interval=1):
    results = []

    # 清除之前的帧文件夹
    os.system('rm -rf ' + os.path.join(generated_video_folder, 'ref_frames'))
    os.system('rm -rf ' + os.path.join(generated_video_folder, 'gen_frames'))

    for video_file in os.listdir(reference_video_folder):
        if video_file.endswith('.mp4'):
            print(f"Processing video: {video_file}")
            video_name = video_file.split('.')[0]
            reference_video_path = os.path.join(reference_video_folder, video_file)
            generated_video_path = os.path.join(generated_video_folder, video_name + "_0000.mp4")

            # 检查生成的视频是否存在
            if not os.path.exists(generated_video_path):
                print(f"Generated video not found: {generated_video_path}")
                continue

            # 提取视频帧
            ref_frames_dir = os.path.join(generated_video_folder, 'ref_frames', video_name)
            gen_frames_dir = os.path.join(generated_video_folder, 'gen_frames', video_name)

            extract_frames(reference_video_path, ref_frames_dir, interval)
            extract_frames(generated_video_path, gen_frames_dir, interval)

            # 计算 generated 视频的分数
            print(f"Calculating scores for generated video: {video_name}...")
            generated_scores = calculate_average_scores(ref_frames_dir, gen_frames_dir)

            if generated_scores:
                results.append({
    'name': video_name,
    'clip': generated_scores['clip'],
    'psnr_first100': generated_scores['psnr_first100'],
    'psnr_last100': generated_scores['psnr_last100'],
    'ssim_first100': generated_scores['ssim_first100'],
    'ssim_last100': generated_scores['ssim_last100'],
    'psnr': generated_scores['psnr'],
    'ssim': generated_scores['ssim'],
    'lpips': generated_scores['lpips'],
    'dists': generated_scores['dists']
})
                print(f"Scores for {video_name}: CLIP={generated_scores['clip']}, PSNR={generated_scores['psnr']}, SSIM={generated_scores['ssim']}, LPIPS={generated_scores['lpips']}, DISTS={generated_scores['dists']}, PSNR_first100={generated_scores['psnr_first100']}, PSNR_last100={generated_scores['psnr_last100']}, SSIM_first100={generated_scores['ssim_first100']}, SSIM_last100={generated_scores['ssim_last100']}")

                # 添加到列表中以计算平均值
                clip_generateds.append(generated_scores['clip'])
                psnr_generateds.append(generated_scores['psnr'])
                ssim_generateds.append(generated_scores['ssim'])
                lpips_generateds.append(generated_scores['lpips'])
                dists_generateds.append(generated_scores['dists'])  # 添加 DISTS 生成结果
                psnr_first_100_generateds.append(generated_scores['psnr_first100'])
                psnr_last_100_generateds.append(generated_scores['psnr_last100'])
                ssim_first_100_generateds.append(generated_scores['ssim_first100'])
                ssim_last_100_generateds.append(generated_scores['ssim_last100'])

            else:
                print(f"No scores calculated for {video_name}")
            
            clean_frame_dirs(generated_video_folder, video_name)  # 删除帧文件夹

    # 计算平均值
    if clip_generateds:
        clip_generateds_mean = np.mean(clip_generateds)
        psnr_generateds_mean = np.mean(psnr_generateds)
        ssim_generateds_mean = np.mean(ssim_generateds)
        lpips_generateds_mean = np.mean(lpips_generateds)
        dists_generateds_mean = np.mean(dists_generateds)  # 添加 DISTS 生成平均值
        psnr_first_100_generateds_mean = np.mean(psnr_first_100_generateds)
        psnr_last_100_generateds_mean = np.mean(psnr_last_100_generateds)
        ssim_first_100_generateds_mean = np.mean(ssim_first_100_generateds)
        ssim_last_100_generateds_mean = np.mean(ssim_last_100_generateds)
        print(f"Average Scores: CLIP={clip_generateds_mean}, PSNR={psnr_generateds_mean}, SSIM={ssim_generateds_mean}, LPIPS={lpips_generateds_mean}, DISTS={dists_generateds_mean}, PSNR_first100={psnr_first_100_generateds_mean}, PSNR_last100={psnr_last_100_generateds_mean}, SSIM_first100={ssim_first_100_generateds_mean}, SSIM_last100={ssim_last_100_generateds_mean}")

        # 将平均值添加到结果中
        results.append({
            'name': 'Average',
            'clip': clip_generateds_mean,
            'psnr_first100': psnr_first_100_generateds_mean,
            'psnr_last100': psnr_last_100_generateds_mean,
            'ssim_first100': ssim_first_100_generateds_mean,
            'ssim_last100': ssim_last_100_generateds_mean,
            'psnr': psnr_generateds_mean,
            'ssim': ssim_generateds_mean,
            'lpips': lpips_generateds_mean,
            'dists': dists_generateds_mean,  # 添加 DISTS 生成平均结果
        })
        

    # 保存结果到 CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['name', 'clip', 'psnr_first100', 'psnr_last100',
              'ssim_first100', 'ssim_last100', 'psnr', 'ssim', 'lpips', 'dists']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # 删除帧文件夹
    os.system('rm -rf ' + os.path.join(generated_video_folder, 'ref_frames'))
    os.system('rm -rf ' + os.path.join(generated_video_folder, 'gen_frames'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_video_folder', type=str, default=os.path.join(os.environ.get("DATA_ROOT",""),"16x24"))
    parser.add_argument('--generated_video_folder', type=str, default=os.path.join(os.environ.get("DATA_ROOT",""),"save_dir_diff_0.35"))
    parser.add_argument('--output_csv', type=str, default='evaluation_scores_100.csv')
    args = parser.parse_args()

    reference_video_folder = args.reference_video_folder
    generated_video_folder = args.generated_video_folder
    output_csv = generated_video_folder + '/' + args.output_csv

    print(f"Reference folder: {reference_video_folder}")
    print(f"Generated folder: {generated_video_folder}")
    print(f"Output CSV: {output_csv}")
    main(reference_video_folder, generated_video_folder, output_csv)