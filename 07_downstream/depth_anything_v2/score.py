# 读取两个参数csv_1和csv_2，将两个csv文件中的path对应的视频,计算其深度信息，并计算整个视频的深度指标，将结果保存到csv_1_vs_csv_2.csv中
import os
import sys
import pandas as pd
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.util.metric import eval_depth
import matplotlib

# 调用方法：python score.py origin_path.csv key_framesinternvl_diff_0.35_path.csv

def calculate_metrics(ref_path, hyp_path, depth_anything, method1, method2):
    # 用于存储所有帧的指标
    total_metrics = {
        'd1': 0, 'd2': 0, 'd3': 0, 
        'abs_rel': 0, 'sq_rel': 0, 
        'rmse': 0, 'rmse_log': 0, 
        'log10': 0, 'silog': 0
    }
    frame_count = 0

    # 读取视频
    ref_video = cv2.VideoCapture(ref_path)
    hyp_video = cv2.VideoCapture(hyp_path)
    frame_width1, frame_height1 = int(ref_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(ref_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width2, frame_height2 = int(hyp_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(hyp_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_rate1 = int(ref_video.get(cv2.CAP_PROP_FPS))
    frame_rate2 = int(hyp_video.get(cv2.CAP_PROP_FPS))

    out_path = "Sora"
    output_path1 = os.path.join(out_path, method1)
    # 讲output_path1结尾的_path去掉
    output_path1 = output_path1.split('_path')[0]
    # 如果输出目录不存在则创建
    if not os.path.exists(output_path1):
        os.makedirs(output_path1)
    output_path1 = os.path.join(output_path1, os.path.splitext(os.path.basename(ref_path))[0] + '.mp4')

    output_path2 = os.path.join(out_path, method2)
    output_path2 = output_path2.split('_path')[0]
    if not os.path.exists(output_path2):
        os.makedirs(output_path2)
    output_path2 = os.path.join(output_path2, os.path.splitext(os.path.basename(hyp_path))[0] + '.mp4')

    out1 = cv2.VideoWriter(output_path1, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate1, (frame_width1, frame_height1))
    out2 = cv2.VideoWriter(output_path2, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate2, (frame_width2, frame_height2))
    while True:
        ret1, frame1 = ref_video.read()
        ret2, frame2 = hyp_video.read()
        
        if not ret1 or not ret2:
            break

        with torch.no_grad():
            # 处理参考视频帧
            depth1 = depth_anything.infer_image(frame1, 518)
            # 处理假设视频帧
            depth2 = depth_anything.infer_image(frame2, 518)
            
            # 转换为tensor
            depth1_tensor = torch.from_numpy(depth1).cuda()
            depth2_tensor = torch.from_numpy(depth2).cuda()
            
            # 创建有效mask (假设所有像素都有效)
            valid_mask = torch.ones_like(depth1_tensor, dtype=torch.bool)
            # 找到depth1_tensor的最大值
            max_depth1 = depth1_tensor.max()
            # 找到depth2_tensor的最大值
            max_depth2 = depth2_tensor.max()

            # max_depth = min(max_depth1, max_depth2)
            max_depth = max_depth1
            # print(f"max_depth: {max_depth}, max_depth1: {max_depth1}, max_depth2: {max_depth2}")

            

            valid_mask = valid_mask & (depth1_tensor >= 0.001) & (depth1_tensor <= max_depth) & (depth2_tensor >= 0.001) & (depth2_tensor <= max_depth)
            
            # 计算指标
            metrics = eval_depth(depth1_tensor[valid_mask], depth2_tensor[valid_mask])
            
            depth1 = (depth1 - depth1.min()) / (depth1.max() - depth1.min()) * 255.0
            depth1 = depth1.astype(np.uint8)
            depth2 = (depth2 - depth2.min()) / (depth2.max() - depth2.min()) * 255.0
            depth2 = depth2.astype(np.uint8)
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth1 =(cmap(depth1)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            depth2 =(cmap(depth2)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            out1.write(depth1)
            out2.write(depth2)
            
            # 累加指标
            for k, v in metrics.items():
                if torch.isnan(torch.tensor(v)):
                    print(f"Warning: {k} is nan!")
                    print(metrics)
                    continue
                total_metrics[k] += v
        frame_count += 1

    # 释放视频资源
    ref_video.release()
    hyp_video.release()
    out1.release()
    out2.release()

    # print(f"frame_count: {frame_count}")
    # 计算平均指标
    if frame_count > 0:
        for k in total_metrics:
            total_metrics[k] /= frame_count

    return total_metrics

def normalize_path(path):
    # 获取基础文件名
    basename = os.path.basename(path)
    # 移除_0000等后缀
    basename = basename.split('_0000')[0]
    # basename = basename.split('.')[0]
    return basename

def main():
    if len(sys.argv) != 3:
        print("Usage: python score.py <csv_1> <csv_2>")
        sys.exit(1)
        
    # 配置设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 配置并加载模型(只加载一次)
    model_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    depth_anything = DepthAnythingV2(**model_config)
    depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    csv_1_path = sys.argv[1]
    csv_2_path = sys.argv[2]
    
    # 读取CSV文件
    df1 = pd.read_csv(csv_1_path)
    df2 = pd.read_csv(csv_2_path)
    
    # 创建结果列表
    results = []
    
    # 创建df2的path到标准化basename的映射
    df2_path_map = {normalize_path(path): path for path in df2['path']}
    
    # 计算每对视频的指标
    for path1 in df1['path']:
        basename1 = os.path.basename(path1)
        basename1 = basename1.split('.')[0]
        
        # 查找对应的path2
        if basename1 in df2_path_map:
            path2 = df2_path_map[basename1]
            print(f"Calculating metrics for 【{basename1}】 ...")
            metrics = calculate_metrics(path1, path2, depth_anything, os.path.splitext(os.path.basename(csv_1_path))[0], os.path.splitext(os.path.basename(csv_2_path))[0])
            print(f"average metrics: {metrics}")

            results.append({
                'path1': path1,
                **metrics
            })

    
    # 创建结果DataFrame并保存
    output_df = pd.DataFrame(results)
    output_name = f"Sora/{os.path.splitext(os.path.basename(csv_1_path))[0]}_vs_{os.path.splitext(os.path.basename(csv_2_path))[0]}.csv"
    output_df.to_csv(output_name, index=False)
    print(f"Results saved to {output_name}")

    # 计算平均值
    metrics_avg = {
        'path1': 'Average',
        'd1': output_df['d1'].mean(),
        'd2': output_df['d2'].mean(),
        'd3': output_df['d3'].mean(),
        'abs_rel': output_df['abs_rel'].mean(),
        'sq_rel': output_df['sq_rel'].mean(),
        'rmse': output_df['rmse'].mean(),
        'rmse_log': output_df['rmse_log'].mean(),
        'log10': output_df['log10'].mean(),
        'silog': output_df['silog'].mean()
    }

    #打印平均值结果

    print("\nAverage Metrics:")
    print(metrics_avg)


if __name__ == "__main__":
    main()
