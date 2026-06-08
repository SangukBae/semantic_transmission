import os
import glob
import numpy as np
from helper.utils import AverageMeter, cal_psnr, cal_msssim, np2tensor
from skimage import io
from loss.distortion import Distortion


class config:
    distortion_metric = "MS-SSIM"


distortion = Distortion(config).cuda()

# step 1: crop the video
testset_dir = r"HEVC/ClassA"
savedir = testset_dir + r"/crop"
if not os.path.exists(savedir):
    os.mkdir(savedir)
    print("Create the crop directory")

seq_list = glob.glob(r"HEVC/ClassA/*.yuv")
for seq_dir in seq_list:
    seq_name = seq_dir.split("/")[-1][:-4]
    seq_resolution = seq_name.split("_")[-2]
    H = int(seq_resolution.split("x")[1])
    W = int(seq_resolution.split("x")[0])
    new_H = H // 64 * 64
    new_W = W // 64 * 64
    out_name = "/{}_{}x{}_{}".format(
        seq_name.split("_")[0], new_W, new_H, seq_name.split("_")[-1]
    )
    # 1. Crop the original YUV via ffmpeg 裁剪原始YUV
    cropped_video_path = savedir + out_name + ".yuv"
    if not os.path.exists(cropped_video_path):
        os.system(
            "ffmpeg -pix_fmt yuv420p  -s {} -i {}.yuv -vf crop={}:{}:0:0 {}".format(
                seq_resolution,
                testset_dir + r"/" + seq_name,
                new_W,
                new_H,
                cropped_video_path,
            )
        )
    # 2. Make the video path
    img_save_dir = savedir + out_name
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    # 3. Convert YUV to PNG
    if not os.path.exists(img_save_dir + r"/im00001.png"):
        os.system(
            r"ffmpeg -pix_fmt yuv420p -s {}x{} -i {} -f image2 {}/im%05d.png".format(
                new_W, new_H, cropped_video_path, img_save_dir
            )
        )

# step 2: encode the video

encoder = "h265"

crop_dir = r"HEVC/ClassA/crop"
seq_list = glob.glob(r"HEVC/ClassA/crop/*.yuv")
# 41 39 37 35 33 30
preset = "veryfast" #这
QP = 35
GoP_size = 4
vframes = 100
from loss.perceptual_similarity.perceptual_loss import PerceptualLoss
import torch

perceptual_loss = PerceptualLoss(
    model="net-lin", net="alex", use_gpu=torch.cuda.is_available(), gpu_ids=[0]
)

for seq_dir in seq_list:
    # seq_dir = r"HEVC/ClassA/crop/PeopleOnStreet_2560x1600_30.yuv"
    seq_name = seq_dir.split("/")[-1][:-4]
    seq_resolution = seq_name.split("_")[-2]
    H = int(seq_resolution.split("x")[1])
    W = int(seq_resolution.split("x")[0])

    FR = int(seq_name.split("_")[-1]) 
    video_save_dir = r"HEVC/ClassA/crop/out/{}/{}_QP{}_GoP{}".format(
        encoder, preset, QP, GoP_size
    )
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    video_save_name = video_save_dir + r"/{}.mkv".format(seq_name)
    report_save_name = video_save_dir + r"/{}.log".format(seq_name)
    metric_save_name = video_save_dir + r"/Results_{}.txt".format(seq_name)
    if encoder == "h265":
        os.system(
            r"ffmpeg -y -pix_fmt yuv420p -s {}x{} "
            r"-i {} -vframes {} -c:v libx265 "
            r"-preset {} -tune zerolatency "
            r"-x265-params slices=1:qp={}:keyint={}:csv={}:csv-log-level=1:verbose=1 {}".format(
                W,
                H,
                seq_dir,
                vframes,
                preset,
                QP,
                GoP_size,
                report_save_name,
                video_save_name,
            )
        )
    elif encoder == "h264":
        # print(r"ffmpeg -pix_fmt yuv420p -s {}x{} -r {} " \
        #           r"-i {} -vframes {} -c:v libx264 " \
        #           r"-preset {} -tune zerolatency -qp {} -g {} -bf 0 -b_strategy 0 -sc_threshold 0 "
        #           r"-loglevel debug {}".format(W, H, FR, seq_dir, vframes, preset, QP, GoP_size, video_save_name))
        os.system(
            r"ffmpeg -y -pix_fmt yuv420p -s {}x{} -r {} "
            r"-i {} -vframes {} -c:v libx264 "
            r"-preset {} -tune zerolatency -qp {} -g {} -bf 2 -b_strategy 0 -sc_threshold 0 "
            r" {}".format(
                W, H, FR, seq_dir, vframes, preset, QP, GoP_size, video_save_name
            )
        )

    #  step 3: video2img
    img_save_dir = video_save_dir + r"/{}".format(seq_name)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    os.system(
        r"ffmpeg -i {} -f image2 {}/im%06d.png".format(video_save_name, img_save_dir)
    )

#  step 4: calculate bpp PSNR SSIM
bpps = [] #用于存储每一帧的bit per pixel
psnrs, ms_ssims, lpipses = [AverageMeter() for _ in range(3)]
print(psnrs, ms_ssims, lpipses)
print("Start to calculate the metrics")

metric_save_name = video_save_dir + r"/Results.txt"
with open(metric_save_name, "w") as file:
    for seq_dir in seq_list:
        seq_name = seq_dir.split("/")[-1][:-4]
        seq_resolution = seq_name.split("_")[-2]
        H = int(seq_resolution.split("x")[ 1])
        W = int(seq_resolution.split("x")[0])
        img_save_dir = video_save_dir + r"/{}".format(seq_name)
        report_save_name = video_save_dir + r"/{}.log".format(seq_name)

        # 先算出每一帧的bpp
        if encoder == "h265":
            with open(report_save_name) as f:
                lines = f.readlines()

            for l in lines:
                if "I-SLICE" in l or "i-SLICE" in l:
                    bpp = int(l.split(",")[4]) / (W * H * 3)
                    bpps.append(bpp)

                if "P-SLICE" in l:
                    bpp = int(l.split(",")[4]) / (W * H * 3)
                    bpps.append(bpp)
        else:
            file_size = os.path.getsize(video_save_name) * 8
            bpps.append(file_size / (W * H * vframes * 3))

        # 再计算PSNR SSIM LPIPS
        for i in range(vframes):
            source = (
                crop_dir
                + r"/{}".format(seq_name)
                + r"/im{}.png".format(str(i + 1).zfill(5))
            )

            h265 = img_save_dir + r"/im{}.png".format(str(i + 1).zfill(6))

            source_img = io.imread(source)
            h265_img = io.imread(h265)

            psnr_val = cal_psnr(source_img, h265_img)
            # tmpssim = 0
            # tmpssim = cal_msssim(h265_img[:, :, 0], source_img[:, :, 0])
            # tmpssim += cal_msssim(h265_img[:, :, 1], source_img[:, :, 1])
            # tmpssim += cal_msssim(h265_img[:, :, 2], source_img[:, :, 2])
            # ms_ssim_val = tmpssim / 3.0
            h265_img_cuda = (
                torch.as_tensor(h265_img / 255.0, dtype=torch.float)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .cuda()
            )
            source_img_cuda = (
                torch.as_tensor(source_img / 255.0, dtype=torch.float)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .cuda()
            )
            # print(h265_img_cuda.shape)
            ms_ssim_val = 1 - distortion(h265_img_cuda, source_img_cuda)

            lpips_val = torch.mean(
                perceptual_loss.forward(
                    np2tensor(source_img), np2tensor(h265_img), normalize=True
                )
            )
            psnrs.update(psnr_val)
            ms_ssims.update(ms_ssim_val.item())
            lpipses.update(lpips_val.item())

            if encoder == "h265":
                log = " | ".join(
                    [
                        f"[{seq_name}]",
                        f"bpp {bpps[i]:.4f} ({np.array(bpps).mean():.4f})",
                        f"PSNR {psnrs.val:.4f} ({psnrs.avg:.4f})",
                        f"MS-SSIM {ms_ssims.val:.4f} ({ms_ssims.avg:.4f})",
                        f"LPIPS {lpipses.val:.4f} ({lpipses.avg:.4f})",
                    ]
                )
            elif encoder == "h264":
                log = " | ".join(
                    [
                        f"[{seq_name}]",
                        f"bpp {bpps[-1]:.4f} ({np.array(bpps).mean():.4f})",
                        f"PSNR {psnrs.val:.4f} ({psnrs.avg:.4f})",
                        f"MS-SSIM {ms_ssims.val:.4f} ({ms_ssims.avg:.4f})",
                        f"LPIPS {lpipses.val:.4f} ({lpipses.avg:.4f})",
                    ]
                )
            print(log)
            file.write(log + "\n")
