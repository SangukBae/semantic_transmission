import os
import glob
import numpy as np
from helper.utils import AverageMeter, cal_psnr, cal_msssim, np2tensor
from skimage import io
from loss.distortion import Distortion
import clip
import torch
from PIL import Image
import torchvision.transforms as T

# 定义转换组合
transform = T.Compose(
    [
        T.CenterCrop(256),
        # T.ToTensor()
    ]
)

cos = torch.nn.CosineSimilarity(dim=1)


class config:
    distortion_metric = "MS-SSIM"


# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def calculate_clip_similarity(frame_path_1, frame_path_2):
    image1 = Image.open(frame_path_1)
    image2 = Image.open(frame_path_2)
    image1 = transform(image1)
    image2 = transform(image2)

    image1 = preprocess(image1).unsqueeze(0).to(device)
    image2 = preprocess(image2).unsqueeze(0).to(device)

    image1_features = model.encode_image(image1)
    image2_features = model.encode_image(image2)

    similarity = cos(image1_features, image2_features).item()
    return (similarity + 1) / 2  # Normalize to [0, 1]


# Function to calculate the average CLIP score for a video
def calculate_average_clip_score(video_folder_1, video_folder_2):
    clip_scores = []

    for frame_file in os.listdir(video_folder_1):
        frame_path_1 = os.path.join(video_folder_1, frame_file)
        frame_path_2 = os.path.join(video_folder_2, frame_file)

        if os.path.exists(frame_path_1) and os.path.exists(frame_path_2):
            clip_score = calculate_clip_similarity(frame_path_1, frame_path_2)
            clip_scores.append(clip_score)

    if clip_scores:
        return np.mean(clip_scores)
    else:
        return 0


distortion = Distortion(config).cuda()

# step 1: rescale the video
testset_dir = r"video_compress/clipscore/test_video_vc_v1"
savedir = os.path.dirname(os.path.dirname(testset_dir)) + r"/new_264_265/rescale"
if not os.path.exists(savedir):
    os.mkdir(savedir)

seq_list = glob.glob(testset_dir + r"/*.mp4")
for seq_dir in seq_list:
    seq_name = os.path.basename(seq_dir)[:-4]
    # Get video resolution using ffprobe
    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {seq_dir}"
    resolution = os.popen(probe_cmd).read().strip().split(",")
    W, H = int(resolution[0]), int(resolution[1])
    new_H = 256
    new_W = 256
    out_name = r"/{}".format(seq_name)
    # 1. Rescale the original MP4 video via ffmpeg
    rescaled_video_path = savedir + out_name + ".mp4"
    if not os.path.exists(rescaled_video_path):
        os.system(
            r"ffmpeg -i {} -pix_fmt yuv420p {}".format(
                seq_dir,
                rescaled_video_path,
            )
        )
    # 2. Make the video path
    img_save_dir = savedir + out_name
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    # 3. Convert MP4 to PNG with scaling
    # if not os.path.exists(img_save_dir + r"/im001.png"):
    #     os.system(
    #         r"ffmpeg -i {} -vf scale={}:{} -pix_fmt yuv420p -f image2 {}/im%03d.png".format(
    #             rescaled_video_path, new_W, new_H, img_save_dir
    #         )
    #     )
    os.system(
        r"ffmpeg -i {} -pix_fmt yuv420p -f image2 {}/im%03d.png".format(
            rescaled_video_path, img_save_dir
        )
    )

# step 2: encode the video
preset = "veryfast"
encoder = "h265"

rescaled_dir = savedir
seq_list = glob.glob(rescaled_dir + r"/*.mp4")
# 41 39 37 35 33 30
QP = 48
GoP_size = 4
vframes = 8

from loss.perceptual_similarity.perceptual_loss import PerceptualLoss
import torch

perceptual_loss = PerceptualLoss(
    model="net-lin", net="alex", use_gpu=torch.cuda.is_available(), gpu_ids=[0]
)

for seq_dir in seq_list:
    # seq_dir = r"HEVC/ClassA/crop/PeopleOnStreet_2560x1600_30.yuv"
    seq_name = os.path.basename(seq_dir)[:-4]

    # Get video frame rate using ffprobe
    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 {seq_dir}"
    frame_rate = os.popen(probe_cmd).read().strip()
    num, denom = map(int, frame_rate.split("/"))
    FR = num / denom
    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {seq_dir}"
    resolution = os.popen(probe_cmd).read().strip().split(",")
    W, H = int(resolution[0]), int(resolution[1])
    video_save_dir = os.path.dirname(rescaled_dir) + r"/out/{}/{}_QP{}_GoP{}".format(
        encoder, preset, QP, GoP_size
    )
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    video_save_name = video_save_dir + r"/{}.mp4".format(seq_name)
    report_save_name = video_save_dir + r"/{}.log".format(seq_name)
    metric_save_name = video_save_dir + r"/Results_{}.txt".format(seq_name)

    if not os.path.exists(video_save_name):
        if encoder == "h265":
            os.system(
                r"ffmpeg -y -i {} -pix_fmt yuv420p "
                r"-s {}x{} -c:v libx265 "
                r"-preset {} -tune zerolatency -threads 4 "
                r"-x265-params slices=1:qp={}:bframes=0:keyint={}:csv={}:csv-log-level=1:verbose=1 {}".format(
                    seq_dir,
                    W,
                    H,
                    preset,
                    QP,
                    GoP_size,
                    report_save_name,
                    video_save_name,
                )
            )
        elif encoder == "h264":
            os.system(
                r"ffmpeg -y -i {} -pix_fmt yuv420p "
                r"-s {}x{} -r {} -c:v libx264 "
                r"-preset {} -tune zerolatency -qp {} -g {} -bf 0 -b_strategy 0 -threads 4 "
                r" {}".format(seq_dir, W, H, FR, preset, QP, GoP_size, video_save_name)
            )
        # if encoder == "h265":
        #     os.system(
        #         r"ffmpeg -y -i {} -pix_fmt yuv420p "
        #         r"-c:v libx265 "
        #         r"-preset {} -tune zerolatency -threads 4 "
        #         r"-x265-params slices=1:qp={}:bframes=0:keyint={}:csv={}:csv-log-level=1:verbose=1 {}".format(
        #             seq_dir,
        #             preset,
        #             QP,
        #             GoP_size,
        #             report_save_name,
        #             video_save_name,
        #         )
        #     )
        # elif encoder == "h264":
        #     os.system(
        #         r"ffmpeg -y -i {} -pix_fmt yuv420p "
        #         r"-r {} -c:v libx264 "
        #         r"-preset {} -tune zerolatency -qp {} -g {} -bf 0 -b_strategy 0 -threads 4 "
        #         r" {}".format(seq_dir, FR, preset, QP, GoP_size, video_save_name)
        #     )
    #  step 3: video2img
    img_save_dir = video_save_dir + r"/{}".format(seq_name)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    # if not os.path.exists(img_save_dir + r"/im001.png"):
    #     os.system(
    #         r"ffmpeg -i {} -vf scale={}:{} -f image2 {}/im%03d.png".format(
    #             video_save_name, new_W, new_H, img_save_dir
    #         )
    #     )
    # if not os.path.exists(img_save_dir + r"/im001.png"):
    os.system(
        r"ffmpeg -i {} -vf scale={}:{} -f image2 {}/im%03d.png".format(
            video_save_name, W, H, img_save_dir
        )
    )

#  step 4: calculate bpp PSNR SSIM
bpps = []
psnrs, ms_ssims, lpipses, clip_scores = [AverageMeter() for _ in range(4)]
metric_save_name = video_save_dir + r"/Results.txt"
with open(metric_save_name, "w") as file:
    for seq_dir in seq_list:
        seq_name = os.path.basename(seq_dir)[:-4]
        W = 256
        H = 256
        img_save_dir = video_save_dir + r"/{}".format(seq_name)
        report_save_name = video_save_dir + r"/{}.log".format(seq_name)
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
            file_size = (
                os.path.getsize(video_save_dir + r"/{}.mp4".format(seq_name)) * 8
            )
            bpps.append(file_size / (W * H * vframes * 3))

        for i in range(vframes):
            source = (
                rescaled_dir
                + r"/{}".format(seq_name)
                + r"/im{}.png".format(str(i + 1).zfill(3))
            )

            h265 = img_save_dir + r"/im{}.png".format(str(i + 1).zfill(3))

            source_img = io.imread(source)
            h265_img = io.imread(h265)

            psnr_val = cal_psnr(source_img, h265_img)

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

            clip_score_val = calculate_clip_similarity(h265, source)

            psnrs.update(psnr_val)
            ms_ssims.update(ms_ssim_val.item())
            lpipses.update(lpips_val.item())
            clip_scores.update(clip_score_val)

            if encoder == "h265":
                log = " | ".join(
                    [
                        f"[{seq_name}]",
                        f"bpp {bpps[i]:.4f} ({np.array(bpps).mean():.4f})",
                        f"PSNR {psnrs.val:.4f} ({psnrs.avg:.4f})",
                        f"MS-SSIM {ms_ssims.val:.4f} ({ms_ssims.avg:.4f})",
                        f"LPIPS {lpipses.val:.4f} ({lpipses.avg:.4f})",
                        f"CLIP {clip_scores.val:.4f} ({clip_scores.avg:.4f})",
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
                        f"CLIP {clip_scores.val:.4f} ({clip_scores.avg:.4f})",
                    ]
                )
            print(log)
            file.write(log + "\n")


code_rate = 0.75

modulation_map = {"BPSK": 1, "QPSK": 2, "4QAM": 2, "16QAM": 4, "64QAM": 6}

modulation_method = "4QAM"
modulation_order = modulation_map[modulation_method]
channel_config = code_rate, modulation_order
average_bpp = np.array(bpps).mean()


def bpp2cbr(bpp, channel_config):
    # channel_config = [channel_code_rate, modulation_order]

    return bpp * 2 / (channel_config[0] * channel_config[1])


cbr = bpp2cbr(average_bpp, channel_config)

print(f"{cbr:.4f}")
