import os
import subprocess
import cv2

input_bpps = []
output_bpps = []
def compress_video(input_video, output_video, snr, qp=51, gop_size=400, preset="ultrafast", bv="100"):
    """
    使用FFmpeg对视频进行压缩
    :param input_video: 输入的视频文件路径 (如: input.mp4)
    :param output_video: 输出的视频文件路径 (如: output.mp4)
    :param qp: 量化参数(QP)值，控制压缩质量，51是最低质量 (范围：0-51)
    :param gop_size: GOP (Group of Pictures) 大小，关键帧之间的间隔
    :param preset: 编码预设，控制编码速度和压缩效率之间的平衡
    """
    # 构建 FFmpeg 命令
    snr_config = {
        10: {"qp": 51, "gop_size": 20, "preset": "ultrafast"}, #46,400;0.01099 #49,40;0.01062 # 50,20;0.01031 #51,20；0.01086
        8: {"qp": 51, "gop_size": 80, "preset": "ultrafast"}, #0.00814
        6: {"qp": 51, "gop_size": 400, "preset": "ultrafast"}, #0.00764
        4: {"qp": 51, "gop_size": 400, "preset": "ultrafast"}, #0.00764
        2: {"qp": 51, "gop_size": 400, "preset": "ultrafast"}, #0.00764
        0: {"qp": 51, "gop_size": 400, "preset": "ultrafast"}, #0.00764
    }
    if snr not in snr_config:
        raise ValueError(f"SNR {snr} not in supported list: {list(snr_config.keys())}")

    config = snr_config[snr]
    qp = config["qp"]
    gop_size = config["gop_size"]
    preset = config["preset"]
    command = [
        'ffmpeg',
        '-i', input_video,           # 输入文件
        '-c:v', 'libx264',           # 使用 H.264 编码器
        '-preset', preset,           # 设置编码速度
        '-qp', str(qp),              # 设置量化参数
        '-g', str(gop_size),         # 设置GOP大小
        '-y',                        # 覆盖输出文件
        # '-b:v', bv,                  # 设置比特率
        '-profile:v', 'baseline',
        output_video                 # 输出文件
    ]

    # 打印并执行命令
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

    # 读取输出文件大小,计算bpp
    output_size = os.path.getsize(output_video)*8
    input_size = os.path.getsize(input_video)*8
    cap_out = cv2.VideoCapture(output_video)
    width_out = cap_out.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_out = cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count_out = cap_out.get(cv2.CAP_PROP_FRAME_COUNT)
    pixel_count_out = width_out * height_out * frame_count_out
    bpp_out = output_size / pixel_count_out
    cap_in = cv2.VideoCapture(input_video)
    width_in = cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_in = cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count_in = cap_in.get(cv2.CAP_PROP_FRAME_COUNT)
    pixel_count_in = width_in * height_in * frame_count_in
    #判断pixel_count_in和pixel_count_out是否相等
    if pixel_count_in != pixel_count_out:
        print("pixel count not equal!")
    bpp_in = input_size / pixel_count_in
    print(f'input bpp: {bpp_in}')
    print(f'output bpp: {bpp_out}')
    input_bpps.append(bpp_in)
    output_bpps.append(bpp_out)

def get_video_files(directory):
    """获取目录中的所有视频文件"""
    return [f for f in os.listdir(directory) if f.endswith('.mp4')]

if __name__ == "__main__":
    # Input folder of 576x320 mp4s; override with $DATA_ROOT (source env.sh) or edit here.
    video_folder = os.environ.get("VIDEO_FOLDER", os.path.join(os.environ.get("DATA_ROOT", ""), "16x24"))
    method = "H264_test"
    snr = 10
    method = method +"_"+ str(snr)
    video_files = get_video_files(video_folder)
    for video_file in video_files:
        input_file = os.path.join(video_folder, video_file)
        output_file = os.path.join(video_folder.replace("16x24", ""), method)
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, video_file)
        # 调用压缩函数
        compress_video(input_file, output_file, snr)
    #计算平均bpp
    print(f'average input bpp: {sum(input_bpps)/len(input_bpps)}')
    print(f'average output bpp: {sum(output_bpps)/len(output_bpps)}')
