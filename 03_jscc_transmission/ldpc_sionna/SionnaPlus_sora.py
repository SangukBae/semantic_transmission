
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.channel import AWGN
from sionna.utils import ebnodb2no
import glob
import cv2
import numpy as np

import math
import time
import argparse






physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("We got a GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Sorry, no GPU for you...")


def main():
    parser = argparse.ArgumentParser(description="SionnaPlus Sora Script")
    parser.add_argument("--snr", type=float, default=10, help="Signal-to-noise ratio")
    parser.add_argument("--seq_list", type=str, default=os.path.join(os.environ.get("DATA_ROOT",""),"H264_test_10"), help="Sequence list directory")
    args = parser.parse_args()

    snr = args.snr
    seq_list = args.seq_list

    # snr_config = { #h264
    #     0:  {"k": 1024 *3, "n": 1024 * 9, "order": 2}, #0.00382
    #     2: {"k": 1024 * 4, "n": 1024 * 8, "order": 2}, #1 #0.00255
    #     4: {"k": 1024 * 6, "n": 1024 * 9, "order": 2}, #1.33 #0.00191
    #     6: {"k": 1024 * 6, "n": 1024 * 8, "order": 2}, #1.5 #6,8 0.00170
    #     8: {"k": 1024 * 4, "n": 1024 * 8, "order": 4}, #2 #0.00138
    #     10: {"k": 1024 * 6, "n": 1024 * 9, "order": 4}, #0.00147  #0.00142 #0.00138 #*0.00136
    # }
    snr_config = { #h265
        0:  {"k": 1024 *3, "n": 1024 * 9, "order": 2}, 
        2: {"k": 1024 * 4, "n": 1024 * 8, "order": 2}, 
        4: {"k": 1024 * 6, "n": 1024 * 9, "order": 2}, 
        6: {"k": 1024 * 6, "n": 1024 * 8, "order": 2}, #1.5 #6,8 0.00172
        8: {"k": 1024 * 4, "n": 1024 * 8, "order": 4}, #2 #0.00138
        #Average,0.8633500336310751,24.42126796074922,24.21353602035406,0.6585796342487932,0.6517752582826016,24.298075129841713,0.6553776458521849,0.498521182368021,0.267732094449436
        10: {"k": 1024 * 6, "n": 1024 * 9, "order": 4}, #0.00138 #*0.00140
        #Average,0.8675437229085746,25.042430949311946,24.79702271580697,0.6773715110215248,0.6701627138829483,24.89977564170952,0.6736910379719014,0.4879872529987556,0.264467394999434
    }
    if snr not in snr_config:
        raise ValueError(f"SNR {snr} not in supported list: {list(snr_config.keys())}")

    print(f"Processing SNR: {snr}")
    config = snr_config[snr]
    k = config["k"]
    n = config["n"]
    order = config["order"]

    # 计算码率
    z = k / n * order
    print(f"Code rate: {z:.2f}")

    # video_save_dir = seq_list + f"_{int(snr) if snr == int(snr) else snr}_save_dir_individual"
    video_save_dir = seq_list + "_save_dir_individual"
    print(f"Saving videos to {video_save_dir}")
    # t = 0
    # k = 512 * 10
    # n = 512 * 16
    #k/n越大 压缩率越大，现在这个相当于6比特数据用8比特传输
    # 信道相关参数
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder=encoder, num_iter=20, return_infobits=True)
    # order = 4 #调制阶数，这个是4QAM调制的意思
    constellation = Constellation("qam", num_bits_per_symbol=order)
    mapper = Mapper(constellation=constellation)
    channel = AWGN()
    demapper = Demapper("app", constellation=constellation)
    ebno = snr - 10 * math.log10(order * k / n)

    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    cbrs = []
    log_save_name = video_save_dir + r"/Results.txt"
    with open(log_save_name, "w") as write_file:
        for seq_dir in glob.glob(seq_list + "/*"):
            #判断是不是视频文件
            print(seq_dir)
            if not seq_dir.endswith(".mp4"):
                continue
            seq_name = seq_dir.split("/")[-1].split(".")[0]
            print(f"Processing {seq_name}...")
            # 转成bitstream
            with open(seq_dir, "rb") as file:
                data = file.read()
            bitstream = "".join(format(byte, "08b") for byte in data)
            bitstream = np.array(list(bitstream), dtype=int)
            bitstream = bitstream.astype(int)
            # 读取视频，获取视频信息
            cap = cv2.VideoCapture(seq_dir)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            n_bits_total = bitstream.size
            n_blocks = n_bits_total // k
            residual = n_bits_total % k # 如果有剩余的bit，就多加一个block
            if residual:
                n_blocks += 1
            resized_bits = np.zeros(k * n_blocks)
            resized_bits[:n_bits_total] = bitstream
            resized_bits = resized_bits.reshape(n_blocks, k)
            resized_bits_tensor = tf.convert_to_tensor(resized_bits, dtype=tf.float32)
            # t1 = time.time()
            ldpc_code = encoder(resized_bits_tensor)
            no = ebnodb2no(ebno, num_bits_per_symbol=order, coderate=k / n)
            x = mapper(ldpc_code)
            y = channel([x, no])
            llr_ch = demapper([y, no])
            ldpc_decoded_bits = decoder(llr_ch)
            # t2 = time.time()
            # t = t2 - t1
            # print(f"{seq_name} cost: {t}")
            # ldpc_decoded_bits = decoder(ldpc_code)
            numpy_decoded_bits = ldpc_decoded_bits.numpy()
            int_decoded_bits = numpy_decoded_bits.astype(int)
            decoded_bits = int_decoded_bits.flatten()[:n_bits_total]
            # 将解码后的bitstream转换为视频文件
            video_save_name = video_save_dir + r"/{}_0000.mp4".format(seq_name)
            bytes_list = [decoded_bits[i : i + 8] for i in range(0, n_bits_total, 8)]
            bytes_list_str = ["".join(str(num) for num in byte) for byte in bytes_list]
            # 将每个字节转换为整数，并将它们组合成字节串
            byte_string = bytes([int(byte, 2) for byte in bytes_list_str])
            with open(video_save_name, "wb") as file_point:
                file_point.write(byte_string)
            # 计算cbr
            cbr = n_bits_total * n / (k * order * height * width * 3)
            cbrs.append(cbr / n_frames)

            log = " | ".join(
                [
                    f"[{seq_name}]",
                    f"cbr {cbrs[-1]:.5f} avg ({np.array(cbrs).mean():.5f})",
                ]
            )
            print(log)
            write_file.write(log + "\n")
# 将保存的文件夹赋权777

    os.system(f"chmod -R 777 {video_save_dir}")
    # 计算平均值
    avg_cbr = np.array(cbrs).mean()
    print(f"Average cbr: {avg_cbr:.5f}")


if __name__ == "__main__":
    main()
