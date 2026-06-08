import cv2
import os


def convert_bgr_to_rgb(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(rgb_frame)

            cap.release()
            out.release()


if __name__ == "__main__":
    input_folder = r"E:/Python_project/compressai/compressai/video_compress/clipscore/test_video_vc"
    output_folder = r"E:/Python_project/compressai/compressai/video_compress/clipscore/test_video_vc_v1"
    convert_bgr_to_rgb(input_folder, output_folder)
