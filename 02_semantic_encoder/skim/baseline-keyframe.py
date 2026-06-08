import logging
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm

def baseline_keyframe_extraction(frame_paths, num_keyframes):
    """
    Baseline method for extracting keyframes.
    Args:
        frame_paths: List of all frame paths from the video.
        num_keyframes: The number of keyframes to extract.
    Returns:
        keyframe_paths: List of selected keyframe paths.
    """
    # Ensure we have at least as many frames as keyframes requested
    total_frames = len(frame_paths)
    assert total_frames >= num_keyframes, "Number of keyframes exceeds available frames."

    # Select first and last frames as keyframes
    keyframe_indices = [0, total_frames - 1]

    # Calculate the interval for evenly spaced keyframes
    interval = total_frames // (num_keyframes - 1)
    
    # Select evenly spaced keyframes between the first and last frame
    for i in range(1, num_keyframes - 1):
        keyframe_indices.append(i * interval)

    # Sort indices to maintain the order of frames
    keyframe_indices = sorted(set(keyframe_indices))

    # Extract keyframe paths
    keyframe_paths = [frame_paths[idx] for idx in keyframe_indices]

    return keyframe_paths


def main(args):
    csv_path = args.csv_path
    num_keyframes = args.num_keyframes
    method = "baseline_"+str(num_keyframes)+"frames"

    df = pd.read_csv(csv_path)
    frame_paths = []
    
    for index, row in df.iterrows():
        frame_path = row['frame_save_dir']
        frame_paths.append(frame_path)
    
    for frame_path in frame_paths:
        all_frames = []
        frame_numbers = []
        
        # Load frame paths from the frames.csv
        open_frames = pd.read_csv(frame_path + '/frames.csv')
        for index, row in open_frames.iterrows():
            frame = row['frame_path']
            all_frames.append(frame)
        
        logging.info(f"Processing video at {frame_path} with {len(all_frames)} frames.")
        
        # Extract keyframes using the baseline method
        keyframes = baseline_keyframe_extraction(all_frames, num_keyframes)
        
        # Create keyframes directory
        key_frame_path = os.path.join(frame_path, 'key_frames-'+ method)
        os.makedirs(key_frame_path, exist_ok=True)

        # Copy selected keyframes to key_frames folder
        for keyframe in keyframes:
            frame_number = keyframe.split('/')[-1].split('.')[0]
            key_frame_file = os.path.join(key_frame_path, f'{frame_number}.png')
            shutil.copy(keyframe, key_frame_file)
            logging.info(f"Copied keyframe: {keyframe} to {key_frame_file}")

        logging.info(f"Keyframes extraction completed for {frame_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the CSV file containing frame directories.")
    parser.add_argument("--num-keyframes", type=int, required=True, help="Number of keyframes to extract per video.")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler("baseline_keyframe_extraction.log"),
            logging.StreamHandler()
        ]
    )
    
    main(args)
