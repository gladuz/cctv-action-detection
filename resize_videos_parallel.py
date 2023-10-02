"""
dataset.pkl is a list of tuples (csv_index, filename, labels)

resize_videos_parallel.py resizes the videos in parallel using multiprocessing library.
note that max_workers is set to a half of the number of cpu cores.
"""

import os
from tqdm import tqdm
import cv2
import pickle
from concurrent.futures import ThreadPoolExecutor

def resize_video(input_path, output_path, dim=(720, 480)):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, dim)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, dim)
        out.write(resized_frame)

    cap.release()
    out.release()
    
def process_video(data_path):
    tqdm.write(f"Currently processing: {data_path}")
    resize_video(data_path, data_path[:-4]+'_resized.mp4', dim=(TARGET_HEIGHT, TARGET_WIDTH))

if __name__ == '__main__':

    with open('processed_data/dataset.pkl', 'rb') as f:
        data = pickle.load(f) 

    TARGET_WIDTH = 480
    TARGET_HEIGHT = 720

    ROOT_PATH = os.path.abspath("G:\내 드라이브\Project\RGBLab\ABB\Data")
    DATA_PATH = []
    
    for single_data in data:
        DATA_PATH.append(os.path.join(ROOT_PATH, single_data[1]))
        
    # add .mp4 to the end of each path
    DATA_PATH = [path + '.mp4' for path in DATA_PATH]
    
    max_workers = os.cpu_count() // 2
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_video, DATA_PATH), total=len(DATA_PATH)))