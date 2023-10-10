"""
dataset.pkl is a list of tuples (csv_index, filename, labels)
"""

import os
from tqdm import tqdm
import cv2
import pickle

def resize_video(input_path, output_path, dim=(720, 480)):
    
    # if the video is already resized, skip
    # if os.path.exists(output_path):
    #     tqdm.write(f"Video already resized: {output_path}")
    #     return

    # if the video is already resized and its size is bigger than 1 MB, skip
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
        tqdm.write(f"Video already resized: {output_path}")
        return
    
    elif os.path.exists(output_path) and os.path.getsize(output_path) <= 1000000:
        tqdm.write(f"Video already resized but its size is smaller than 1 MB: {output_path}")
        os.remove(output_path)
    
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

if __name__ == '__main__':

    with open('processed_data/dataset.pkl', 'rb') as f:
        data = pickle.load(f) 

    TARGET_WIDTH = 480
    TARGET_HEIGHT = 720

    ROOT_PATH = os.path.abspath("I:\내 드라이브\Project\RGBLab\ABB\Data")
    DATA_PATH = []
    
    for single_data in data:
        DATA_PATH.append(os.path.join(ROOT_PATH, single_data[1]))
        
    # add .mp4 to the end of each path
    DATA_PATH = [path + '.mp4' for path in DATA_PATH]
    
    for data_path in tqdm(DATA_PATH, desc="Resizing videos", ncols=100):
        tqdm.write(f"Currently processing: {data_path}")
        resize_video(data_path, data_path[:-4]+'_resized.mp4', dim=(TARGET_HEIGHT, TARGET_WIDTH))
        # resize_video(data_path, './resized.mp4', dim=(TARGET_HEIGHT, TARGET_WIDTH))
