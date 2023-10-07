"""
[Reference]
# data is a list of tuples (csv_index, filename, labels).
# labels is a numpy array of shape (num_frames, num_actions).
# Array is multilabel, if action j is performed at frame i, then labels[i][j] = 1.

[Note]
# This script extracts RGB features from the videos using pretrained-with-mmaction ResNet50.
# The features are saved in a dictionary with the video path as the key.
# The dictionary is saved as a pickle file.
"""

import cv2
import pickle
import numpy as np
import os
import torch

from resnet import resnet50
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ROOT_PATH = os.path.abspath("G:\내 드라이브\Project\RGBLab\ABB\Data")
DATA_PATH = []

resized_postpix = '_resized'

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    return duration

def uniform_temporal_subsample(video, num_frames):
    total_frames = len(video)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return [video[i] for i in indices]

def short_side_scale(image, short_side_length):
    h, w, _ = image.shape
    aspect_ratio = w / h

    if h < w:
        new_h = short_side_length
        new_w = int(new_h * aspect_ratio)
    else:
        new_w = short_side_length
        new_h = int(new_w / aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
])

if __name__ == '__main__':

    with open('processed_data/dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    model = resnet50()
    model.to(device)
    model.eval()

    # to get full abs path of the videos
    for single_data in data:
        DATA_PATH.append(os.path.join(ROOT_PATH, single_data[1]))

    for i, data_path in enumerate(DATA_PATH):
        DATA_PATH[i] = data_path + resized_postpix + '.mp4'

    num_frames = 32
    sampling_rate = 6
    frames_per_second = 30
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    side_size = 360
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]

    all_outputs = {}

    for video_file in tqdm(DATA_PATH, desc='Extracting videos', ncols=150):

        tqdm.write(f"Currently processing: {video_file}")

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Reading frames", ncols=100) as pbar:
            video = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                resized_video = short_side_scale(frame, side_size)
                video.append(resized_video)
                pbar.update(1)

        video = uniform_temporal_subsample(video, num_frames)

        for i, frame in enumerate(video):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR을 사용하므로 RGB로 변환
            frame = transform(frame)  # numpy 배열을 텐서로 변환하고 정규화
            video[i] = frame

        video_tensor = torch.stack(video).to(device)

        with torch.no_grad():
            outputs = model(video_tensor)

        all_outputs[video_file] = outputs.cpu().numpy()

        cap.release()

    # save features
    with open('processed_data/features.pkl', 'wb') as f:
        pickle.dump(all_outputs, f)