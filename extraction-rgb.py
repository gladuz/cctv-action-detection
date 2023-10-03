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

import pickle
import numpy as np
import os
import torch
import av

from resnet import resnet50
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
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
    container = av.open(video_path)
    duration = container.duration / av.time_base
    return duration

if __name__ == '__main__':

    with open('processed_data/dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    model = resnet50()
    model.to(device)
    model.eval()

    # to get full abs path of the videos
    for single_data in data:
        DATA_PATH.append(os.path.join(ROOT_PATH, single_data[1]))

    for data_path in DATA_PATH:
        # add resized postfix to the video path
        data_path = data_path + resized_postpix

        # add .mp4 extension to the video path
        data_path = data_path + '.mp4'

    num_frames = 32
    sampling_rate = 6
    frames_per_second = 30
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    side_size = 360
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
            ]
        ),
    )

    all_outputs = {}

    for video_file in tqdm(DATA_PATH, desc='Extracting videos', ncols=150):

        tqdm.write(f"Currently processing: {video_file}")

        # feature extraction
        video = EncodedVideo.from_path(video_file)
        total_duration = get_video_duration(video_file)
        num_clips = int(total_duration//clip_duration)

        for i in tqdm(range(num_clips), desc="Extracting features", ncols=100):
            start_sec = i * clip_duration
            end_sec = start_sec + clip_duration

            # load clip and transform
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            video_data = transform(video_data)
            inputs = video_data["video"]
            inputs = inputs.transpose(0,1).to(device)

            # run model to get features
            with torch.no_grad():
                outputs = model(inputs)

            if video_file not in all_outputs:
                all_outputs[video_file] = outputs.cpu().numpy()
            else:
                all_outputs[video_file] = np.concatenate((all_outputs[video_file], outputs.cpu().numpy()), axis=0)

    # save features
    with open('processed_data/features.pkl', 'wb') as f:
        pickle.dump(all_outputs, f)