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
#%%
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
from combined_flow_extractor import CombinedFlowModel

class SelectFirstAndLastFrames:
    '''
    Returns the first and last frames of each group of sampling_rate frames.
    Ex: sample_rate = 6, num_frames = 192 -> 0,5 6,11 12,17 18,23 ... 186,191
    '''
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, video):
        indices = self.get_first_last_frames_indices(video.shape[-3], self.sampling_rate)
        return torch.index_select(video, -3, indices)
    
    def get_first_last_frames_indices(self, num_frames, sampling_rate):
        indices = []
        for i in range(0, num_frames, sampling_rate):
            indices.append(i)
            indices.append(i+sampling_rate-1)
        return torch.clamp(torch.tensor(indices, dtype=torch.long), 0, num_frames-1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ROOT_PATH = os.path.abspath("G:\내 드라이브\Project\RGBLab\ABB\Data")
DATA_PATH = []

resized_postpix = ''

def get_video_duration(video_path):
    container = av.open(video_path)
    duration = container.duration / av.time_base
    return duration


#%%
if __name__ == '__main__':

    with open('processed_data/dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    model = CombinedFlowModel()
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

    side_size = 512
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                SelectFirstAndLastFrames(sampling_rate),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),

            ]
        ),
    )

    all_outputs = {}
    #DATA_PATH = ["processed_data/resized_480.mp4"]
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
            inputs = inputs.transpose(0,1)
            # Cat the odd and even frames together on the channel dimension (first and last frames of each group of sampling_rate frames)
            inputs = torch.cat([inputs[::2, :, :, :], inputs[1::2, :, :, :]], dim=1)
            inputs = inputs.to(device) # (N, 6, H, W)
            #print(inputs.shape)
            # run model to get fea[tures
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs.squeeze()
            if video_file not in all_outputs:
                all_outputs[video_file] = outputs.cpu().numpy()
            else:
                all_outputs[video_file] = np.concatenate((all_outputs[video_file], outputs.cpu().numpy()), axis=0)
    # save features
    with open('processed_data/flow_features.pkl', 'wb') as f:
       pickle.dump(all_outputs, f)
