import pickle
import numpy as np
import os
import torch
import av
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from resnet import resnet50
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from torchvision.transforms import (
    CenterCrop,
    Normalize,
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ROOT_PATH = os.path.abspath("G:/내 드라이브/Project/RGBLab/ABB/Data")
DATA_PATH = []

resized_postpix = '_resized'

def get_video_duration(video_path):
    container = av.open(video_path)
    duration = container.duration / container.streams.video[0].time_base
    return duration

def process_video(video_file, model, transform):
    # feature extraction
    video = EncodedVideo.from_path(video_file)
    total_duration = get_video_duration(video_file)
    num_clips = int(total_duration // clip_duration)

    local_outputs = {}

    for i in range(num_clips):
        start_sec = i * clip_duration
        end_sec = start_sec + clip_duration

        # load clip and transform
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = transform(video_data)
        inputs = video_data["video"]
        inputs = inputs.transpose(0, 1).to(device)

        # run model to get features
        with torch.no_grad():
            outputs = model(inputs)

        if video_file not in local_outputs:
            local_outputs[video_file] = outputs.cpu().numpy()
        else:
            local_outputs[video_file] = np.concatenate((local_outputs[video_file], outputs.cpu().numpy()), axis=0)

    return video_file, local_outputs[video_file]

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

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Normalize(mean, std),
                ShortSideScale(
                    size=side_size
                ),
            ]
        ),
    )

    all_outputs = {}
    futures = []

    with ProcessPoolExecutor() as executor:
        func = partial(process_video, model=model, transform=transform)

        futures = {executor.submit(func, data_path): data_path for data_path in DATA_PATH}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing videos'):
            data_path = futures[future]
            try:
                video_file, outputs = future.result()
                all_outputs[video_file] = outputs
            except Exception as e:
                print(f"Processing failed for video: {data_path}, error: {e}")


    # save features
    with open('processed_data/features.pkl', 'wb') as f:
        pickle.dump(all_outputs, f)
