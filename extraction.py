#%%
from resnet import resnet50
import torch
import torchvision
import tqdm
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

video_path = 'processed_data/resized_480.mp4'
side_size = 360
mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
num_frames = 32
sampling_rate = 6
frames_per_second = 30
alpha = 4

#Uniformly select num_frames from the video with a given sampling rate.
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

clip_duration = (num_frames * sampling_rate)/frames_per_second

# Code below can be run on loop until the end of the video 
# by iterating over the start_sec and end_sec variables

start_sec = 0
end_sec = start_sec + clip_duration

# Initialize an EncodedVideo helper class
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

# Apply a transform to normalize the video input
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = inputs.transpose(0,1).to(device)

# %%
model = resnet50()
model.to(device)
model.eval()
with torch.no_grad():
    outputs = model(inputs)
print(outputs.size())