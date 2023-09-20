#%%
from mmaction.apis import inference_recognizer, init_recognizer
import torch

config_path = 'mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_path = 'mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth' # can be a local path
img_path = 'mmaction/demo/demo.mp4'   # you can specify your own picture path

# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device can be 'cuda:0'

# %%
model
# %%
import torchvision
import tqdm
video_path = 'data/resized_480.mp4'
reader = torchvision.io.VideoReader(video_path, "video")
reader_md = reader.get_metadata()
print(reader_md)
metadata = reader.get_metadata()
reader.set_current_stream("video")

frames = []  # we are going to save the frames here.
ptss = []  # pts is a presentation timestamp in seconds (float) of each frame
for i, frame in enumerate(tqdm.tqdm(reader)):
    if (i - 5) % 10 != 0:
        continue
    frames.append(frame['data'])
    ptss.append(frame['pts'])

print("PTS for first five frames ", ptss[:5])
print("Total number of frames: ", len(frames))
print("Read data size: ", frames[0].size(0) * len(frames))
#%%
print(frames[0:18])


# %%
import torch
# process batch of frames and return the result
outputs = []
for i in tqdm.tqdm(range(0, len(frames), 16)):
    inp = torch.stack(frames[i:i+16]).numpy()
    print(inp.shape)
    model(inp)
    feat_resnet = model.backbone(model.data_preprocessor(inp)) # B x 2048 x W x H
    out = torch.nn.functional.adaptive_avg_pool2d(feat_resnet, (1, 1))
    out = out.view(out.size(0), -1)
    outputs.append(out.detach().cpu().numpy())

# %%
