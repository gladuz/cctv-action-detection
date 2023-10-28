#%%
import argparse
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from resnet import resnet50
from combined_flow_extractor import CombinedFlowModel
import torchvision.transforms as transforms
from PIL import Image
import torch_tensorrt
import numpy as np

MEAN=[123.675/255, 116.28/255, 103.53/255]
STD=[58.395/255, 57.12/255, 57.375/255]
EVERY_N_FRAMES = 6
SHORT_SIDE = 512
class FrameDataset(Dataset):

    def __init__(self, frames_dir) -> None:
        super().__init__()
        self.frames_dir = frames_dir
        self.video_name = frames_dir.split("/")[-1]
        self.labels = np.load(f"/data/common/abb_project/features/target_perframe/{self.video_name}.npy")
        self.frame_names = os.listdir(self.frames_dir)
        self.frame_paths = [os.path.join(self.frames_dir, x) for x in self.frame_names]
        self.transform = transforms.Compose([
            transforms.Resize(SHORT_SIDE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.frame_groups = [[i, i + 2, i+5] for i in range(1, len(self.frame_names)-4, EVERY_N_FRAMES)]

    def __len__(self):
        return len(self.frame_groups)
    
    def __getitem__(self, group_idx):
        # 000000.jpg 000002.jpg 000005.jpg
        image_names = [f"{x:06d}.jpg" for x in self.frame_groups[group_idx]]
        images = [Image.open(os.path.join(self.frames_dir, x)) for x in image_names]
        images = [self.transform(x) for x in images]
        images = torch.stack(images)
        label = torch.tensor(self.labels[group_idx], dtype=torch.long)
        return images, label

videos = os.listdir("/data/common/abb_project/frames")
video_paths = [os.path.join("/data/common/abb_project/frames", x) for x in videos]
#dataset = FrameDataset(video_paths[0])

#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
rgb_model = resnet50().to(device)
flow_model = CombinedFlowModel().to(device)
rgb_model.eval()
flow_model.eval()


# %%
for i in trange(len(videos)):
    dataset = FrameDataset(video_paths[i])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    rgb_feautures_list = []
    flow_features_list = []
    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        print(images.shape, labels.shape)
        break
        with torch.no_grad():
            rgb_features = rgb_model(batch[:, 1, :, :, :])
            assert rgb_features.shape == (batch.shape[0], 2048), rgb_features.shape
            flow_features = flow_model(torch.cat([images[:, 0, :, :, :], images[:, 2, :, :, :]], dim=1))
            flow_features = flow_features.squeeze((2, 3))
            assert flow_features.shape == (images.shape[0], 1024), flow_features.shape
# %%
