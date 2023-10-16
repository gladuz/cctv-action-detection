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

MEAN=[123.675/255, 116.28/255, 103.53/255]
STD=[58.395/255, 57.12/255, 57.375/255]
EVERY_N_FRAMES = 6
SHORT_SIDE = 512
class FrameDataset(Dataset):

    def __init__(self, frames_dir) -> None:
        super().__init__()
        self.frames_dir = frames_dir
        self.frame_names = os.listdir(self.frames_dir)
        self.frame_paths = [os.path.join(self.frames_dir, x) for x in self.frame_names]
        self.transform = transforms.Compose([
            transforms.Resize(SHORT_SIDE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.frame_groups = [[i, i + 2, i+5] for i in range(1, len(self.frame_names)-5, EVERY_N_FRAMES)]

    def __len__(self):
        return len(self.frame_groups)
    
    def __getitem__(self, group_idx):
        # 000000.jpg 000002.jpg 000005.jpg
        image_names = [f"{x:06d}.jpg" for x in self.frame_groups[group_idx]]
        images = [Image.open(os.path.join(self.frames_dir, x)) for x in image_names]
        images = [self.transform(x) for x in images]
        images = torch.stack(images)
        return images

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--video_id_start", type=int, default=0)
parser.add_argument("--video_id_end", type=int, default=1908)
args = parser.parse_args()

videos = os.listdir("/data/common/abb_project/frames")
video_paths = [os.path.join("/data/common/abb_project/frames", x) for x in videos]
#dataset = FrameDataset(video_paths[0])

#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
rgb_model = resnet50().to(device)
flow_model = CombinedFlowModel().to(device)
rgb_model.eval()
flow_model.eval()

inputs_rgb = [
    torch_tensorrt.Input(
        shape=(1, 3, 512, 768),
        dtype=torch.float,
    )
]
inputs_flow = [
    torch_tensorrt.Input(
        shape=(1, 6, 512, 768),
        dtype=torch.float,
    )
]
enabled_precisions = {torch.float}  # Run with fp16

rgb_module = torch_tensorrt.compile(
    rgb_model, inputs=inputs_rgb, enabled_precisions=enabled_precisions
)
flow_module = torch_tensorrt.compile(
    flow_model, inputs=inputs_flow, enabled_precisions=enabled_precisions
)
torch.jit.save(rgb_module, "checkpoints/resnet_tensorrt.ts")
torch.jit.save(flow_module, "checkpoints/flownet_tensorrt.ts")



# %%
for i in trange(args.video_id_start, args.video_id_end):
    dataset = FrameDataset(video_paths[i])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    for batch in dataloader:
        batch = batch.to(device)
        st_time = time.time()
        with torch.no_grad():
            rgb_features = rgb_module(batch[:, 1, :, :, :]).squeeze()
            flow_features = flow_module(torch.cat([batch[:, 0, :, :, :], batch[:, 2, :, :, :]], dim=1)).squeeze()
        print(time.time() - st_time)
# %%
