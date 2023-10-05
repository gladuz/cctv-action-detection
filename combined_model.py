#%%
import torch
import torch.nn as nn
from bn_inception import BNInception
from flownet import FastFlowNet
import time
from tqdm import trange

class CombinedFlowModel(nn.Module):

    def __init__(self, num_frames) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.flow_extractor = FastFlowNet()
        self.bn_inception = BNInception((num_frames-1) * 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    # x is a tensor of shape (num_frames, 3, height, width)
    # where num_frames is every N frames
    def forward(self, x):        
        x = self.flow_extractor(x) # (num_frames-1, 2, height, width)
        #Should be (x,y,x,y,x,y,x,y,x,y,x,y)
        x = x.view(-1, x.shape[2], x.shape[3]) # (2*(num_frames-1), height, width)
        #Add batch dimension
        x = x.unsqueeze(0)
        x = self.bn_inception(x)
        x = self.avg_pool(x)
        return x


#%%
if __name__ == '__main__':
    comb_model = CombinedFlowModel(num_frames=6).cuda().eval()
    # input is stacked pair of frames (N-1, 3*2, H, W)
    # N-1 acts as the batch dimension for flow extractor
    input_t = torch.randn(5, 6, 384, 512).cuda()
    num_passes = 100
    print(f"Running {num_passes} passes of forward pass")
    start = time.time()
    for x in trange(num_passes):
        output_t = comb_model(input_t) 
    end = time.time()
    print(f'Time elapsed: {end-start:.3f}s for {num_passes} passes, Each forward pass took: {(end-start)/num_passes*1000:.3f}ms')
    out_t = comb_model(input_t)
    print(out_t.shape)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = comb_model.train()
    print('Number of parameters: {:.2f} M'.format(count_parameters(model) / 1e6))
# %%
