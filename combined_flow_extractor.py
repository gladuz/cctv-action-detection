#%%
import torch
import torch.nn as nn
from bn_inception import BNInception
from flownet import FastFlowNet
import time
from tqdm import trange

class CombinedFlowModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.flow_extractor = FastFlowNet()
        self.bn_inception = BNInception(2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):        
        x = self.flow_extractor(x) # (B, 2, height, width)
        #Should be (x,y,x,y,x,y,x,y,x,y,x,y)
        #x = x.view(-1, 2, x.shape[2], x.shape[3]) # (B, 2, height, width)
        x = self.bn_inception(x)
        x = self.avg_pool(x)
        return x


#%%
if __name__ == '__main__':
    comb_model = CombinedFlowModel().cuda().eval()
    # input is stacked pair of frames (N-1, 3*2, H, W)
    # N-1 acts as the batch dimension for flow extractor
    input_t = torch.randn(1, 6, 384, 512).cuda()
    num_passes = 5
    print(f"Running {num_passes} passes of forward pass")
    start = time.time()
    with torch.no_grad():
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
