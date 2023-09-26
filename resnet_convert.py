#%%
import torch
import torchvision
from torchvision import transforms
from resnet import resnet50

tv_model = resnet50()

# %%
weights = torch.load('checkpoints/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth')

tv_weights = tv_model.state_dict()
# %%
replacements = {'downsample.conv': 'downsample.0', 'downsample.bn': 'downsample.1', 
                'conv1.conv': 'conv1', 'conv1.bn': 'bn1',
                'conv2.conv': 'conv2', 'conv2.bn': 'bn2',
                'conv3.conv': 'conv3', 'conv3.bn': 'bn3',}
for k,v in weights['state_dict'].items():
    if 'backbone' in k:
        #print without backbone name
        name = k.split('backbone.')[1]
        original_name = name
        for rep_key in replacements.keys():
            original_name = original_name.replace(rep_key, replacements[rep_key])
        assert tv_weights[original_name].shape == v.shape, f"Shape mismatch: {original_name} {tv_weights[original_name].shape} {v.shape}"
        if original_name in tv_weights.keys():
            tv_weights[original_name].copy_(v)
        else:
            print(f"Key {original_name} not found in tv_weights")
        # weights have conv{i}.conv and conv{i}.bn but tv_weights has conv{i} and bn{i}
        # we use a ConvModule to wrap conv+bn+relu layers, thus the name mapping is needed        
        #model.state_dict()[k].copy_(v)
torch.save(tv_model.state_dict(), 'checkpoints/resnet50.pth')