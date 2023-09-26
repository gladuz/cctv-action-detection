# ABB project 

## Dataset parsing
All the dataset preperation related code is in `explore.py`. By running `python explore.py` it will create 2 files in `processed_data` folder. 

- `data_files.csv` is the file paths with the name of the video and xml
- `dataset.pkl` is the pickle dataset with `list(tuple) -> [(index, path, labels)]`. Path expludes the extension (easy to use by adding .xml or .mp4)
```bash
processed_data/
├── data_files.csv
├── dataset.pkl   --> [(index, path, labels),...]
└── resized_480.mp4
```

It needs to be run only ONCE at the start. After that only run if the data changes.

## Preparing Resnet50 backbone weights and converting

Resnet50 implementation is taken from the `TSN` backbone from the [MMAaction2 checkpoint](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth). To download the checkpoint run:
```bash
mkdir checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth
```

To get the pretrained weights, run 
```bash 
python resnet_convert.py
```
It will save the weights to `checkpoints/resnet50.pth`

### Usage
```python
from resnet import resnet50
model = resnet50()
model.eval()
```
## Extracting the features of the frames
For starting `extraction.py` has a code to get first 32 frames with sample rate of 6 and running it through resnet50. It uses the 480p resized sample video `processed_data/resized_480.mp4` for efficiency.