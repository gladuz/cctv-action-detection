import torch
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def my_collate_fn(batch):
    flow_features = [item[0] for item in batch]
    rgb_features = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # print("---before padding---")
    
    # print(f"shape of flow_features: {flow_features[0].shape}")
    # print(f"shape of rgb_features: {rgb_features[0].shape}")
    # print(f"shape of labels: {labels[0].shape}")
    
    flow_features_padded = pad_sequence(flow_features, batch_first=True, padding_value=0)
    rgb_features_padded = pad_sequence(rgb_features, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    # print("---after padding---")
    
    # print(f"shape of flow_features: {flow_features_padded.shape}")
    # print(f"shape of rgb_features: {rgb_features_padded.shape}")
    # print(f"shape of labels: {labels_padded.shape}")
    
    return flow_features_padded, rgb_features_padded, labels_padded

class CustomDataset(Dataset):
    def __init__(self, total_data_dict):
        self.data_dict = total_data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        video_name = list(self.data_dict.keys())[idx]
        flow_features = self.data_dict[video_name][0]
        rgb_features = self.data_dict[video_name][1]
        labels = self.data_dict[video_name][2]
        
        return torch.tensor(flow_features), torch.tensor(rgb_features), torch.tensor(labels)
    
def get_data_loader(total_data_pkl: str, batch_size: int = 4):
    total_data_dict = pickle.load(open(total_data_pkl, 'rb'))
    dataset = CustomDataset(total_data_dict)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    
    return data_loader
    
    
    