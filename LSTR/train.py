from LSTRmodel import LSTREncoderDecoder, Memory
from custom_data_loader import get_data_loader
from criterions import MultipCrossEntropyLoss

import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

# cuda available check
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    
else:
    device = torch.device("cpu")
    print("No GPU available. Using CPU instead.")

long_term_size = 512
short_term_size = 32

d_model = 2048 + 1024  # Embedding dimension
nhead = 8  # Number of heads in multi-head attention
num_layers = 3  # Number of transformer layers

memory = Memory(long_term_size, short_term_size)
model = LSTREncoderDecoder(d_model, nhead, num_layers, 13).to(device)
train_loader = get_data_loader("/data/common/abb_project/processed_data/total_data.pkl", batch_size=1)

num_epochs = 10
criterion = MultipCrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for i, (flow, rgb, labels) in enumerate(train_loader):
        
        flow = flow.to(device)
        rgb = rgb.to(device)
        labels = labels.to(device)
        
        for frame in range(flow.shape[1]):
            flow_frame = flow[:, frame, :]
            rgb_frame = rgb[:, frame, :]
            labels_frame = labels[:, frame, :]
            
            # print(f"flow_frame shape: {flow_frame.shape}")
            # print(f"rgb_frame shape: {rgb_frame.shape}")
            # print(f"labels_frame shape: {labels_frame.shape}")
            
            # combine features
            combined_feature = torch.cat((rgb_frame, flow_frame), dim=1)
            
            # print(f"combined_feature shape: {combined_feature.shape}")
            
            # in here the combined feature shape is [1, 3072]
            
            # Update Memory
            memory.update(combined_feature)
            
            # Check if memory is filled
            if len(memory.long_term_memory) > 0 and len(memory.short_term_memory) > 0:
                long_term_memory_tensor = torch.stack(list(memory.long_term_memory)).to(device)
                short_term_memory_tensor = torch.stack(list(memory.short_term_memory)).to(device)
                
                # Forward pass
                outputs = model(long_term_memory_tensor, short_term_memory_tensor)
                
                # Compute loss
                loss = criterion(outputs, labels_frame)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (frame+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Frame [{frame+1}/{flow.shape[1]}], Loss: {loss.item():.4f}')
            else:
                print("Memory is not yet filled. Continue collecting features.")

print("Training complete.")