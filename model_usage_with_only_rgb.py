import pickle
import torch
import torch.nn as nn
from collections import deque

with open('processed_data/dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    
"""
Array is multilabel, if action j is performed at frame i, then labels[i][j] = 1

in every data, there are 3 elements: csv_index, filename, labels
"""

# Long-Term and Short-Term Memory
class Memory:
    def __init__(self, long_term_size, short_term_size):
        self.long_term_memory = deque(maxlen=long_term_size)
        self.short_term_memory = deque(maxlen=short_term_size)
        
    def update(self, new_feature):
        self.short_term_memory.append(new_feature)
        if len(self.short_term_memory) == self.short_term_memory.maxlen:
            self.long_term_memory.append(self.short_term_memory[0])
            
# LSTR Encoder-Decoder using Transformer
class LSTREncoderDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes=4):
        super(LSTREncoderDecoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # classifier (4 classes)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, long_term_memory, short_term_memory):
        long_term_encoded = self.encoder(long_term_memory)
        decoder_output = self.decoder(short_term_memory, long_term_encoded)

        # Assuming the last output is what you want to classify
        last_output = decoder_output[-1]
        class_output = self.classifier(last_output)
        return class_output     # this will be of shape (batch_size, num_classes)

only_rgb = True

# init
long_term_size = 2048  # Example size
short_term_size = 32  # Example size
memory = Memory(long_term_size, short_term_size)

d_model = 2048 + 512  if not only_rgb else 2048  # Dimension of the model
nhead = 8  # Number of heads in multi-head attention
num_layers = 3  # Number of transformer layers

model = LSTREncoderDecoder(d_model, nhead, num_layers)

# Simulate feature update
for _ in range(short_term_size):  # iterate over short-term memory size at least it is filled
    rgb_feature = torch.rand(1, 2048)  # RGB features
    optical_flow_feature = torch.rand(1, 512)  # Optical Flow features
    
    # Combine or process features if needed
    combined_feature = torch.cat((rgb_feature, optical_flow_feature), dim=1) if not only_rgb else rgb_feature
    memory.update(combined_feature)

# Forward pass
if len(memory.long_term_memory) > 0 and len(memory.short_term_memory) > 0:
    long_term_memory_tensor = torch.stack(list(memory.long_term_memory))
    short_term_memory_tensor = torch.stack(list(memory.short_term_memory))
    class_output = model(long_term_memory_tensor, short_term_memory_tensor)
    class_probabilities = nn.Softmax(dim=1)(class_output)
else:
    print("Memory is not yet filled. Continue collecting features.")

