import torch
import torch.nn as nn
from collections import deque

from custom_data_loader import get_data_loader

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Self Attention Mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed-forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization and Dropout Layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self Attention Layer with Add & Norm
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward Layer with Add & Norm
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # EncoderLayer를 num_layers만큼 반복
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)  # 최종 출력 전에 추가되는 normalization layer

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return self.norm(src)  # 최종적으로 normalization 수행 후 반환
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Self Attention Mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Multi-head Attention Mechanism (for encoder-decoder attention)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed-forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization and Dropout Layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self Attention Layer with Add & Norm
        attn_output, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # Multi-head Attention Layer with encoder-decoder attention, Add & Norm
        attn_output, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        # Feed-forward Layer with Add & Norm
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        # DecoderLayer를 num_layers만큼 반복
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)  # 최종 출력 전에 추가되는 normalization layer

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)  # 각 레이어를 통과하면서 memory(즉, encoder의 출력)를 사용
        return self.norm(tgt)  # 최종적으로 normalization 수행 후 반환

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
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.encoder = TransformerEncoder(d_model=d_model, nhead=8, num_layers=num_layers)#(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.decoder = TransformerDecoder(d_model=d_model, nhead=8, num_layers=num_layers)#(self.decoder_layer, num_layers=num_layers)

        # classifier (4 classes)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, long_term_memory, short_term_memory):
        long_term_encoded = self.encoder(long_term_memory)
        decoder_output = self.decoder(short_term_memory, long_term_encoded)

        # Assuming the last output is what you want to classify
        last_output = decoder_output[-1]
        class_output = self.classifier(last_output)
        return class_output     # this will be of shape (batch_size, num_classes)

if __name__ == "__main__":
    # init
    long_term_size = 2048  # Example size
    short_term_size = 32  # Example size
    memory = Memory(long_term_size, short_term_size)

    d_model = 2048 + 1024  # Embedding dimension
    nhead = 8  # Number of heads in multi-head attention
    num_layers = 3  # Number of transformer layers

    model = LSTREncoderDecoder(d_model, nhead, num_layers, 13)

    # Simulate feature update
    for _ in range(short_term_size):  # iterate over short-term memory size at least it is filled
        rgb_feature = torch.rand(1, 2048)  # RGB features
        optical_flow_feature = torch.rand(1, 1024)  # Optical Flow features

        # Combine or process features if needed
        combined_feature = torch.cat((rgb_feature, optical_flow_feature), dim=1)  # Example
        # print(f"Combined feature shape: {combined_feature.shape}")
        memory.update(combined_feature)

    # Forward pass
    if len(memory.long_term_memory) > 0 and len(memory.short_term_memory) > 0:
        long_term_memory_tensor = torch.stack(list(memory.long_term_memory))
        short_term_memory_tensor = torch.stack(list(memory.short_term_memory))
        class_output = model(long_term_memory_tensor, short_term_memory_tensor)
        class_probabilities = nn.Softmax(dim=1)(class_output)
    else:
        print("Memory is not yet filled. Continue collecting features.")
        
    print(f"Class probabilities: {class_probabilities}")
        
    