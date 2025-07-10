import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, dModel, maxLen=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(maxLen, dModel).float()
        pe.require_grad = False

        position = torch.arange(0, maxLen).float().unsqueeze(1)
        divTerm = (torch.arange(0, dModel, 2).float() * -(math.log(10000.0) / dModel)).exp()

        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, cIn, dModel):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=cIn, out_channels=dModel,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, dModel):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(3, dModel, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, cIn, dModel, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.valueEmbedding = TokenEmbedding(cIn=cIn, dModel=dModel)
        self.positionEmbedding = PositionalEmbedding(dModel=dModel)
        self.temporalEmbedding = TimeFeatureEmbedding(dModel=dModel)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, xMark):
        x = self.valueEmbedding(x) + self.temporalEmbedding(xMark) + self.positionEmbedding(x)
        return self.dropout(x)

