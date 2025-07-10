import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, dModel, dFf=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        dFf = dFf or 4 * dModel
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=dModel, out_channels=dFf, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dFf, out_channels=dModel, kernel_size=1)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, attnMask=None):
        newX, attn = self.attention(x, x, x, attnMask=attnMask)
        x = x + self.dropout(newX)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attnLayers, convLayers=None, normLayer=None):
        super(Encoder, self).__init__()
        self.attnLayers = nn.ModuleList(attnLayers)
        self.convLayers = nn.ModuleList(convLayers) if convLayers is not None else None
        self.norm = normLayer

    def forward(self, x, attnMask=None):
        attns = []
        if self.convLayers is not None:
            for attnLayer, convLayer in zip(self.attnLayers, self.convLayers):
                x, attn = attnLayer(x, attnMask=attnMask)
                x = convLayer(x)
                attns.append(attn)
            x, attn = self.attnLayers[-1](x)
            attns.append(attn)
        else:
            for attnLayer in self.attnLayers:
                x, attn = attnLayer(x, attnMask=attnMask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, selfAttention, crossAttention, dModel, dFf=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        dFf = dFf or 4 * dModel
        self.selfAttention = selfAttention
        self.crossAttention = crossAttention
        self.conv1 = nn.Conv1d(in_channels=dModel, out_channels=dFf, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dFf, out_channels=dModel, kernel_size=1)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.norm3 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, cross, xMask=None, crossMask=None):
        x = x + self.dropout(self.selfAttention(x, x, x, attnMask=xMask)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.crossAttention(x, cross, cross, attnMask=crossMask)[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, normLayer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = normLayer
        self.projection = projection

    def forward(self, x, cross, xMask=None, crossMask=None):
        for layer in self.layers:
            x = layer(x, cross, xMask=xMask, crossMask=crossMask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
