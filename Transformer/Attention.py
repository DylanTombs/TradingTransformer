import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from Transformer.Mask import Mask

class FullAttention(nn.Module):
    def __init__(self, maskFlag=True, factor=5, scale=None, attentionDropout=0.1, outputAttention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.maskFlag = maskFlag
        self.outputAttention = outputAttention
        self.dropout = nn.Dropout(attentionDropout)

    def forward(self, queries, keys, values, attnMask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.maskFlag:
            if attnMask is None:
                attnMask = Mask(B, L, device=queries.device)
            scores.masked_fill_(attnMask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.outputAttention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, dModel, nHeads, dKeys=None, dValues=None):
        super(AttentionLayer, self).__init__()

        dKeys = dKeys or (dModel // nHeads)
        dValues = dValues or (dModel // nHeads)

        self.innerAttention = attention
        self.queryProjection = nn.Linear(dModel, dKeys * nHeads)
        self.keyProjection = nn.Linear(dModel, dKeys * nHeads)
        self.valueProjection = nn.Linear(dModel, dValues * nHeads)
        self.outProjection = nn.Linear(dValues * nHeads, dModel)
        self.nHeads = nHeads

    def forward(self, queries, keys, values, attnMask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.nHeads

        queries = self.queryProjection(queries).view(B, L, H, -1)
        keys = self.keyProjection(keys).view(B, S, H, -1)
        values = self.valueProjection(values).view(B, S, H, -1)

        out, attn = self.innerAttention(queries, keys, values, attnMask)
        out = out.view(B, L, -1)

        return self.outProjection(out), attn
