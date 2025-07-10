import torch
import torch.nn as nn
from Transformer.EncodingnDecoding import Decoder, DecoderLayer, Encoder, EncoderLayer
from Transformer.Attention import FullAttention, AttentionLayer
from Transformer.Embedding import DataEmbedding

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.predLen = configs.predLen

        # Embedding
        self.encEmbedding = DataEmbedding(configs.encIn, configs.dModel, configs.dropout)
        self.decEmbedding = DataEmbedding(configs.decIn, configs.dModel, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attentionDropout=configs.dropout, outputAttention=True),
                        configs.dModel,
                        configs.nHeads
                    ),
                    configs.dModel,
                    configs.dFf,
                    dropout=configs.dropout,
                ) for _ in range(configs.eLayers)
            ],
            normLayer=torch.nn.LayerNorm(configs.dModel)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attentionDropout=configs.dropout, outputAttention=False),
                        configs.dModel,
                        configs.nHeads
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attentionDropout=configs.dropout, outputAttention=False),
                        configs.dModel,
                        configs.nHeads
                    ),
                    configs.dModel,
                    configs.dFf,
                    dropout=configs.dropout,
                ) for _ in range(configs.dLayers)
            ],
            normLayer=torch.nn.LayerNorm(configs.dModel),
            projection=nn.Linear(configs.dModel, configs.cOut, bias=True)
        )

    def forward(self, xEnc, xMarkEnc, xDec, xMarkDec,
                encSelfMask=None, decSelfMask=None, decEncMask=None):

        encOut = self.encEmbedding(xEnc, xMarkEnc)
        encOut, attns = self.encoder(encOut, attnMask=encSelfMask)

        decOut = self.decEmbedding(xDec, xMarkDec)
        decOut = self.decoder(decOut, encOut, xMask=decSelfMask, crossMask=decEncMask)
        
        return decOut[:, -self.predLen:, :], attns
