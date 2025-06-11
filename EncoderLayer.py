from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward
from LayerNorm import LayerNorm
import numpy as np

class EncoderLayer:
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        self.self_attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout

    def forward(self, x, mask=None):
        # Self-attention + residual connection
        attn_output = self.self_attn.forward(x, x, x, mask)
        attn_output = dropout(attn_output, self.dropout_rate)
        x = self.norm1.forward(x + attn_output)

        # Feed-forward network + residual connection
        ffn_output = self.ffn.forward(x)
        ffn_output = dropout(ffn_output, self.dropout_rate)
        x = self.norm2.forward(x + ffn_output)

        return x
    
def dropout(x, rate):
    if rate == 0:
        return x
    mask = (np.random.rand(*x.shape) > rate).astype(np.float32)
    return x * mask / (1.0 - rate)

