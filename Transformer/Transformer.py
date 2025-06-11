from InputEmbedding import InputEmbedding
from PositionalEncoding import PositionalEncoding
from Linear import Linear
from EncoderLayer import EncoderLayer
    
class Transformer:
    def __init__(self, vocab_size: int, d_model: int, N: int, h: int, d_ff: int, dropout: float, max_len: int = 5000):
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, max_len)
        self.encoder_layers = [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        self.output_layer = Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding.forward(x)  # shape: (batch, seq_len, d_model)
        x = self.pe.forward(x)         # add positional encoding

        for layer in self.encoder_layers:
            x = layer.forward(x, mask)

        logits = self.output_layer.forward(x)  # shape: (batch, seq_len, vocab_size)
        return logits
