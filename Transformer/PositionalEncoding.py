import numpy as np

class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model

        # Create a (max_len, d_model) matrix
        position = np.arange(max_len)[:, np.newaxis]             # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)  # even indices
        pe[:, 1::2] = np.cos(position * div_term)  # odd indices

        self.pe = pe  # shape: (max_len, d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        returns: x + positional encoding
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]
