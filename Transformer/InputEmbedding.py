import numpy as np
import math

class InputEmbedding:
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        # Randomly initialize the embedding matrix
        self.embedding_matrix = np.random.randn(vocab_size, d_model) / math.sqrt(d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len) of token indices (integers)
        returns: (batch_size, seq_len, d_model)
        """
        embedded = self.embedding_matrix[x]  # shape: (batch_size, seq_len, d_model)
        return embedded * math.sqrt(self.d_model)
