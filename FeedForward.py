import numpy as np

class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.w1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return np.maximum(0, x @ self.w1 + self.b1) @ self.w2 + self.b2
