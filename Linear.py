import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        self.bias = np.zeros(out_features)

    def forward(self, x):
        return np.dot(x, self.weight) + self.bias
