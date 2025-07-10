import torch

class Mask():
    def __init__(self, B, L, device="cpu"):
        maskShape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(maskShape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask