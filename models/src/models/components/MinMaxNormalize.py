import torch
from torchvision import transforms

class MinMaxNormalize:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        # Assuming input tensor is a PyTorch tensor
        return (tensor - self.min_val) / (self.max_val - self.min_val)