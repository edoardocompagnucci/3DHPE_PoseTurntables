import torch

class NormalizerJoints2d:
    def __init__(self, img_size=512):
        self.img_size = img_size
        
    def __call__(self, sample):
        sample["joints_2d"] = (sample["joints_2d"] / self.img_size) * 2 - 1
        return sample