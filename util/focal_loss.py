import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.__gamma = gamma
    
    def forward(self, probs: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        p_t = torch.where(actual == 1, probs, 1-probs)
        p_t = torch.clamp(p_t, min=eps, max=1.0)
        return (-torch.pow(1-p_t, self.__gamma) * torch.log(p_t)).mean()

if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
