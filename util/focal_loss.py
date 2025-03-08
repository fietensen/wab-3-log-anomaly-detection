import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.__gamma = torch.tensor(gamma)
    
    def forward(self, prediction: torch.tensor, actual: torch.tensor) -> torch.tensor:
        p_t = torch.where(actual == 1, prediction, 1-prediction)
        return (-torch.pow(1-p_t, self.__gamma) * torch.log(p_t)).mean()

if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
