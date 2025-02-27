import torch.nn as nn
import torch


class _AutoLogEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement
        return torch.zeros((1, 4), dtype=torch.float32)


class _AutoLogDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement
        return torch.zeros((1, 4), dtype=torch.float32)


class AutoLogAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Add seed option (for numpy / pytorch)

        self.__encoder = _AutoLogEncoder()
        self.__decoder = _AutoLogDecoder()

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        "Classifies input as either normative (0) or anomalous (1)"

        self.eval()

        with torch.no_grad():
            y = self(x)
            # TODO: Compute RE and use Threshold to decide
            return torch.tensor((1, 4), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement
        return torch.zeros((1, 4), dtype=torch.float32)


if __name__ == '__main__':
    print(type(torch.tensor([1, 2, 3, 4], dtype=torch.float32)))
    print("This file is not meant to be run as a script.")
