from pathlib import Path
from torch.optim import RMSprop

import numpy as np
import torch.nn.init as init
import torch.nn as nn
import torch

def initialize_weights(m):
    "Sample initial weights from gaussian N(mu=0, sigma=0.05)"
    if isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            init.constant_(m.bias, 0)


class _AutoLogEncoder(nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()

        self._l1_lambda = 10e-5

        self.fc1 = nn.Linear(N, 128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.Tanh()
        self.drop2 = nn.Dropout(0.6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1_loss = 0 # regularization loss

        x = self.fc1(x)
        l1_loss += torch.norm(x, p=1)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        l1_loss += torch.norm(x, p=1)
        x = self.act2(x)
        y = self.drop2(x)
    
        return y, l1_loss * self._l1_lambda


class _AutoLogDecoder(nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()

        self._l1_lambda = 10e-8

        self.fc1 = nn.Linear(64, 128)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(128, N)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor, l1_loss = 0.0) -> torch.Tensor:
        x = self.fc1(x)
        l1_loss += torch.norm(x, p=1)
        x = self.act1(x)

        x = self.fc2(x)
        l1_loss += torch.norm(x, p=1)
        y = self.act2(x)

        return y, l1_loss * self._l1_lambda


class AutoLogAutoencoder(nn.Module):
    def __init__(self, N: int, datasets: dict) -> None:
        super().__init__()
        
        self.__encoder = _AutoLogEncoder(N)
        self.__decoder = _AutoLogDecoder(N)
        self.__datasets = datasets
        self.__threshold = 0

        self.apply(initialize_weights)


    def classify(self, x: torch.Tensor) -> bool:
        "Classifies input as either normative (0) or anomalous (1)"

        self.eval()

        with torch.no_grad():
            y,_ = self(x)
            if y.device.type != "cpu":
                y = y.cpu()
                x = x.cpu()
            val_mse = np.mean(np.power(y.numpy() - x.numpy(), 2), axis=1)
            return val_mse>self.__threshold


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, l1_loss = self.__encoder(x)
        reconstruction, l1_loss = self.__decoder(latent, l1_loss)

        return reconstruction, l1_loss

    def train_batch(self, epochs: int = 50, **kwargs):
        print("[INFO] Starting training for {} epochs".format(epochs))
        optim = RMSprop(self.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        train_batches = len(self.__datasets["train"])

        for epoch in range(epochs):
            self.train()
            train_loss = 0

            for (batch_x,) in self.__datasets["train"]:
                y, l1_loss = self(batch_x)
                loss_re = loss_fn(y, batch_x)
                loss = loss_re + l1_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_loss += loss_re.mean().item()
            
            self.eval()
            with torch.no_grad():
                y, l1_loss = self(self.__datasets["val"])
                val_mse = loss_fn(y, self.__datasets["val"]).mean().item()
                self.__threshold = np.percentile(val_mse, 90)
                
                if kwargs.get("log_train_loss") != None:
                    kwargs.get("log_train_loss").append(train_loss/train_batches)
                if kwargs.get("log_val_loss") != None:
                    kwargs.get("log_val_loss").append(val_mse)

                print(f"[INFO] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/train_batches:.4} | Val Loss: {val_mse:.4} | Threshold: {self.__threshold:.4}")


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
