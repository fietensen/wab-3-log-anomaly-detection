from pathlib import Path
from torch.optim import RMSprop

# TODO: Temporary, remove later
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
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

        self._l1_lambda = 10e-5

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
    def __init__(self, N: int) -> None:
        super().__init__()
        # TODO: Add seed option (for numpy / pytorch)

        self.__encoder = _AutoLogEncoder(N)
        self.__decoder = _AutoLogDecoder(N)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        "Classifies input as either normative (0) or anomalous (1)"

        self.eval()

        with torch.no_grad():
            y = self(x)
            # TODO: Compute RE and use Threshold to decide
            return torch.tensor((1, 4), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, l1_loss = self.__encoder(x)
        reconstruction, l1_loss = self.__decoder(latent, l1_loss)

        return reconstruction, l1_loss

"""
# TODO: Replace by something that isn't horrible :)
def mk_classifier(dataset_path: Path, n_epochs: int, model_out_path: Path, feature_names: list[str]) -> AutoLogAutoencoder:
    ae = AutoLogAutoencoder(len(feature_names))
    ae.apply(initialize_weights)
    df = pd.read_csv(dataset_path)
    df_norm = df[df["anomalous"] == False]
    df_anom = df[df["anomalous"] == True]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(df_norm[feature_names].to_numpy())

    ds_data = torch.tensor(x_train, dtype=torch.float32)
    dataset = TensorDataset(ds_data)

    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

    optim = RMSprop(ae.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    ae.eval()
    # evaluate
    with torch.no_grad():
        n_iter = len(dataset)
        loss_total = 0
        for (x,) in dataloader:
            y, l1_loss = ae.forward(x)
            loss = l1_loss + loss_fn(y, x)
            loss_total += loss.item()
            
        print("Loss:", loss_total/n_iter)

    for epoch in range(n_epochs):
        ae.train()
        print("Epoch", epoch)

        # paper doesn't seem to batch inputs
        for (x,) in dataloader:
            y, l1_loss = ae.forward(x)
            loss = l1_loss + loss_fn(y, x)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        ae.eval()
        # evaluate
        with torch.no_grad():
            n_iter = len(dataset)
            loss_total = 0
            for (x,) in dataloader:
                y, l1_loss = ae.forward(x)
                loss = l1_loss + loss_fn(y, x)
                loss_total += loss.item()
            
            print("Loss:", loss_total/n_iter)

    return ae
"""

if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
