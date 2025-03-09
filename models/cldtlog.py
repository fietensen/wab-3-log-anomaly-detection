from transformers import BertModel
from torch.utils.data import DataLoader
from util.online_triplet_loss import OnlineTripleLoss
from util.focal_loss import FocalLoss
from tqdm import tqdm

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch

class CLDTLog(nn.Module):
    def __init__(self, bert_model: BertModel, tl_alpha: float = 1.0, fl_gamma: float = 1.0):
        super().__init__()

        self.__bert = bert_model
        self.__classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        self.__triplet_loss = OnlineTripleLoss(margin=tl_alpha)
        self.__val_triplet_loss = OnlineTripleLoss(margin=tl_alpha, sampling_strategy="fixed_sh")

        self.__focal_loss = FocalLoss(gamma=fl_gamma)

        # lock pretrained BERT embedding layer
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False

        # lock first 9 encoding layers
        for layer in bert_model.encoder.layer[:9]:
            for param in layer.parameters():
                param.requires_grad = False

        # probably already enabled but just to be sure
        for layer in bert_model.encoder.layer[10:]:
            for param in layer.parameters():
                param.requires_grad = True


    def train_batch(self, data: dict, epochs: int = 100, lr: float = 0.01, alpha: float = 0.2):
        optimizer = optim.Adam(self.parameters(), lr = lr)

        n_batches = len(data["train"])

        self.train()
        for epoch in range(epochs):
            loss_total = 0
            loss_batches = n_batches

            for (batch_iids, batch_ams, batch_lbls) in tqdm(data["train"], total=n_batches):
                batch_iids_conv = batch_iids.to(torch.int64)

                embeddings, y = self(input_ids=batch_iids_conv, attention_mask=batch_ams)

                triplet_loss, num_triplets = self.__triplet_loss(embeddings, batch_lbls)
                triplet_loss = alpha * triplet_loss
                if num_triplets == 0:
                    triplet_loss = 0
                
                focal_loss = (1 - alpha) * self.__focal_loss(y, batch_lbls.to(torch.float))

                loss = triplet_loss + focal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_total += loss.item()
            
            # calculate validation loss
            val_loss_total = 0
            val_loss_trip = 0
            val_loss_iters = 0
            self.eval()
            with torch.no_grad():
                for (batch_iids, batch_ams, batch_lbls) in data["val"]:
                    batch_iids_conv = batch_iids.to(torch.int64)

                    embeddings, y = self(input_ids=batch_iids_conv, attention_mask=batch_ams)

                    triplet_loss, num_triplets = self.__val_triplet_loss(embeddings, batch_lbls)
                    triplet_loss = alpha * triplet_loss
                    if num_triplets == 0:
                        triplet_loss = 0
                
                    focal_loss = (1 - alpha) * self.__focal_loss(y, batch_lbls.to(torch.float))

                    loss = triplet_loss + focal_loss

                    val_loss_trip += triplet_loss.item() if triplet_loss else 0
                    val_loss_total += focal_loss.item()
                    val_loss_iters += 1

            print("[INFO] Epoch {}/{} | Train Loss: {:.4f} | Val. Loss (TL): {:.4f} | Val. Loss (FL): {:.4f}".format(
                epoch+1,
                epochs,
                loss_total / loss_batches if loss_batches else torch.nan,
                val_loss_trip / val_loss_iters,
                val_loss_total / val_loss_iters
            ))


    def forward(self, *args, **kwargs):
        output = self.__bert(*args, **kwargs)
        embeddings = output.last_hidden_state[:, 0, :]
        y = self.__classifier(embeddings)
        return embeddings, y


    def classify(self, x: tuple[torch.Tensor, torch.Tensor]) -> np.array:
        self.eval()

        with torch.no_grad():
            _,y = self(input_ids=x[0], attention_mask=x[1])
            if y.device.type != "cpu":
                return y.round().cpu().numpy()
            else:
                return y.round().numpy()

if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
