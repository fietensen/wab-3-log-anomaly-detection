from transformers import BertModel
from torch.utils.data import DataLoader
from util.online_triplet_loss import OnlineTripleLoss
from util.focal_loss import FocalLoss
from tqdm import tqdm

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


    def train(self, dataloader: DataLoader, epochs: int = 100, lr: float = 0.01, alpha: float = 0.2):
        optimizer = optim.Adam(self.parameters(), lr = lr)

        n_batches = len(dataloader)*2

        for epoch in range(epochs):
            loss_total = 0
            loss_batches = n_batches

            for (batch_iids, batch_ams, batch_lbls) in tqdm(dataloader, total=n_batches):
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

            print("[INFO] Epoch {}/{} | Train Loss: {:.4f} | Val. Loss: {:.4f}".format(
                epoch+1,
                epochs,
                loss_total / loss_batches if loss_batches else torch.nan,
                -1
            ))


    def forward(self, *args, **kwargs):
        output = self.__bert(*args, **kwargs)
        embeddings = output.last_hidden_state[:, 0, :]
        y = self.__classifier(embeddings)
        return embeddings, y


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
