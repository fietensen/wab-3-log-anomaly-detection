from models.cldtlog import CLDTLog
from transformers import BertModel

import torch

def train_cldtlog(datasets: dict, epochs: int = 50, **kwargs) -> CLDTLog:
    # experimentally determined parameters, not present in the CLDTLog paper
    tl_alpha = 2.0
    fl_gamma = 3.5
    learn_rate = 0.01
    
    # move dataset onto gpu if possible for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert = BertModel.from_pretrained(".\\bert_base_uncased_hf_model").to(device)
    model = CLDTLog(bert, tl_alpha=tl_alpha, fl_gamma=fl_gamma).to(device)

    # Alpha of 0.2 was determined to be the best fit in the CLDTLog paper
    model.train_batch(datasets, epochs=epochs, lr=learn_rate, alpha=0.2, **kwargs)

    return model

if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
