"""
Credits to @rjnclarke on medium for their Medium article on
Fine-Tuning Embedding Models using Triplet Margin Loss.

https://medium.com/@rjnclarke/fine-tune-an-embedding-model-with-triplet-margin-loss-in-pytorch-62bf00865a6c
"""

from torch.utils.data import Dataset, SequentialSampler
from collections import defaultdict

import numpy as np
import torch


class TripletDataset(Dataset):
    def __init__(self, tokens, attention_masks, labels):
        self.tokens = tokens
        self.attention_masks = attention_masks
        self.labels = torch.Tensor(labels)
        self.label_dict = defaultdict(list)

        for i in range(len(tokens)):
            self.label_dict[int(self.labels[i])].append(i)
        self.unique_classes = list(self.label_dict.keys())

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        ids = self.tokens[index]
        ams = self.attention_masks[index]
        y = self.labels[index]
        return ids, ams, y


class TripletBatchSampler(SequentialSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.unique_classes = dataset.unique_classes
        self.label_dict = dataset.label_dict
        self.num_batches = len(self.dataset) // self.batch_size
        self.class_size = self.batch_size // 4

    def __iter__(self):
        total_samples_used = 0
        weights = np.repeat(1, len(self.unique_classes))

        while total_samples_used < len(self.dataset):
            batch = []
            classes = []
            for _ in range(2):
                next_selected_class = self._select_class(weights)
                while next_selected_class in classes:
                  next_selected_class = self._select_class(weights)
                weights[next_selected_class] += 1
                classes.append(next_selected_class)
                new_choices = self.label_dict[next_selected_class]
                remaining_samples = list(np.random.choice(new_choices, min(self.class_size, len(new_choices)), replace=False))
                batch.extend(remaining_samples)

            total_samples_used += len(batch)

            yield batch

    def _select_class(self, weights):
        dist = 1/weights
        dist = dist/np.sum(dist)
        selected = int(np.random.choice(self.unique_classes, p=dist))
        return selected

    def __len__(self):
        return self.num_batches