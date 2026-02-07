# model/utils/sampler.py

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict


class TripletSampler(Sampler):
    """
    Sampler for Triplet Loss.
    Generates batches of size P x M, where:
        P = number of identities
        M = number of samples per identity
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: DataSet object
            batch_size: tuple (P, M)
        """
        self.dataset = dataset
        self.P, self.M = batch_size

        # Build index dictionary: label -> list of indices
        self.index_dict = defaultdict(list)
        for idx, label in enumerate(dataset.label):
            self.index_dict[label].append(idx)

        self.labels = list(self.index_dict.keys())

        # Estimate number of batches per epoch
        self.length = sum(
            len(v) // self.M for v in self.index_dict.values()
        ) // self.P

    def __iter__(self):
        """
        Yield a batch of indices each time.
        """
        labels = self.labels.copy()
        np.random.shuffle(labels)

        batch = []

        for label in labels:
            indices = self.index_dict[label]
            if len(indices) < self.M:
                continue

            selected = np.random.choice(indices, self.M, replace=False)
            batch.extend(selected)

            if len(batch) == self.P * self.M:
                yield batch
                batch = []

    def __len__(self):
        return self.length
