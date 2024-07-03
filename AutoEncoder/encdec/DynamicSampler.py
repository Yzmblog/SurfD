
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, BatchSampler
from typing import Iterator, Sequence
from torch.utils.data import Dataset, DataLoader
import random


class DummyDataset(Dataset):
    def __init__(self):
        self.training_idxes = list(range(1))

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 10000

    def get_training_idxes(self):
        return self.training_idxes

    def update_training_idxes(self, new_idxes):
        self.training_idxes = self.training_idxes + new_idxes

class SequenceSampler(Sampler):
    def __init__(self, training_idxes, N):
        self.training_idxes = training_idxes
        self.N = N

    def __iter__(self) -> Iterator[int]:
        
        yield from iter(list(set(range(self.N)) - set(self.training_idxes)))

    def __len__(self) -> int:
        return self.N - len(self.training_idxes)

    def update_training_idxes(self, new_add_idxes):
        self.training_idxes = self.training_idxes+new_add_idxes

class SequenceSampler_Train(Sampler):
    def __init__(self, training_idxes):
        self.training_idxes = training_idxes


    def __iter__(self) -> Iterator[int]:
        random.shuffle(self.training_idxes)
        yield from iter(self.training_idxes)

    def __len__(self) -> int:
        return len(self.training_idxes)

    def update_training_idxes(self, new_add_idxes):
        self.training_idxes = self.training_idxes+new_add_idxes

class WeightedDynamicSampler(WeightedRandomSampler):

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={num_samples}")
        if not isinstance(replacement, bool):
            raise ValueError(f"replacement should be a boolean value, but got replacement={replacement}")

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             f"weights have shape {tuple(weights_tensor.shape)}")

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, (self.weights > 0).sum().int(), self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return ((self.weights > 0).sum().int())

    def update_weights(self, weights):
        self.weights = weights


class DynamicBatchSampler(BatchSampler):

    def update_weights(self, weights):
        self.sampler.update_weights(weights)

    def update_training_idxes(self, training_idxes):
        self.sampler.update_training_idxes(training_idxes)


if __name__ == '__main__':

    dataset = DummyDataset()

    training_idxes = dataset.get_training_idxes()

    weights = [1 if i in training_idxes else 0 for i in range(len(dataset))]

    sampler = WeightedDynamicSampler(weights, len(dataset))
    batch_sampler = DynamicBatchSampler(sampler = sampler, batch_size=8, drop_last=False)

    loader = DataLoader(dataset, sampler=batch_sampler)

    count = 1
    for epoch in range(10):
        for batch in loader:
            print(batch)
        
        curr_training_idxes = dataset.get_training_idxes().copy()
        total_samples = len(dataset)
        if len(curr_training_idxes) < total_samples:
            ##testing here
            new_add_idxes = [count, count+1]
            dataset.update_training_idxes(new_add_idxes)
            count += 2
            easy_p = 0.5/len(curr_training_idxes)
            hard_p = 0.5 / len(new_add_idxes)
            new_weights = torch.zeros(total_samples)

            
            new_weights.index_fill_(0, torch.LongTensor(curr_training_idxes), easy_p)
            new_weights.index_fill_(0, torch.LongTensor(new_add_idxes), hard_p)

            new_weights[curr_training_idxes] = easy_p
            new_weights[new_add_idxes] = hard_p

            print(curr_training_idxes, new_add_idxes, new_weights[:22])
            batch_sampler.update_weights(new_weights)


