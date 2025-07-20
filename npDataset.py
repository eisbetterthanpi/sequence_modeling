# @title npDataset
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class npDataset(Dataset):
    def __init__(self, X, y, p=8):
        self.X, self.y = X, y
        chars = sorted(list(set(y)))
        self.vocab_size = len(chars) #
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.y = self.data_process(y) #
        # self.seq_len = min([len(a) for a in X])
        self.seq_len = (min([len(a) for a in X])//p)*p
        # print('seq_len',self.seq_len)

    def data_process(self, data): # str
        # return torch.tensor([self.stoi.get(c) for c in data]) #
        return np.array([self.stoi.get(c) for c in data]) #
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        # # return self.X[idx], self.y[idx]
        # return torch.tensor(self.X[idx]).unsqueeze(-1), self.y[idx]
        i = np.random.randint(0, len(self.X[idx])-self.seq_len+1)
        return torch.tensor(self.X[idx,i:i+self.seq_len]).unsqueeze(-1), self.y[idx]

train_data = npDataset(xtrain, ytrain)
test_data = npDataset(xtest, ytest)
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

for X, y in train_loader:
    print(X.shape, y.shape)
    break
