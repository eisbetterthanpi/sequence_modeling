# @title pandasDataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class pandasDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        chars = sorted(list(set(y)))
        self.vocab_size = len(chars) #
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.y = self.data_process(y) #
        self.seq_len = min([len(a) for a in X])
        print('seq_len',self.seq_len)

    def data_process(self, data): # str
        # return torch.tensor([self.stoi.get(c) for c in data]) #
        return np.array([self.stoi.get(c) for c in data]) #

    def __len__(self): return len(self.X)
    # def __getitem__(self, idx): return self.X.iloc[idx].to_numpy(), self.y.iloc[idx]
    # def __getitem__(self, idx): return self.X[idx].to_numpy(), self.y[idx]
    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.X[idx])-self.seq_len+1)
        return self.X[idx].to_numpy()[i:i+self.seq_len].astype(float), self.y[idx]

train_data = pandasDataset(X_train, y_train)
test_data = pandasDataset(X_test, y_test)
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

for X, y in train_loader:
    print(X.shape, y.shape)
    break
