from sklearn.datasets import make_regression
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
torch.manual_seed(0)

def LinearRegression(seed = 0):
    X, y = make_regression(n_samples = 5000, n_features = 1000, n_informative=500, n_targets=1, random_state=seed)
    X_torch = torch.from_numpy(X).to(torch.float32)
    y_torch = torch.from_numpy(y).to(torch.float32)
    dataset = TensorDataset(X_torch, y_torch)
    return DataLoader(dataset, batch_size=128)
