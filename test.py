import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_optimizer import FAdam
from torch.utils.data import TensorDataset, DataLoader
from run_benchmark import run_benchmark
from optimizers.sophia import SophiaG
#from optimizers.ngd import NGD
#from optimizers.adahessian import Adahessian
#from optimizers.lbfgsnew import LBFGSNew
from datasets2.cifar10 import CIFAR10

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
torch.manual_seed(0)

if len(sys.argv) != 8:
    print("python test.py OptimizerClass ModelClass CriterionClass Dataset lr epochs seed")
    exit()
optimizer_str = sys.argv[1]
model_str = sys.argv[2]
criterion_str = sys.argv[3]
ds_str = sys.argv[4]
lr_str = sys.argv[5]
epochs = int(sys.argv[6])
seed = int(sys.argv[7])
ModelClass = eval(model_str)
OptimizerClass = eval(optimizer_str)
CriterionClass = eval(criterion_str)
lr = float(lr_str)
DSLoader = eval(ds_str)
ds = DSLoader()
ds.label = ds_str

model = ModelClass()
model.label = model_str
model.to(device)

run_benchmark(model, OptimizerClass, CriterionClass, ds, epochs, seed, lr)
