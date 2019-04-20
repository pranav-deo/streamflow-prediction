import torch
import torch.nn as nn
import torch.autograd as Variable
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam
import os
import time

from datasetMaker import makeDataset
from new_model import Net
from train import train

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# print(model)

nepochs = 100
lookback = 1000
batchSize = 100
PATH = os.path.join(os.getcwd(),'models','{}.h5'.format(time.time()))
dataFileName = 'dv.csv'

model = Net(lookback)

trainDataset, valDataset, testDataset = makeDataset(dataFileName, lookback)

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle = True)
valLoader = torch.utils.data.DataLoader(valDataset, batch_size = batchSize, shuffle = True)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size = batchSize, shuffle = True)

dataLoader = { 'train': trainLoader, 'val': valLoader, 'test': testLoader }

optimiser = Adam(params = model.parameters(), lr = 10)

best_model = train(model, dataLoader, nepochs, device, optimiser)

torch.save(best_model, PATH )