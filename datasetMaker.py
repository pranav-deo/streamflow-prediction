import torch
import pandas as pd
import numpy as np
import torch
import torch.utils.data as utils
from sklearn.utils import shuffle


def makeDataset(csvFile, lookback):
    
    dataFile = pd.read_csv(csvFile)
    _data = dataFile.to_numpy()
    
    lenTrain = int(len(_data)*0.6)
    lenVal = int(len(_data)*0.2)
    
    _trainData = np.ndarray(shape = (lenTrain, lookback))
    _trainLabels = np.ndarray( shape = (lenTrain,1))

    _valData = np.ndarray(shape = (lenVal, lookback))
    _valLabels = np.ndarray( shape = (lenVal,1))
    
    _testData = np.ndarray(shape = (len(_data) - lenTrain - lenVal,  lookback))
    _testLabels = np.ndarray(shape = (len(_data) - lenTrain -lenVal, 1))

    for i in range(lenTrain):
        _trainData[i,:] = _data[i:i+lookback,0]
        _trainLabels[i,:] = _data[i+lookback,0]
    
    for i in range(lenVal):
        _valData[i,:] = _data[i+lenTrain - lookback:i+lenTrain,0]
        _valLabels[i,:] = _data[i+lenTrain,0]

    for i in range(len(_data) - lenTrain - lenVal):
        _testData[i,:] = _data[i+lenTrain + lenVal - lookback:i+lenTrain + lenVal,0]
        _testLabels[i,:] = _data[i+lenTrain + lenVal,0]

    _trainData, _trainLabels = shuffle(_trainData, _trainLabels)
    _valData, _valLabels = shuffle(_valData, _valLabels)
    _testData, _testLabels = shuffle(_testData, _testLabels)

    tensorTrainData = torch.stack([torch.tensor(i) for i in _trainData])
    tensorTrainLabels = torch.stack([torch.tensor(i) for i in _trainLabels])
    tensorValData = torch.stack([torch.tensor(i) for i in _valData])
    tensorValLabels = torch.stack([torch.tensor(i) for i in _valLabels])
    tensorTestData = torch.stack([torch.tensor(i) for i in _testData])
    tensorTestLabels = torch.stack([torch.tensor(i) for i in _testLabels])

    trainDataset = utils.TensorDataset(tensorTrainData, tensorTrainLabels)
    valDataset = utils.TensorDataset(tensorValData, tensorValLabels)
    testDataset = utils.TensorDataset(tensorTestData, tensorTestLabels)

    return trainDataset, valDataset, testDataset