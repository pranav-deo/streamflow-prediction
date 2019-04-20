import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
import time
import os
import copy

def train(model, dataLoader, nepochs, device, optimiser):
    
    tick = time.time()
    best_loss = 1e10
    best_model_weights = copy.deepcopy(model.state_dict())
    model = model.to(device)
    for epoch in range(nepochs):
        print("Epoch {}/{}".format(epoch,nepochs))
        print("-"*30)

        for phase in ["train", "val"]:
            if phase == 'train':
                scheduler.StepLR(optimiser, nepochs//5)
                model.train()

            else :
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataLoader[phase]:
                inputs = inputs.to(device).to(torch.float)
                labels = labels.to(device).to(torch.float)

                inputs = inputs[np.newaxis,:,:]
                labels = labels[np.newaxis,:,:]

                optimiser.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # print('-'*30)
                    # print(inputs.shape)
                    # print('-'*30)
                    out = model(inputs,10)
                    loss = nn.MSELoss()
                    lloss = loss(out, labels)

                    # Backprop
                    if phase == 'train':
                        lloss.backward()
                        optimiser.step()

                running_loss += lloss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataLoader)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())
        
        print()
    
    tock = time.time()
    time_taken = tock - tick
    print("Training completed in {:.0f}m {:.0f}s".format(time_taken // 60, time_taken % 60))
    print("Lowest val loss {:.4f}".format(best_loss))

    model.load_state_dict(best_model_weights)
    return model
                