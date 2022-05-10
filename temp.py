import torch 
import torchvision as tv
import torch.optim as optim
from import_data import HI_Dataset
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import random
import time
import os
#os.system("temp.py")
from DNN_model import convNet


device = torch.device('cpu')
loader=torch.load('C:/Users/sika/Desktop/joint_CNN_LSTM_AAD-main/PreProc_folder/test.csv')
mini_batch_size = 4
num_spkr = 2
model = convNet(mini_batch_size, num_spkr)
model = model.to(device)
def check_accuracy(loader, model):
    

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    print (f"{float(num_correct)/float(num_samples)*100:.2f}")
    #print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    