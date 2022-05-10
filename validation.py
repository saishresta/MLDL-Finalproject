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

from DNN_model import convNet



if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



data_path = os.getcwd() + '\\PreProc_folder'

# change mini_batch_size acording to the graphics card memory capability
mini_batch_size = 1
num_spkr = 2
model = convNet(mini_batch_size, num_spkr)

# If pre-trained network needs to be used, set the load_flag to True
load_flag = True

if load_flag:
    file_2b_loaded = 'trained_model'
else:
    file_2b_loaded = []

if load_flag:
    load_PATH = data_path + '\\' + file_2b_loaded + '.pth'    

    checkpoint = torch.load(load_PATH,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Total num_params', pytorch_total_params)

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(m.bias)

if False == load_flag:
    model.apply(weight_init)

tmp_wt_decay = 0
tmp_lr = 5*1e-4

optimizer = optim.Adam(model.parameters(), lr=tmp_lr, weight_decay=tmp_wt_decay)
if True == load_flag:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss_function = torch.nn.BCELoss()


print( 'Load_flag:', load_flag, 'learn_rate: ',tmp_lr, 'weight_decay: ', tmp_wt_decay, 'Loaded: ', file_2b_loaded)





model.eval()
with torch.no_grad():   

    current_Acc = []
    Final_mean_Acc = 0
    Final_med_Acc = 0
            
    test_total = 0
    test_correct = 0

    Te_loss = []
    current_Te_loss = []
    val_set_len = 0
    val_dataset = HI_Dataset(data_path, 'validation.csv')
   

    
    val_set_len += val_dataset.__len__()
    print("Validation set size: ", val_set_len ,"samples")

    valLoader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=mini_batch_size, shuffle=True)
    
    test_total_dataset = 0
    test_correct_dataset = 0
    Te_loss_dataset = []
    
    for test_batch_E, test_batch_A, test_batch_y in valLoader:
        print('#', end='', sep='')

        test_batch_E = test_batch_E.to(device)
        test_batch_A = test_batch_A.to(device)
        test_batch_y = test_batch_y.to(device)

        real_class = torch.argmax(test_batch_y, dim=1)
        net_out = model(test_batch_E.view(-1, 1, 192, 10), test_batch_A.view(-1, 2, 151, 257))
        
        test_loss = loss_function(net_out, test_batch_y)
        Te_loss_dataset.append(test_loss.cpu().data.numpy())
        Te_loss.append(test_loss.cpu().data.numpy())
        predicted_class = torch.argmax(net_out, dim=1)
        
        for i in range(test_batch_y.shape[0]):
            if predicted_class[i] == real_class[i]:
                print("True")    
                test_correct_dataset += 1
                test_correct += 1
            else:
                print("False")   

            test_total_dataset += 1
            test_total += 1

    current_Acc.append(np.round(test_correct_dataset/test_total_dataset, 4)*100)
    current_Te_loss.append(np.mean(Te_loss_dataset))
    del test_correct_dataset, test_total_dataset, i

    Final_mean_Acc = np.mean(current_Acc)
    Final_mean_Te_loss = np.mean(current_Te_loss)

    Final_med_Acc = np.median(current_Acc)
    Final_med_Te_loss = np.median(current_Te_loss)        
    mean_acc = np.round(test_correct/test_total, 4)*100


print(f"Validation Accuracy: {mean_acc} \n" )
