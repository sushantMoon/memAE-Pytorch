# Author : Sushant Moon
# Script Objective : 
# 1. Give overview on how to use the model
# 2. Basic Demo with dummy data


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from memAE import VectorDataset, AutoEncoderFCMem, MemAELoss
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
import shutil
from tqdm import tqdm


# Dummy Data Creation
r = lambda x, i: [x for _ in range(i)]
X_train = np.array([ r(random.randint(1,3), 2) for _ in range(1000)])
X_valid = np.array([r(random.randint(1,3), 2) for _ in range(10)])
X_test = np.array([r(random.randint(1,3), 2) for _ in range(10)])

print(f'Preview of the dummy data')
print(X_valid)
print(X_train.shape)
print('\n\n')

X_train_t, X_valid_t, X_test_t = map(
    torch.tensor, (X_train, X_valid, X_test)
)


# Hyper-Parameter
batch_size = 1024
version = '1'
in_col_dim = 2 # number of feature columns in the data
mem_dim = 1024
shrink_thres = 0.00025
learning_rate = 0.9
regularization_parameter = 0.0001
epochs = 5


PATH = f'./memAE/checkpoint/{version}/'

# ### when retraining, clear the version folder
if Path(PATH).exists():
    shutil.rmtree(Path(PATH))
    
Path(PATH+'graphs').mkdir(parents=True, exist_ok=True)

# ## Init Model, loss and optimizer 
model = AutoEncoderFCMem(in_col_dim=in_col_dim, mem_dim=mem_dim, shrink_thres=shrink_thres)
memae_loss = MemAELoss(regularization_parameter=regularization_parameter)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

avg_train_loss = []
avg_val_loss = []
val_loss = []
train_loss = []

for epoch in range(epochs):
    
    # Training Part
    model.train()
    batch_train_loss = []
    for xb, yb in tqdm(train_dl, total=int(len(train_dl))):
        optimizer.zero_grad()
        pred = model(xb.float())
        loss = memae_loss(prediction=pred, ground_truth=yb, training=True)
        batch_train_loss.append(loss)
        train_loss.append(loss)
        loss.backward()
        optimizer.step()
        
    # Evaluation Part
    model.eval()
    batch_valid_loss = []
    with torch.no_grad():
        for xb, yb in tqdm(valid_dl, total=int(len(valid_dl))):
            pred = model(xb.float())
            loss = memae_loss(prediction=pred, ground_truth=yb, validating=True)
            batch_valid_loss.append(loss)
            val_loss.append(loss)
            
    avg_train_loss.append(sum(batch_train_loss)/len(train_dl))
    avg_val_loss.append(sum(batch_valid_loss)/len(valid_dl))
    
    print(f"Epoch : {epoch} ==> Avg. Train Loss : {avg_train_loss[-1]}  & Avg. Validation Loss : {avg_val_loss[-1]}")
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': memae_loss
        }, 
            PATH + f'model_data_epoch_{epoch}_mem_dim_{mem_dim}_shrink_thres_{shrink_thres}_learning_rate_{learning_rate}_regularization_parameter_{regularization_parameter}_batch_size_{batch_size}_trainloss_{avg_train_loss[-1]}_valloss_{avg_val_loss[-1]}_.pt'
    )
    print(f'Model Saved at : {PATH}model_data_epoch_{epoch}_mem_dim_{mem_dim}_shrink_thres_{shrink_thres}_learning_rate_{learning_rate}_regularization_parameter_{regularization_parameter}_batch_size_{batch_size}_trainloss_{avg_train_loss[-1]}_valloss_{avg_val_loss[-1]}_.pt')


# Train loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='red', label='train loss')
# plt.plot(avg_train_loss, color='pink', label='avg train loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig(f'{PATH+"graphs/"}train_loss.png')
plt.show()

# Validation loss plots
plt.figure(figsize=(10, 7))
plt.plot(val_loss, color='orange', label='val loss')
# plt.plot(avg_val_loss, color='yellow', label='avg val loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.savefig(f'{PATH+"graphs/"}val_loss.png')
plt.show()


print("Proceeding for Validation")


test_ds = VectorDataset(X_test_t, X_test_t)
test_dl = DataLoader(test_ds, batch_size=batch_size)

model.eval()
with torch.no_grad():
    batch_test_loss = []
    predictions = []
    for xb, yb in tqdm(test_dl, total=int(len(test_dl))):
        pred = model(xb.float())
        loss = memae_loss(prediction=pred, ground_truth=yb, testing=True)
        loss = np.mean(loss.numpy(), axis=1)
        batch_test_loss.extend(loss)
        predictions.append(pred)

print(f'Individual Loss for Validation : {batch_test_loss}\n')
print(f'Predictions : {predictions}\n')
print(f'Ground Truth : {X_test}\n')