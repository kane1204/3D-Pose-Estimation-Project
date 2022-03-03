import numpy as np
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import gc
class Train_LiftNetwork():
    def __init__(self,model,optimiser,accurate_dist,train_dataloader,valid_dataloader,reducedkey=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))   
        self.model = model.to(self.device)
        
        self.optimiser = optimiser
        self.accz_dists = accurate_dist
        self.train_ds = train_dataloader
        self.valid_ds = valid_dataloader
        self.reduce = reducedkey
        self.reducedKeypoints = [0,2,3,4,6 , 7,10,13 , 14,17,20, 21,24,27, 28,31,34, 35,38,41, 42,45,48, 52,53,55, 58,59,61]

    def train_step(self):
        size = len(self.train_ds.dataset)
        correct = 0
        losses = 0

        self.model.train()
        for data in tqdm(self.train_ds, desc="Training Step"):
            X = data['key_points_2D']
            y = data['key_points_3D'][:,:,2]
            mask = data['visibility'].to(self.device)
            
            X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
            
            X_var = torch.autograd.Variable(X)
            y_var = torch.autograd.Variable(y)
            # y_var = torch.flatten(y_var, start_dim=1, end_dim=2)

            pred = self.model(torch.flatten(X_var, start_dim=1, end_dim=2))

            # For 3 Dims 
            # mask = torch.flatten(torch.stack([mask,mask,mask],dim=1),start_dim=1, end_dim= 2)

            loss = torch.mean(((pred - y_var)*mask)**2)
            # correct += (abs(pred - y_var)<torch.flatten(self.accz_dists.to(self.device).T)).type(torch.float).sum().item()

            # Backpropagation
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            
            losses += loss.item()

        train_acc = (correct / size)*100
        train_loss = losses/size
        return train_acc, train_loss

    def valid_step(self):
        size = len(self.valid_ds.dataset)
        num_batches = len(self.valid_ds)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in tqdm(self.valid_ds, desc="Validation Step"):
                X = data['key_points_2D']
                y = data['key_points_3D'][:,:,2]
                mask = data['visibility'].to(self.device)

                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                
                X_var = torch.autograd.Variable(X)
                y_var = torch.autograd.Variable(y)
                # y_var = torch.flatten(y_var, start_dim=1, end_dim=2)

                pred = self.model(torch.flatten(X_var, start_dim=1, end_dim=2))
                
                # For 3 Dims 
                # mask = torch.flatten(torch.stack([mask,mask,mask],dim=1),start_dim=1, end_dim= 2)

                loss = torch.mean(((pred - y_var)*mask)**2)
                test_loss += loss.item()
                # correct += (abs(pred - y_var)<torch.flatten(self.accz_dists.to(self.device).T)).type(torch.float).sum().item()
        test_loss /= num_batches

        val_acc = (correct / size)*100
        val_loss = test_loss
        # print(f"Validation Error: \n Accuracy: {val_acc:>4f}%, Avg loss: {val_loss:>8f} \n")
        return val_acc, val_loss
        
    def run(self, epochs):
        for t in tqdm(range(1, epochs+1), desc="CPM Model Epochs"):
            print(f'Epoch {t+0:03}:')
            train_acc, train_loss = self.train_step()
            val_acc, val_loss = self.valid_step()
            print(f'Finished Epoch {t+0:03}: | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Train Acc: {train_acc:.3f}| Val Acc: {val_acc:.3f}')
        print("Done!")
        return self.model

