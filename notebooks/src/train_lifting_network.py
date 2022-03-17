import numpy as np
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import gc
import time

from notebooks.src.eval.accuracies import keypoint_3d_pck
class Train_LiftNetwork():
    def __init__(self,model,optimiser,accurate_dist,train_dataloader,valid_dataloader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))   
        self.model = model.to(self.device)
        
        self.optimiser = optimiser
        self.accz_dists = accurate_dist
        self.train_ds = train_dataloader
        self.valid_ds = valid_dataloader
        
    def train_step(self):
        train_epoch_acc, train_epoch_loss = 0, 0
        batches = 0
        self.model.train()
        for data in tqdm(self.train_ds, desc="Training Step", disable=True):
            X = data['key_points_2D']
            y = data['key_points_3D'] # 3d
            mask = data['visibility'].to(self.device)
            
            X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
            # 3dim to unflatten reshape((batch,limbs,3))
            X_flattened = torch.flatten(X, start_dim=1, end_dim=2)
            y_flattened  = torch.flatten(y, start_dim=1, end_dim=2)
            mask_flattened = torch.flatten(torch.stack([mask,mask,mask],dim=1),start_dim=1, end_dim= 2)

            pred = self.model(X_flattened)

            train_loss = torch.mean(((pred - y_flattened)*mask_flattened)**2)
            train_acc = keypoint_3d_pck(pred.detach().cpu().numpy().reshape((len(pred),28,3)),
                                        y.detach().cpu().numpy(), mask)

            # Backpropagation
            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            batches += 1
            # Stops training after one batch
            # break

        # Normal calculation
            train_acc = train_epoch_acc/batches
            train_loss = train_epoch_loss/batches
            return train_acc, train_loss

    def valid_step(self):
        val_epoch_acc, val_epoch_loss = 0, 0
        batches = 0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.valid_ds, desc="Validation Step", disable=True):
                X = data['key_points_2D']
                y = data['key_points_3D'] # 3d
                mask = data['visibility'].to(self.device)

                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                # 3dim to unflatten reshape((batch,limbs,3))
                X_flattened = torch.flatten(X, start_dim=1, end_dim=2)
                y_flattened  = torch.flatten(y, start_dim=1, end_dim=2)
                mask_flattened = torch.flatten(torch.stack([mask,mask,mask],dim=1),start_dim=1, end_dim= 2)

                pred = self.model(X_flattened)

                val_loss = torch.mean(((pred - y_flattened)*mask_flattened)**2)
                val_acc = keypoint_3d_pck(pred.detach().cpu().numpy().reshape((len(pred),28,3)),
                                        y.detach().cpu().numpy(), mask)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                batches +=1


        val_acc = val_epoch_acc/batches
        val_loss = val_epoch_loss/batches
        # print(f"Validation Error: \n Accuracy: {val_acc:>4f}%, Avg loss: {val_loss:>8f} \n")
        return val_epoch_acc, val_epoch_loss
        
    def run(self, epochs):
        torch.cuda.empty_cache()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_desc = f"lifting_3d_model_{timestr}"
        for t in tqdm(range(1, epochs+1), desc="Lifting Model Epochs"):
            train_acc, train_loss, val_acc, val_loss  = 0,0,0,0

            train_acc, train_loss = self.train_step()
            val_acc, val_loss = self.valid_step()

            # Append Training and validation stats to file
            self.append_file(f"{file_desc}_train_loss", train_loss)
            self.append_file(f"{file_desc}_train_acc", train_acc)
            self.append_file(f"{file_desc}_val_loss", val_loss)
            self.append_file(f"{file_desc}_val_acc", val_acc)
            print(f'Finished Epoch {t+0:03}: | Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.5f} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.5f}')

        print("Done!")
        return self.model

    def append_file(self,filename, data):
        file1 = open(f"../results/{filename}.txt", "a")  # append mode
        file1.write(f"{data}\n")
        file1.close()

