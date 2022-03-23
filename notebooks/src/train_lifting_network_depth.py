import numpy as np
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import gc
import time
import torch.optim as optim
from src.eval.accuracies import keypoint_depth_pck
class Train_LiftNetwork():
    def __init__(self,model,optimiser,train_dataloader,valid_dataloader, std, means):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))   
        self.model = model.to(self.device)

        self.optimiser = optimiser
        gamma = 0.96
        decay_step = 100000
        self.lr_scheduler =  optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lambda step: gamma ** (step / decay_step))
        self.train_ds = train_dataloader
        self.valid_ds = valid_dataloader
        self.means=means
        self.stds = std

    def train_step(self):
        max_norm =True
        train_epoch_acc, train_epoch_loss = 0, 0
        batches = 0
        self.model.train()
        for data in tqdm(self.train_ds, desc="Training Step", disable=True):
            X = data['key_points_2D']
            y = data['key_points_3D'][:,:,2]
            mask = data['visibility'].to(self.device)
            X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
            # 3dim to unflatten reshape((batch,limbs,3))
            X_flattened = torch.flatten(X, start_dim=1, end_dim=2)
            
            pred = self.model(X_flattened)

            train_loss = torch.mean(((pred - y)*mask)**2) # 
            train_acc = keypoint_depth_pck(pred.detach().cpu().numpy(),
                                        y.detach().cpu().numpy(),
                                        mask.detach().cpu().numpy(),
                                        self.stds[:,2], self.means[:,2])
            # Backpropagation
            self.optimiser.zero_grad()
            train_loss.backward()
            if max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimiser.step()
            self.lr_scheduler.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc
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
                y = data['key_points_3D'][:,:,2] 
                mask = data['visibility'].to(self.device)

                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                # 3dim to unflatten reshape((batch,limbs,3))
                X_flattened = torch.flatten(X, start_dim=1, end_dim=2)

                pred = self.model(X_flattened)

                val_loss = torch.mean(((pred - y)*mask)**2) # 
                val_acc = keypoint_depth_pck(pred.detach().cpu().numpy(),
                                          y.detach().cpu().numpy(), 
                                          mask.detach().cpu().numpy(),
                                          self.stds[:,2], self.means[:,2])

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc
                batches +=1
                # break

        val_acc = val_epoch_acc/batches
        val_loss = val_epoch_loss/batches
        # print(f"Validation Error: \n Accuracy: {val_acc:>4f}%, Avg loss: {val_loss:>8f} \n")
        return val_acc, val_loss
        
    def run(self, epochs):
        save = True
        torch.cuda.empty_cache()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_desc = f"lifting_depth_model_{timestr}"
        for t in tqdm(range(1, epochs+1), desc="Lifting Model Epochs"):
            train_acc, train_loss, val_acc, val_loss  = 0,0,0,0

            train_acc, train_loss = self.train_step()
            val_acc, val_loss = self.valid_step()
            if save:
                # Append Training and validation stats to file
                self.append_file(f"{file_desc}_train_loss", train_loss)
                self.append_file(f"{file_desc}_train_acc", train_acc)
                self.append_file(f"{file_desc}_val_loss", val_loss)
                self.append_file(f"{file_desc}_val_acc", val_acc)
                path = f"../models/lifting_depth/{file_desc}_{t}"
                torch.save(self.model.state_dict(), path)
            print(f'Finished Epoch {t+0:03}: | Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.5f} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.5f}')

        print("Done!")
        return self.model

    def append_file(self,filename, data):
        file1 = open(f"../results/{filename}.txt", "a")  # append mode
        file1.write(f"{data}\n")
        file1.close()

