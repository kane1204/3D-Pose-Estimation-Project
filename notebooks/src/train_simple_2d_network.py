import numpy as np
import time
import torch
from tqdm.notebook import tqdm
from src.eval.loss import JointsMSELoss
from src.eval.accuracies import accuracy
import time
import gc
class Train_simple_2d_Network():
    def __init__(self,model,optimiser,train_dataloader,valid_dataloader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))   
        self.model = model.to(self.device)
        self.imageshape = 152
        self.optimiser = optimiser
        self.train_ds = train_dataloader
        self.valid_ds = valid_dataloader
        # self.criterion = self.loss_func
        self.criterion = JointsMSELoss(use_target_weight=True)

    def train_step(self):
        train_epoch_acc, train_epoch_loss = 0, 0
        batches = 0
        self.model.train()
        for data in tqdm(self.train_ds, desc="Training Step", disable=True):
            input = data['image'].to(self.device, dtype=torch.float)
            target_heatmap = data['heatmap'].to(self.device, dtype=torch.float)
            heat_weight = data['heat_weight'].to(self.device, dtype=torch.float)

            pred_heatmap =  self.model(input)
            # print(pred_heatmap.shape)
            train_loss = self.criterion(pred_heatmap, target_heatmap, heat_weight)

            _, train_acc, _, _ = accuracy(pred_heatmap.detach().cpu().numpy(),
                                               target_heatmap.detach().cpu().numpy())
            
            # Backpropagation
            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()
            #  Adds Loss to losses list and acc
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
        self.model.eval()
        batches = 0
        with torch.no_grad():
            for data in tqdm(self.valid_ds, desc="Validation Step", disable=True):
                input = data['image'].to(self.device, dtype=torch.float)
                target_heatmap = data['heatmap'].to(self.device, dtype=torch.float)
                heat_weight = data['heat_weight'].to(self.device, dtype=torch.float)

                pred_heatmap =  self.model(input)

                val_loss = self.criterion(pred_heatmap, target_heatmap, heat_weight)

                _, val_acc, _, _ = accuracy(pred_heatmap.detach().cpu().numpy(),
                                              target_heatmap.detach().cpu().numpy())
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                batches +=1
        val_acc = val_epoch_acc/batches
        val_loss = val_epoch_loss/batches
        # print(f"Validation Error: \n Accuracy: {val_acc:>4f}%, Avg loss: {val_loss:>8f} \n")
        return val_acc, val_loss
        
        
    def run(self, epochs):
        torch.cuda.empty_cache()
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_desc = f"simple_2d_model_{timestr}"
        for t in tqdm(range(1, epochs+1), desc="Simple 2D Model Epochs"):
            
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

