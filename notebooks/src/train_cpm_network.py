import numpy as np
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import gc
class Train_CPM_Network():
    def __init__(self,model,optimiser,accurate_dist,train_dataloader,valid_dataloader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))   
        self.model = model.to(self.device)
        self.imageshape = 152
        self.optimiser = optimiser
        self.accz_dists = accurate_dist
        self.train_ds = train_dataloader
        self.valid_ds = valid_dataloader
        self.criterion = JointsMSELoss(self.device, use_target_weight=False)
    def train_step(self):
        size = len(self.train_ds.dataset)
        correct = 0
        losses = 0
        ########
        #                     8 is stride     62  keypoints 
        heat_weight =  1.0
        self.model.train()
        for data in tqdm(self.train_ds, desc="Training Step"):
            image = data['image'].to(self.device, dtype=torch.float)
            center = data['centermap'].to(self.device, dtype=torch.float)
            heatmap = data['heatmap'].to(self.device, dtype=torch.float)
            vis = data['visibility']
            heat_weight = ((heatmap.shape[1]-1)**2)*8

            input_var = torch.autograd.Variable(image)
            heatmap_var = torch.autograd.Variable(heatmap)
            centermap_var = torch.autograd.Variable(center)

            heat1, heat2, heat3, heat4, heat5, heat6 = self.model(input_var, centermap_var)
            
            loss1 = self.criterion(heat1, heatmap_var, vis) * heat_weight
            loss2 = self.criterion(heat2, heatmap_var, vis) * heat_weight
            loss3 = self.criterion(heat3, heatmap_var, vis) * heat_weight
            loss4 = self.criterion(heat4, heatmap_var, vis) * heat_weight
            loss5 = self.criterion(heat5, heatmap_var, vis) * heat_weight
            loss6 = self.criterion(heat6, heatmap_var, vis) * heat_weight

            # print(loss1.shape, loss2.shape, loss3.shape,  loss4.shape, loss5.shape, loss6.shape)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            

            pred = torch.from_numpy(self.get_kpts(heat6)).to(self.device, dtype=torch.float)
            y = torch.from_numpy(self.get_kpts(heatmap)).to(self.device, dtype=torch.float)

            # correct += (abs(pred - y)<self.accz_dists.to(self.device)).type(torch.float).sum().item()

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
        heat_weight = 1.0
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in tqdm(self.valid_ds, desc="Validation Step"):
                image = data['image'].to(self.device, dtype=torch.float)
                center = data['centermap'].to(self.device, dtype=torch.float)
                heatmap = data['heatmap'].to(self.device, dtype=torch.float)
                vis = data['visibility']
                heat_weight = ((heatmap.shape[1]-1)**2)*8

                input_var = torch.autograd.Variable(image)
                heatmap_var = torch.autograd.Variable(heatmap)
                centermap_var = torch.autograd.Variable(center)

                heat1, heat2, heat3, heat4, heat5, heat6 = self.model(input_var, centermap_var)
                
                loss1 = self.criterion(heat1, heatmap_var, vis) * heat_weight
                loss2 = self.criterion(heat2, heatmap_var, vis) * heat_weight
                loss3 = self.criterion(heat3, heatmap_var, vis) * heat_weight
                loss4 = self.criterion(heat4, heatmap_var, vis) * heat_weight
                loss5 = self.criterion(heat5, heatmap_var, vis) * heat_weight
                loss6 = self.criterion(heat6, heatmap_var, vis) * heat_weight

                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                test_loss += loss.item()

                pred = torch.from_numpy(self.get_kpts(heat6)).to(self.device, dtype=torch.float)
                y = torch.from_numpy(self.get_kpts(heatmap)).to(self.device, dtype=torch.float)
                
                # correct += (abs(pred - y)<self.accz_dists.to(self.device)).type(torch.float).sum().item()

        test_loss /= num_batches
        val_acc = (correct / size)*100
        val_loss = test_loss
        # print(f"Validation Error: \n Accuracy: {val_acc:>4f}%, Avg loss: {val_loss:>8f} \n")
        return val_acc, val_loss
        
        
    def run(self, epochs):
        torch.cuda.empty_cache()

        for t in tqdm(range(1, epochs+1), desc="CPM Model Epochs"):
            print(f'Epoch {t+0:03}:')
            train_acc, train_loss = self.train_step()
            val_acc, val_loss = self.valid_step()
            print(f'Finished Epoch {t+0:03}: | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Train Acc: {train_acc:.3f}| Val Acc: {val_acc:.3f}')
        print("Done!")
        return self.model
    
    def get_kpts(self, maps, img_h = 152.0, img_w = 152.0):
        # maps (1,63,76,76)
        maps = maps.clone().cpu().data.numpy()
        map_6 = maps[0]

        kpts = []
        for m in map_6[1:]:
            h, w = np.unravel_index(m.argmax(), m.shape)
            x = int(w * img_w / m.shape[1])
            y = int(h * img_h / m.shape[0])
            kpts.append([x,y])
        return np.array(kpts)

    def loss_func(self, pred, expect, mask):
        # print(torch.from_numpy(np.array([1])))
        mask =  torch.cat((torch.from_numpy(np.array([[1]]*len(mask))), mask), dim= 1).to(self.device)
        new_mask = torch.zeros(pred.shape,device=self.device)
        for m in range(len(mask)):
            for i in range(len(mask[m])):
                if mask[m][i] == 1:
                    new_mask[m][i] = torch.ones((pred.shape[2],pred.shape[3]))
                elif mask[m][i] == 0:
                    new_mask[m][i] =  torch.zeros((pred.shape[2],pred.shape[3]))
        masked = (pred - expect)*new_mask
        return torch.mean(masked**2)

class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,device, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.device = device
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)
        
        # target_weight =  torch.cat((torch.from_numpy(np.array([[1]]*len(target_weight))), target_weight), dim= 1).to(self.device)
        # new_mask = torch.zeros(output.shape,device=self.device)
        # for m in range(len(target_weight)):
        #     for i in range(len(target_weight[m])):
        #         if target_weight[m][i] == 1:
        #             new_mask[m][i] = torch.ones((output.shape[2],output.shape[3]))
        #         elif target_weight[m][i] == 0:
        #             new_mask[m][i] =  torch.zeros((output.shape[2],output.shape[3]))
        # target_weight = new_mask
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[: idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight