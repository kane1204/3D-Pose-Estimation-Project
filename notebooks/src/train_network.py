import numpy as np
import torch
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
        size = len(self.train_ds .dataset)
        self.model.train()
        for batch, data in enumerate(self.train_ds):
            if self.reduce:
                X = data['key_points_2D'][:,self.reducedKeypoints]
                y = data['key_points_3D'][:,self.reducedKeypoints,2]
                mask = data['visibility'][:,self.reducedKeypoints].to(self.device)
            else:
                X = data['key_points_2D']
                y = data['key_points_3D'][:,:,2]
                mask = data['visibility'].to(self.device)

            X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)

            pred = self.model(X)
            loss = torch.mean(((pred - y)*mask)**2)
            # Backpropagation
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def valid_step(self):
        size = len(self.valid_ds.dataset)
        num_batches = len(self.valid_ds)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in self.valid_ds:
                if self.reduce:
                    X = data['key_points_2D'][:,self.reducedKeypoints]
                    y = data['key_points_3D'][:,self.reducedKeypoints,2]
                    mask = data['visibility'][:,self.reducedKeypoints].to(self.device)
                else:
                    X = data['key_points_2D']
                    y = data['key_points_3D'][:,:,2]
                    mask = data['visibility'].to(self.device)

                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                
                pred = self.model(X)
                loss = torch.mean(((pred - y)*mask)**2)
                test_loss += loss.item()
                if self.reduce:
                    correct += (abs(pred - y)<self.accz_dists[self.reducedKeypoints].to(self.device)).type(torch.float).sum().item()
                else:
                    correct += (abs(pred - y)<self.accz_dists.to(self.device)).type(torch.float).sum().item()

        test_loss /= num_batches
        print(f"Validation Error: \n Accuracy: {(correct / size):>4f}%, Avg loss: {test_loss:>8f} \n")
        
    def run(self, epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_step()
            self.valid_step()
        print("Done!")
        return self.model

class Train_CPM_Network():
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
        size = len(self.train_ds .dataset)
        heat_weight = 46 * 46 * 15 / 1.0
        self.model.train()
        for batch, data in enumerate(self.train_ds):
            if self.reduce:
                image = data['image'][:,self.reducedKeypoints]
                center = data['centermap'][:,self.reducedKeypoints]
                heatmap = data['heatmap'][:,self.reducedKeypoints]
                mask = data['visibility'][:,self.reducedKeypoints].to(self.device)
            else:
                image = data['image']
                center = data['centermap']
                heatmap = data['heatmap']
                mask = data['visibility'].to(self.device)

            image, center, heatmap = image.to(self.device, dtype=torch.float), center.to(self.device, dtype=torch.float), heatmap.to(self.device, dtype=torch.float)

            heat1, heat2, heat3, heat4, heat5, heat6 =  self.model(image, center)
            
            loss1 = self.loss_func(heat1, heatmap, mask) * heat_weight
            loss2 = self.loss_func(heat2, heatmap, mask) * heat_weight
            loss3 = self.loss_func(heat3, heatmap, mask) * heat_weight
            loss4 = self.loss_func(heat4, heatmap, mask) * heat_weight
            loss5 = self.loss_func(heat5, heatmap, mask) * heat_weight
            loss6 = self.loss_func(heat6, heatmap, mask) * heat_weight

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            # Backpropagation
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            print(batch)
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # def valid_step(self):
    #     size = len(self.valid_ds.dataset)
    #     num_batches = len(self.valid_ds)
    #     self.model.eval()
    #     test_loss, correct = 0, 0
    #     with torch.no_grad():
    #         for data in self.valid_ds:
    #             if self.reduce:
    #                 X = data['key_points_2D'][:,self.reducedKeypoints]
    #                 y = data['key_points_3D'][:,self.reducedKeypoints,2]
    #                 mask = data['visibility'][:,self.reducedKeypoints].to(self.device)
    #             else:
    #                 X = data['key_points_2D']
    #                 y = data['key_points_3D'][:,:,2]
    #                 mask = data['visibility'].to(self.device)

    #             X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                
    #             pred = self.model(X)
    #             loss = torch.mean(((pred - y)*mask)**2)
    #             test_loss += loss.item()
    #             if self.reduce:
    #                 correct += (abs(pred - y)<self.accz_dists[self.reducedKeypoints].to(self.device)).type(torch.float).sum().item()
    #             else:
    #                 correct += (abs(pred - y)<self.accz_dists.to(self.device)).type(torch.float).sum().item()

    #     test_loss /= num_batches
    #     print(f"Validation Error: \n Accuracy: {(correct / size):>4f}%, Avg loss: {test_loss:>8f} \n")
        
    def run(self, epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_step()
            self.valid_step()
        print("Done!")
        return self.model

    def loss_func(self, pred, expect, mask):
        return torch.mean(((pred - expect)*mask)**2)
    