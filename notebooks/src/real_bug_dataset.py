from copy import deepcopy
import os
from tkinter import S
import cv2
import numpy as np
import pandas as pd
from skimage import io
import torch
import json
from torch.utils.data import Dataset
import re

class RealBugDataset(Dataset):
    """Bug dataset."""

    def __init__(self, df, root_dir, std, mean, transform=None):
        """
        Args:
            hdf_file (string): Path to the hdf file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.xy = 640
        self.transform_x_y = 256
        self.center = 3
        self.stride = 2
        self.sigma = 2
        self.std_2d = std
        self.means_2d = mean
        self.heatmap_size = [64,64]

        #                   Body    r_1      r_2        r_3       l_1       l_2        l_3       a_r        a_l
        self.reduced_kp = [0,2,3,6, 7,10,13, 14,17,20 , 21,24,27, 28,31,34, 35,38,41, 42,45,48,  52,54,55, 58,60,61]
        
        # Remove all the datapoints that doesnt have center keypoint visible. 
        new_df = pd.DataFrame()
        for col, x in df.iterrows():
            vis = x['visibility'].strip('][').split(', ')
            x['visibility'] = list(map(int,vis))
            temp = re.findall(r'\d+', x['key_points_2D'])
            res = list(map(int, temp))
            if len(res) == 56:
                x['key_points_2D'] = np.array(res).reshape((28,2))
                temp = x['key_points_2D'][:,0].copy()
                x['key_points_2D'][:,0] = x['key_points_2D'][:,1]
                x['key_points_2D'][:,1] = temp
                new_df = new_df.append(x)
            
        new_df.reset_index(drop=True, inplace=True)    
        new_df['visibility'] = new_df['visibility'].apply(np.array)
        # Original DF being used
        self.bugs_frame = new_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.bugs_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        img_name = os.path.join(self.root_dir,
                                self.bugs_frame.iloc[idx, 0])
        # Reads Image
        image = io.imread(img_name)
        w, h, _ = image.shape
        df_columns = self.bugs_frame.columns.values.tolist()
        
        # Add the rest of the attributes
        for x in range(len(df_columns)):    
            sample[df_columns[x]] = deepcopy(self.bugs_frame.iloc[idx,x])


        # Create image crop and scale to desired dims 152
        # Crop KP to bbox scaled img
        cropped_kp = sample['key_points_2D']
        # Crop to bbox
        cropped  = image
        # Scale 2D Keypoints to target
        x_scale = (self.transform_x_y/cropped.shape[1])
        y_scale = (self.transform_x_y/cropped.shape[0])

        cropped_kp[:,0] = cropped_kp[:,0]*x_scale
        cropped_kp[:,1] = cropped_kp[:,1]*y_scale

        # Scales Image to target
        resized_img = cv2.resize(cropped, (self.transform_x_y,self.transform_x_y))
        # Writing new scaled to dict
        sample['key_points_2D'] = cropped_kp
        sample.update({'image': resized_img.astype('uint8')})

        # Create HEATMAP
        sample['heatmap'] = None
     
        # Create Heatmaps for Simple 2d method
        heatmap, heat_weight = self.create_heatmaps_simple(sample['key_points_2D'],sample['visibility'])
        sample['heatmap'] = heatmap
        sample['heat_weight'] = heat_weight
        # Center Map Calculation due to the scaling of the image to self.transform_x_y
        center_x = (self.transform_x_y) / 2
        center_y = (self.transform_x_y) / 2

        # Bounding box is now just a 0,0 and 256,256
        sample["bounding_box"]= np.array([0,0,self.transform_x_y,self.transform_x_y])
        
        if self.transform:
            sample = self.transform(sample)

         # Standardises 2d & 3d Keypoints
        sample['key_points_2D'] = self.normal_2d(sample['key_points_2D'])
        return sample

    def scale_data(self,sample):
        return sample* self.scale_percent / 100
    
    def normal_2d(self, x):
        # np.seterr(invalid='ignore')
        return torch.nan_to_num((x-torch.Tensor(self.means_2d))/torch.Tensor(self.std_2d))

    def crop_scale(self,kp, x_shift, y_shift):
        kp[:,0]=  kp[:,0]-x_shift
        kp[:,1]=  kp[:,1]-y_shift
        return kp


    def create_heatmaps_simple(self, kps, vis):
        '''
        :param kps:  [num_kps, 2]
        :param vis: [num_kps, 2]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_kps = len(kps)
        target_weight = np.ones((num_kps, 1), dtype=np.float32)
        target_weight[:,0] =  vis

        
        target = np.zeros((num_kps,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(num_kps):
            feat_stride = [int(self.transform_x_y/ self.heatmap_size[0]),int(self.transform_x_y/ self.heatmap_size[1])] 
            mu_x = int(kps[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(kps[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sample = deepcopy(sample)
        name = sample['file_name']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image'].transpose((2, 0, 1))
        img_tensor = torch.IntTensor(img)

        # Simple 2d Dict
        dic ={'image': img_tensor, 'heatmap':torch.from_numpy(sample['heatmap']), 
              'heat_weight':torch.from_numpy(sample['heat_weight']),'file_name':name,
              'key_points_2D':torch.from_numpy(sample['key_points_2D']),
              'visibility':torch.from_numpy(sample['visibility'])}
        
        # print("After Adding to Dict",dic['image'].shape)
        return dic