from copy import deepcopy
import os
from tkinter import S
import cv2
import numpy as np
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset


class BugDataset(Dataset):
    """Bug dataset."""

    def __init__(self, df, root_dir, reduced=False,  transform=None):
        """
        Args:
            hdf_file (string): Path to the hdf file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.xy = 640
        self.transform_x_y = 152
        self.center = 3
        self.means_2d = []
        self.means_3d = []
        self.std_2d =  []
        self.std_3d = []
        self.stride = 2
        self.sigma = 3.0
        #                   Body    r_1      r_2        r_3       l_1       l_2        l_3       a_r        a_l
        self.reduced = reduced
        self.reduced_kp = [0,2,3,6, 7,10,13, 14,17,20 , 21,24,27, 28,31,34, 35,38,41, 42,45,48,  52,54,55, 58,60,61]
        

        # Remove all the datapoints that doesnt have center keypoint visible. 
        new_df = pd.DataFrame()
        for col, x in df.iterrows():
            if x['visibility'][self.center-1]==1:
                new_df = new_df.append(x)
                new_df.reset_index(drop=True, inplace=True)
        new_df['key_points_2D'] = new_df['key_points_2D'].apply(np.array)
        new_df['key_points_3D'] = new_df['key_points_3D'].apply(np.array)
        new_df['visibility'] = new_df['visibility'].apply(np.array)
        # For memory space help we remove 
        new_df.drop(['cam_rot', 'cam_trans', 'cam_intrinsics'], axis=1, inplace=True)

        # Fix BBOXS
        new_df["bounding_box"] = new_df['key_points_2D'].apply(self.bbox_fix)

        # Keypoint reduction code
        if self.reduced:
            new_df['key_points_2D'] = new_df['key_points_2D'].apply(self.reduce)
            new_df['key_points_3D'] = new_df['key_points_3D'].apply(self.reduce)
            new_df['visibility'] = new_df['visibility'].apply(self.reduce)          

        
        # self.scale_percent = (self.transform_x_y/self.xy)*100
        # # Second Correct the 2D keypoint
        # new_df['key_points_2D'] = new_df['key_points_2D'].apply(self.scale_data)
        # new_df["bounding_box"] = new_df["bounding_box"].apply(self.scale_data)

        # Centralising dataset around center keypoint center
        new_df['key_points_3D'] = new_df['key_points_3D'].apply(self.centralise_3d)

        # Calculate normals
        self.normalise(new_df)

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
        cropped_kp = self.crop_scale(sample['key_points_2D'],w -(w-sample['bounding_box'][0]),h - (h-sample['bounding_box'][1]))
        # Crop to bbox
        cropped  = image[sample['bounding_box'][1]:sample['bounding_box'][3],sample['bounding_box'][0]:sample['bounding_box'][2]]
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

        # Create HEAT & CENTER MAPS
        sample['heatmap'] = None
        sample['centermap'] = None
        # Create Heatmap guassian for idx
        heat_map = 0
        # keypoints = sample['key_points_2D']
        heat = np.zeros((round(self.transform_x_y/self.stride), round(self.transform_x_y/self.stride), len(sample['key_points_2D']) + 1), dtype=np.float32)
        for i in range(len(sample['key_points_2D'])):
            if sample['visibility'][i] == 1:
                x = int(sample['key_points_2D'][i][0]) * 1.0 / self.stride
                y = int(sample['key_points_2D'][i][1]) * 1.0 / self.stride
                heat_map = self.guassian_kernel(size_h=self.transform_x_y / self.stride, size_w=self.transform_x_y / self.stride, center_x=x, center_y=y, sigma=self.sigma)
                heat_map[heat_map > 1] = 1
                heat_map[heat_map < 0.0099] = 0
                heat[:, :, i + 1] = heat_map

        heat[:, :, 0] = 1.0 - np.max(heat[:, :, 1:], axis=2)
        sample['heatmap'] = heat

        # Center Map Calculation due to the scaling of the image to 152
        center_x = (self.transform_x_y) / 2
        center_y = (self.transform_x_y) / 2
        # Bounding box is now just a 0,0 and 152,152
        sample["bounding_box"]= [0,0,152,152]
        
        sample['centermap'] = np.zeros((self.transform_x_y, self.transform_x_y, 1), dtype=np.float32)
        center_map = self.guassian_kernel(size_h=self.transform_x_y, size_w=self.transform_x_y,  center_x=center_x, center_y=center_y, sigma=self.sigma)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        sample['centermap'][:, :, 0] = center_map

        # Standardises 2d & 3d Keypoints
        sample['key_points_2D'] = self.normal_2d(sample['key_points_2D'])
        sample['key_points_3D'] = self.normal_3d(sample['key_points_3D'])

        if self.transform:
            sample = self.transform(sample)
        return sample
    def scale_data(self,sample):
        return sample* self.scale_percent / 100
    
    def crop_scale(self,kp, x_shift, y_shift):
        kp[:,0]=  kp[:,0]-x_shift
        kp[:,1]=  kp[:,1]-y_shift
        return kp

    def centralise_2d(self, sample):
        x_diff, y_diff = sample[self.center-1][0], sample[self.center-1][1]
        for i in range(len(sample)):
            sample[i][0] = sample[i][0] - x_diff
            sample[i][1] = sample[i][1] - y_diff
        return sample

    def centralise_3d(self, sample):
        x_diff, y_diff, z_diff = sample[self.center-1][0], sample[self.center-1][1], sample[self.center-1][2]
        for i in range(len(sample)):
            sample[i][0] = sample[i][0] - x_diff
            sample[i][1] = sample[i][1] - y_diff
            sample[i][2] = sample[i][2] - z_diff
        return sample

    def normal_2d(self, x):
        np.seterr(invalid='ignore')
        return np.nan_to_num((x-self.means_2d)/self.std_2d)
    def normal_3d(self, x):
        np.seterr(invalid='ignore')
        return np.nan_to_num((x-self.means_3d)/self.std_3d)

    def normalise(self, df):
        if self.reduced:
            kp = 28
        else:
            kp = 62
        array_2d = np.array(df['key_points_2D'].to_numpy())
        array_3d = np.array(df['key_points_3D'].to_numpy())

        for x in range(len(array_2d)):
            array_2d[x] = np.array(array_2d[x])
            array_3d[x] = np.array(array_3d[x])

        fixed_array_2d = np.empty((len(array_2d),kp*2))
        fixed_array_3d = np.empty((len(array_3d),kp*3))
        for x in range(len(fixed_array_2d)):
            z = array_2d[x].reshape(1,kp*2)
            fixed_array_2d[x] = z
        for x in range(len(fixed_array_3d)):
            z = array_3d[x].reshape(1,kp*3)
            fixed_array_3d[x] = z

        for x in range(len(fixed_array_2d[0])):
            self.std_2d.append(np.std(fixed_array_2d[:,x], axis=0))
            self.means_2d.append(np.mean(fixed_array_2d[:,x], axis=0))
        self.means_2d = np.array(self.means_2d).reshape((kp,2))
        self.std_2d = np.array(self.std_2d).reshape((kp,2))

        for x in range(len(fixed_array_3d[0])):
            self.std_3d.append(np.std(fixed_array_3d[:,x], axis=0))
            self.means_3d.append(np.mean(fixed_array_3d[:,x], axis=0))
        self.means_3d = np.array(self.means_3d).reshape((kp,3))
        self.std_3d = np.array(self.std_3d).reshape((kp,3))

        # df['key_points_2D'] = df['key_points_2D'].apply(self.normal_2d)
        # df['key_points_3D'] = df['key_points_3D'].apply(self.normal_3d)
        # return df
        
    def guassian_kernel(self, size_w, size_h, center_x, center_y, sigma):
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)
    
    def bbox_fix(self, keypoints):
        padding = 1
        x_coordinates, y_coordinates = zip(*keypoints)
        x_coordinates = [i for i in x_coordinates if i != 0]
        y_coordinates = [i for i in y_coordinates if i != 0]
        return np.array([int(min(x_coordinates)-padding), int(min(y_coordinates)-padding), int(max(x_coordinates)+padding), int(max(y_coordinates)+padding)])
    
    def reduce(self, sample):
        return sample[self.reduced_kp]

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

        heatmap = sample['heatmap']
        heatmap = heatmap.transpose((2, 0, 1))

        center =  sample['centermap']
        center = center.transpose((2, 0, 1))

        dic ={'image': img_tensor, 'heatmap':torch.from_numpy(heatmap), 
              'centermap':torch.from_numpy(center),'file_name':name,
              'bounding_box':torch.from_numpy(sample['bounding_box']),
              'key_points_2D':torch.from_numpy(sample['key_points_2D']),
              'key_points_3D':torch.from_numpy(sample['key_points_3D']),
              'visibility':torch.from_numpy(sample['visibility'])}
        
        # print("After Adding to Dict",dic['image'].shape)
        return dic