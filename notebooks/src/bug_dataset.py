import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset


class BugDataset(Dataset):
    """Bug dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            hdf_file (string): Path to the hdf file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.center = 3
        self.means_2d = []
        self.means_3d = []
        self.std_2d =  []
        self.std_3d = []
        self.stride = 8
        self.sigma = 3.0
        # Remove all the datapoints that doesnt have center keypoint visible. 
        new_df = pd.DataFrame()
        for col, x in df.iterrows():
            if x['visibility'][self.center-1]==1:
                new_df = new_df.append(x)
                new_df.reset_index(drop=True, inplace=True)
        new_df['key_points_2D'] = new_df['key_points_2D'].apply(np.array)
        new_df['key_points_3D'] = new_df['key_points_3D'].apply(np.array)

        # Fix BBOXS
        new_df["bounding_box"] = new_df['key_points_2D'].apply(self.bbox_fix)

        # TODO: Create some sort of crop algorithm to crop the image with the bounding box in mind preferably a array (C*W*H) WHERE W = H = 368

        # TODO: Create code that centralises the image around keypoint 3 so the middle of the image is at 0,0

        # Centralising dataset around center keypoint center
        new_df['key_points_2D'] = new_df['key_points_2D'].apply(self.centralise_2d)
        new_df['key_points_3D'] = new_df['key_points_3D'].apply(self.centralise_3d)

        # Normalise Dataframe
        new_df = self.normalise(new_df)

        # Original DF being used
        self.bugs_frame = new_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.bugs_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.bugs_frame.iloc[idx, 0])
        image = io.imread(img_name)
        height, width, _ = image.shape
        df_columns = self.bugs_frame.columns.values.tolist()
        sample = {'image':image}
 
        for x in range(len(df_columns)):    
            sample[df_columns[x]] = self.bugs_frame.iloc[idx,x]

        # Create Heatmap guassian for each keypoint
        print(sample['key_points_2D'])
        print(sample['key_points_2D'][2,0], sample['key_points_2D'][2,0])
        sample['heatmap'] = np.zeros((round(height/self.stride), round(width/self.stride), len(sample['key_points_2D']) + 1), dtype=np.float32)
        for i in range(len(sample['key_points_2D'])):
            x = int(sample['key_points_2D'][i][0]) * 1.0 / self.stride
            y = int(sample['key_points_2D'][i][1]) * 1.0 / self.stride
            heat_map = self.guassian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            sample['heatmap'][:, :, i + 1] = heat_map

        sample['heatmap'][:, :, 0] = 1.0 - np.max(sample['heatmap'][:, :, 1:], axis=2)

        sample['centermap'] = np.zeros((height, width, 1), dtype=np.float32)
        center_map = self.guassian_kernel(size_h=height, size_w=width, center_x=sample['key_points_2D'][2,0], center_y=sample['key_points_2D'][2,1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        sample['centermap'][:, :, 0] = center_map

        if self.transform:
            sample = self.transform(sample)

        return sample
    
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
        return np.nan_to_num((x-self.means_2d)/self.std_2d)
    def normal_3d(self, x):
        return np.nan_to_num((x-self.means_3d)/self.std_3d)

    def normalise(self, df):
        array_2d = np.array(df['key_points_2D'].to_numpy())
        array_3d = np.array(df['key_points_3D'].to_numpy())

        for x in range(len(array_2d)):
            array_2d[x] = np.array(array_2d[x])
            array_3d[x] = np.array(array_3d[x])

        fixed_array_2d = np.empty((len(array_2d),124))
        fixed_array_3d = np.empty((len(array_3d),186))
        for x in range(len(fixed_array_2d)):
            z = array_2d[x].reshape(1,124)
            fixed_array_2d[x] = z
        for x in range(len(fixed_array_3d)):
            z = array_3d[x].reshape(1,186)
            fixed_array_3d[x] = z

        for x in range(len(fixed_array_2d[0])):
            self.std_2d.append(np.std(fixed_array_2d[:,x], axis=0))
            self.means_2d.append(np.mean(fixed_array_2d[:,x], axis=0))
        self.means_2d = np.array(self.means_2d).reshape((62,2))
        self.std_2d = np.array(self.std_2d).reshape((62,2))

        for x in range(len(fixed_array_3d[0])):
            self.std_3d.append(np.std(fixed_array_3d[:,x], axis=0))
            self.means_3d.app.end(np.mean(fixed_array_3d[:,x], axis=0))
        self.means_3d = np.array(self.means_3d).reshape((62,3))
        self.std_3d = np.array(self.std_3d).reshape((62,3))

        df['key_points_2D'] = df['key_points_2D'].apply(self.normal_2d)
        df['key_points_3D'] = df['key_points_3D'].apply(self.normal_3d)
        return df

    def guassian_kernel(self, size_w, size_h, center_x, center_y, sigma):
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)
    
    def bbox_fix(self, keypoints):
        padding = 1
        x_coordinates, y_coordinates = zip(*keypoints)
        x_coordinates = [i for i in x_coordinates if i != 0]
        y_coordinates = [i for i in y_coordinates if i != 0]
        return [round(min(x_coordinates)-padding), round(min(y_coordinates)-padding), round(max(x_coordinates)+padding), round(max(y_coordinates)+padding)]
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        sample_keys = list(sample.keys())
        sample_data = list(sample.values())

        dic ={'image': torch.from_numpy(image)}
        dic[sample_keys[1]] = sample_data[1]
        for x in range(2,len(sample_keys)):
            dic[sample_keys[x]] = torch.FloatTensor(sample_data[x])
        return dic