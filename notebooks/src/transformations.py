import random
import torchvision
import cv2
class RandomHorizontalFlip(object):
    """Random horizontal flip the image.

    Args:
        prob: the probability to flip.
    """
    
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        if random.random() < self.prob:
            img = sample['image']
            heatmaps = sample['heatmap']
            kpt = sample['key_points_2D']
            flipped_image = cv2.flip(img, 1)
            for x in range(len(heatmaps)):
                heatmaps[x] = cv2.flip(heatmaps[x],1)

            X = img.shape[1]
            for pos in range(len(kpt)):
                kpt[pos,0] = X - kpt[pos,0]

            sample['image'] = flipped_image
            sample['heatmap'] = heatmaps
            sample['key_points_2D'] = kpt
            return sample
        return sample
class RandomVerticallFlip(object):
    """Random Vertical flip the image.

    Args:
        prob: the probability to flip.
    """
    
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        if random.random() < self.prob:
            img = sample['image']
            heatmaps = sample['heatmap']
            kpt = sample['key_points_2D']
            flipped_image = cv2.flip(img, 0)
            for x in range(len(heatmaps)):
                heatmaps[x] = cv2.flip(heatmaps[x],0)

            Y = img.shape[0]
            for pos in range(len(kpt)):
                kpt[pos,1] = Y - kpt[pos,1]

            sample['image'] = flipped_image
            sample['heatmap'] = heatmaps
            sample['key_points_2D'] = kpt
            return sample
        return sample

class RandomColourShifts(object):
    """Randomly Colour shifts the image"""

    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        # if random.random() < self.prob:
        color_jitter = torchvision.transforms.ColorJitter(contrast =0.5,saturation=0.5)
        img = sample['image']
        sample['image'] = color_jitter(img)
        return sample