import numpy as np
import os
from torch.utils import data
from PIL import Image
import glob
import h5py

class SHHA(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None, data_augment=1):
        self.img_path = data_path + '/images'
        self.gt_path = data_path + '/ground_truth'
        self.data_files = glob.glob(os.path.join(self.img_path, '*.jpg'))
        self.num_samples = len(self.data_files)
        self.num_samples = self.num_samples*data_augment
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.data_aug = data_augment
    
    def __getitem__(self, index):
        index = index // self.data_aug
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.gt_transform is not None:
            den = self.gt_transform(den)               
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        
        img = Image.open(fname)
        if img.mode == 'L':
            img = img.convert('RGB')

        den_path = fname.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth')
        den = h5py.File(den_path, 'r')
        den = np.asarray(den['density'])
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)  
        return img, den    

    def get_num_samples(self):
        return self.num_samples       
            
