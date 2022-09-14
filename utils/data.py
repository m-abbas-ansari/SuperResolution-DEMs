import glob
import torch
import numpy as np
from scipy.io import loadmat
import rasterio as rio
from torch.utils.data import Dataset
from utils import normalize, numeric_kernel

class DemDataset(Dataset):
    """DEM dataset."""

    def __init__(self, root_dir, kernel_dir, crop_size=512, downsampling_ratio=4, transform=None):
        """
        Args:
            root_dir (string): Directory with all the DEMs.
            kernel_dir (string): Directory with all the kernels (learned using KernelGAN) for downsampling.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Function:
            For each loaded DEM, we perform a random crop of crop_size to give us HR, we then downsample it using random kernel to give us LR.
        """
        self.root_dir = root_dir
        self.transform = transform
        dem_list = glob.glob(root_dir + '/*.asc')
        self.dems = [normalize(rio.open(dem).read(1).astype('float')) for dem in dem_list] 
        print(f"Loaded the following DEMS: {dem_list}")

        kernel_list = glob.glob(kernel_dir + '/*.mat')
        self.kernels = [loadmat(k)["Kernel"] for k in kernel_list]
        print(f"Loaded the following kernels: {kernel_list}")
        
        self.crop_size = crop_size 
        self.ratio = 1.0 / downsampling_ratio

    def __len__(self):
        return len(self.dems)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        k = self.kernels[np.random.randint(0, len(self.kernels))]
        dem = self.dems[idx]
        HR = self.RandomCrop(dem, self.crop_size).astype(np.float32)
        lr_shape = (HR.shape[0] // self.ratio, HR.shape[1] // self.ratio)
        LR = numeric_kernel(HR, k, [self.ratio,self.ratio], lr_shape).astype(np.float32) 

        if self.transform:
            HR = self.transform(HR)
            LR = self.transform(LR)
            
        sample = {'HR': HR, 'LR': LR}        
        return sample
    
    
    def RandomCrop(self,image, output_size): # crop the image to the size of output_size
        h, w = image.shape[:2]
        new_h, new_w = (output_size, output_size)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]

        return image