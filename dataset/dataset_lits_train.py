from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(args.dataset_path)

        self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                #RandomFlip_LR(prob=0.5),
                #RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])

    def __getitem__(self, index):
        name = self.filename_list[index]
        # print(name)
        data = np.load(name, allow_pickle=True)

        ct_array = data["vol"].copy()
        seg_array = data["mask"].copy()
        spacing = data["src_spacing"].copy()
        del data
      
        # ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        # seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        # ct_array = sitk.GetArrayFromImage(ct)
        # seg_array = sitk.GetArrayFromImage(seg)

        #ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        ct_array = self._window_array(ct_array)
        if self.transforms:
            ct_array,seg_array = self.transforms(ct_array, seg_array)     

        # if True:
        #     import time
        #     s = str(time.time())
        #     os.makedirs('./debug', exist_ok=True)
        #     sitk.WriteImage(sitk.GetImageFromArray(ct_array), './debug/' + s + '.nii.gz')
        #     sitk.WriteImage(sitk.GetImageFromArray(seg_array), './debug/' + s + '-seg.nii.gz')
        
        return ct_array, seg_array
    



    def _window_array(self, vol):
        win = [
            self.args.win_level - self.args.win_width / 2,
            self.args.win_level + self.args.win_width / 2,
        ]
        vol = torch.clamp(vol, win[0], win[1])
        vol -= win[0]
        vol /= self.args.win_width
        return vol
    
    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            for line in file_to_read:
                line = line.strip()  # 整行读取数据
                if line:
                    file_name_list.append(line)
        assert len(file_name_list) != 0, "file name list is Kong"
        return file_name_list * self.args.sample_num

if __name__ == "__main__":
    sys.path.append(r"C:\Users\sanxi\Desktop\code\github\3DUNet-Pytorch")
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 1, False, num_workers=0)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())