from torch._C import dtype
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk

class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        # 读取一个data文件并归一化 、resize
        self.ct = sitk.ReadImage(data_path,sitk.sitkInt16)
        self.data_np = sitk.GetArrayFromImage(self.ct)
       
        self.ori_shape = self.data_np.shape
        self.win_level = args.win_level
        self.win_width = args.win_width
        # self.data_np = ndimage.zoom(self.data_np, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale), order=3) # 双三次重采样
        # self.data_np[self.data_np > args.upper] = args.upper
        # self.data_np[self.data_np < args.lower] = args.lower
        # self.data_np = self.data_np/args.norm_factor
        # self.resized_shape = self.data_np.shape
        print("img ori shape: ", self.ori_shape)
        self.data_np = self._window_array(self.data_np)
        # 扩展一定数量的slices，以保证卷积下采样合理运算
        self.data_np = self.padding_img(self.data_np, self.cut_size, self.cut_stride)
        # sitk.WriteImage(sitk.GetImageFromArray(self.data_np), './debug_' + "122" + '.nii.gz')
        # raise
        self.padding_shape = self.data_np.shape
        # 对数据按步长进行分patch操作，以防止显存溢出
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)

        # 读取一个label文件 shape:[s,h,w]
        if label_path:
            self.seg = sitk.ReadImage(label_path,sitk.sitkInt8)
            self.label_np = sitk.GetArrayFromImage(self.seg)
            if self.n_labels==2:
                self.label_np[self.label_np > 0] = 1
            self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long()

        # 预测结果保存
        self.result = None

    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])
        data = torch.FloatTensor(data).unsqueeze(0)
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_result(self):

        size = self.result.shape[2]
        
        N_patches_img = self.result.shape[0]
        # assert (self.result.shape[0] == N_patches_img)


        full_prob = torch.zeros((self.padding_shape[0], self.padding_shape[1],self.padding_shape[2]))  # itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros((self.padding_shape[0], self.padding_shape[1], self.padding_shape[2]))
        i = 0
        for s in range((self.padding_shape[0] - size) // self.cut_stride + 1):  # loop over the full images
            for h in range((self.padding_shape[1] - size) // self.cut_stride + 1):
                for w in range((self.padding_shape[2] - size) // self.cut_stride + 1):
                    full_prob[s * self.cut_stride : s * self.cut_stride + size, h * self.cut_stride : h * self.cut_stride + size,w * self.cut_stride : w * self.cut_stride + size,] += self.result[i][0]
                    full_sum[s * self.cut_stride : s * self.cut_stride + size, h * self.cut_stride : h * self.cut_stride + size,w * self.cut_stride : w * self.cut_stride + size,] += 1
                    i += 1
        # for s in range(N_patches_img):
        #     full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s]
        #     full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1

        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        # print(final_avg.size())
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img

    def _window_array(self, vol):
        win = [
            self.win_level - self.win_width / 2,
            self.win_level + self.win_width / 2,
        ]
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= self.win_width
        return vol

    def padding_img(self, img, size, stride):
        assert (len(img.shape) == 3)  # 3D array
        img_s, img_h, img_w = img.shape
        leftover_s = (img_s - size) % stride
        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        leftover_h = (img_h - size) % stride
        if (leftover_h != 0):
            h = img_h + (stride - leftover_h)
        else:
            h = img_h

        leftover_w = (img_w - size) % stride
        if (leftover_w != 0):
            w = img_w + (stride - leftover_w)
        else:
            w = img_w


        tmp_full_imgs = np.zeros((s, h, w),dtype=np.float32)
        tmp_full_imgs[:img_s, :img_h,:img_w] = img
        print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_s, img_h, img_w = img.shape
        assert (img_s - size) % stride == 0
        assert (img_h - size) % stride == 0
        assert (img_w - size) % stride == 0

        N_patches_img = ((img_s - size) // stride + 1) * ((img_h - size) // stride + 1)*((img_w - size) // stride + 1)

        print("Patches number of the image:{}".format(N_patches_img))
        #patches = np.empty((N_patches_img, size, size, size), dtype=np.float32)
        patches = []
        for s in range((img_s - size) // stride + 1):  # loop over the full images
            for h in range((img_h - size) // stride + 1):
                for w in range((img_w - size) // stride + 1):
                    patch = img[s * stride : s * stride + size, h * stride : h * stride + size,w * stride : w * stride + size,]
                    # patches[] = patch
                    patches.append(patch)

        return patches  # array with all the full_imgs divided in patches

def Test_Datasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'ct/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test samples is: ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, labelpath,args=args), datapath.split('-')[-1]

def Test_Datasets_nolabel(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, '*')))
    
    print("The number of test samples is: ", len(data_list))
    for datapath in data_list:
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, label_path=None, args=args), datapath.split('/')[-1]