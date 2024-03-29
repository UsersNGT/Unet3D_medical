import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options

parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=list,default=[0], help='use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=1,help='number of classes') # 默认2类分割，目标和背景
# parser.add_argument('--upper', type=int, default=200, help='')
# parser.add_argument('--lower', type=int, default=-200, help='')
# parser.add_argument('--norm_factor', type=float, default=200.0, help='')
# parser.add_argument('--expand_slice', type=int, default=20, help='')
# parser.add_argument('--min_slices', type=int, default=48, help='')
# parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
# parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
# parser.add_argument('--valid_rate', type=float, default=0.2, help='')
parser.add_argument('--win_width', type=float, default=800.0, help='')      # 窗宽
parser.add_argument('--win_level', type=float, default=300.0, help='')      # 窗位


# train
parser.add_argument('--save',default='Unet3D_test128',help='save path of trained model')
parser.add_argument('--dataset_path',default = "/data/private/data/bone_data/train/precess_data_nocrop/train.lst",help='fixed trainset root path')
parser.add_argument('--epochs', type=int, default=500, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--n_threads', type=int, default=32,help='number of threads for data loading')
parser.add_argument('--batch_size', type=list, default=8,help='batch size of trainset')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=128)  # patch的大小
parser.add_argument('--val_crop_max_size', type=int, default=96)
parser.add_argument('--sample_num', type=int, default=6, help="numbers of sample on every data") # 一个数据的采样次数

# test
parser.add_argument('--test_cut_size', type=int, default=160, help='size of sliding window')  # 测试时 patch大小
parser.add_argument('--test_cut_stride', type=int, default=154, help='stride of sliding window') # 步长
parser.add_argument('--postprocess', type=bool, default=False, help='post process')
parser.add_argument('--test_data_path',default ="/data/private/data/bone_data/test" ,help='Testset path')


args = parser.parse_args()


