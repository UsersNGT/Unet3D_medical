## 3DUNet implemented with pytorch

## Introduction
The repository is a 3DUNet implemented with pytorch, referring to 
this [project](https://github.com/panxiaobai/lits_pytorch).
 I have redesigned the code structure and used the model to perform liver and tumor segmentation on the lits2017 dataset.  
paper: [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf)
### Requirements:  
```angular2
pytorch >= 1.1.0
torchvision
SimpleITK
Tensorboard
Scipy
```
### Code Structure
```angular2
├── dataset          # Training and testing dataset
│   ├── dataset_lits_train.py 
│   ├── dataset_lits_val.py
│   ├── dataset_lits_test.py
|   ├── generate_dataset.py    # 生成npz data，accerate traing
│   └── transforms.py 
├── models           # Model design
│   ├── nn
│   │   └── module.py
│   │── ResUNet.py      # 3DUNet class
│   │── Unet.py      # 3DUNet class
│   │── SegNet.py      # 3DUNet class
│   └── KiUNet.py      # 3DUNet class
├── experiments           # Trained model
|── utils            # Some related tools
|   ├── common.py
|   ├── weights_init.py
|   ├── logger.py
|   ├── metrics.py
|   └── loss.py
├── preprocess_LiTS.py  # preprocessing for  raw data
├── test.py          # Test code
├── train.py         # Standard training code
└── config.py        # Configuration information for training and testing
```
##quick start
### 1) generate_data
       use 'generate_dataset.py' to generate *.npz, this can quick read data.
---
### 2) Training 3DUNet
1. Firstly, you should change the some parameters in `config.py`,especially, please set `--dataset_path` to `./fixed_data`  
All parameters are commented in the file `config.py`. 
2. Secondely,run `python train.py --save model_name`  
3. Besides, you can observe the dice and loss during the training process 
in the browser through `tensorboard --logdir ./output/model_name`. 
---   
### 3) Testing 3DUNet
run `test.py`  
Please pay attention to path of trained model in `test.py`.   
(Since the calculation of the 3D convolution operation is too large,
 I use a sliding window to block the input tensor before prediction, and then stitch the results to get the final result.
 The size of the sliding window can be set by yourself in `config.py`)  

After the test, you can get the test results in the corresponding folder:`./experiments/model_name/result`

You can also read my Chinese introduction about this 3DUNet project [here](https://zhuanlan.zhihu.com/p/113318562). However, I no longer update the blog, I will try my best to update the github code.    
If you have any suggestions or questions, 
welcome to open an issue to communicate with me.  
