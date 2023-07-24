<div align="center">

# MRN: Multiplexed Routing Network <br/> for Incremental Multilingual Text Recognition


![ICCV 2023](https://img.shields.io/badge/ICCV-2023-ff7c00)
[![ArXiv preprint](http://img.shields.io/badge/ArXiv-2305-b31b1b)](https://arxiv.org/abs/2305.14758)
[![Blog](http://img.shields.io/badge/Blog-Link-6790ac)](https://zhuanlan.zhihu.com/p/643948935)
![LICENSE](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)

[Method](#methods) |[IMLTR Dataset](#imltr-dataset) | [Getting Started](#getting-started) | [Citation](#citation)

</div>

It started as code for the paper:

**MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition**
(Accepted by ICCV 2023)

This project is a toolkit for the novel scenario of Incremental Multilingual Text Recognition (IMLTR), the project supports many incremental learning methods and proposes a more applicable method for IMLTR: Multiplexed Routing Network (MRN) and the corresponding dataset. The project provides an efficient framework to assist in developing new methods and analyzing existing ones under the IMLTR task, and we hope it will advance the IMLTR community.

<div align="center">
    
![image](https://github.com/simplify23/MRN/assets/39580716/b865e4c3-e1a4-4fb7-a0d2-91ebc959af46)

</div>

---
## Methods
### Incremental Learning Methods 
* [x] Base: Baseline method which simply updates parameters on new tasks.
* [x] Joint: Bound method: data for all tasks are trained at once, an upper bound for the method
* [x] [EWC](https://arxiv.org/abs/1612.00796) `[PNAS2017]`: Overcoming catastrophic forgetting in neural networks 
* [x] [LwF](https://arxiv.org/abs/1911.07053) `[ECCV2016]`:  Learning without Forgetting
* [x] [WA](https://arxiv.org/abs/1911.07053) `[CVPR2020]`: Maintaining Discrimination and Fairness in Class Incremental Learning 
* [x] [DER](https://arxiv.org/abs/2103.16788) `[CVPR2021]`: DER: Dynamically Expandable Representation for Class Incremental Learning 
* [x] [MRN](https://arxiv.org/abs/2305.14758) `[ICCV2023]`: MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition 

you can change config `config/crnn_mrn.py` for different il methods or setting.
```
common=dict(
    il="mrn",  # joint_mix ｜ joint_loader | base | lwf | wa | ewc ｜ der  | mrn
    memory="random",  # None | random
    memory_num=2000,
    start_task = 0  # checkpoint start
)
```

### Text Recognition Methods
* [x] [CRNN](https://ieeexplore.ieee.org/abstract/document/7801919) `[TPAMI2017]`: An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition
* [x] [TRBA](https://arxiv.org/abs/1904.01906) `[ICCV2019]`:  What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis
* [x] [SVTR](https://arxiv.org/abs/2205.00159) `[IJCAI2022]`: SVTR: Scene Text Recognition with a Single Visual Model
      
you can change config `config/crnn_mrn.py` for different text recognition modules or setting.
```
""" Model Architecture """
common=dict(
    batch_max_length = 25,
    imgH = 32,
    imgW = 256,
)
model=dict(
    model_name="TRBA",
    Transformation = "TPS",      #None TPS
    FeatureExtraction = "ResNet",    #VGG ResNet SVTR
    SequenceModeling = "BiLSTM",  #None BiLSTM
    Prediction = "Attn",           #CTC Attn
    num_fiducial=20,
    input_channel=4,
    output_channel=512,
    hidden_size=256,
)
```


## IMLTR Dataset
The Dataset can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1Qv4utVzWlLu8UPcBpItHbQ)(passwd:c07h).

```
dataset
├── MLT17_IL
│   ├── test_2017
│   ├── train_2017
├── MLT19_IL
│   ├── test_2019
│   ├── train_2019
```


## Getting Started
### Dependency
- This work was tested with PyTorch 1.6.0, CUDA 10.1 and python 3.6.
```
conda create -n mrn python=3.7 -y
conda activate mrn
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
- requirements : 
```
pip3 install lmdb pillow torchvision nltk natsort fire tensorboard tqdm opencv-python einops timm mmcv shapely scipy
pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
```

## Training
```
python3 tiny_train.py --config=config/crnn_mrn.py --exp_name CRNN_real
```
### Arguments
tiny_train.py (as a default, evaluate trained model on IMLTR datasets at the end of training.
* `--select_data`: folder path to training lmdb datasets. </br> `[" ../dataset/MLT17_IL/train_2017", "../dataset/MLT19_IL/train_2019"] `
*  `--valid_datas`: folder path to testing lmdb dataset. </br>  `[" ../dataset/MLT17_IL/test_2017", "../dataset/MLT19_IL/test_2019"] `
* `--batch_ratio`: assign ratio for each selected data in the batch. default is '1 / number of datasets'.
* `--Aug`: whether to use augmentation |None|Blur|Crop|Rot|

### Config Detail
For detailed configuration modifications please use the config file `config/crnn_mrn.py`
```
common=dict(
    exp_name="TRBA_MRN",  # Where to store logs and models
    il="mrn",  # joint_mix ｜ joint_loader | base | lwf | wa | ewc ｜ der  | mrn
    memory="random",  # None | random
    memory_num=2000,
    batch_max_length = 25,
    imgH = 32,
    imgW = 256,
    manual_seed=111,
    start_task = 0
)

""" Model Architecture """
model=dict(
    model_name="TRBA",
    Transformation = "TPS",      #None TPS
    FeatureExtraction = "ResNet",    #VGG ResNet
    SequenceModeling = "BiLSTM",  #None BiLSTM
    Prediction = "Attn",           #CTC Attn
    num_fiducial=20,
    input_channel=4,
    output_channel=512,
    hidden_size=256,
)



""" Optimizer """
optimizer=dict(
    schedule="super", #default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER
    optimizer="adam",
    lr=0.0005,
    sgd_momentum=0.9,
    sgd_weight_decay=0.000001,
    milestones=[2000,4000],
    lrate_decay=0.1,
    rho=0.95,
    eps=1e-8,
    lr_drop_rate=0.1
)


""" Data processing """
train = dict(
    saved_model="",  # "path to model to continue training"
    Aug="None",  # |None|Blur|Crop|Rot|ABINet
    workers=4,
    lan_list=["Chinese","Latin","Japanese", "Korean", "Arabic", "Bangla"],
    valid_datas=[
                 "../dataset/MLT17_IL/test_2017",
                 "../dataset/MLT19_IL/test_2019"
                 ],
    select_data=[
                 "../dataset/MLT17_IL/train_2017",
                 "../dataset/MLT19_IL/train_2019"
                 ],
    batch_ratio="0.5-0.5",
    total_data_usage_ratio="1.0",
    NED=True,
    batch_size=256,
    num_iter=10000,
    val_interval=5000,
    log_multiple_test=None,
    grad_clip=5,
)

```

### Data Analysis
The experimental results of each task are recorded in `data_any.txt` and can be used for analysis of the data.


## Acknowledgements
This implementation has been based on these repositories:
- [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels)
- [PyCIL: A Python Toolbox for Class-Incremental Learning](https://github.com/G-U-N/PyCIL)

## Citation
Please consider citing this work in your publications if it helps your research.
```
@article{zheng2023mrn,
  title={MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition},
  author={Zheng, Tianlun and Chen, Zhineng and Huang, BingChen and Zhang, Wei and Jiang, Yu-Gang},
  journal={ICCV 2023},
  year={2023}
}
```

## License
This project is released under the Apache 2.0 license.
