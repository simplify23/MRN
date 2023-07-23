<div align="center">

# MRN: Multiplexed Routing Network <br/> for Incremental Multilingual Text Recognition

[Paper](https://arxiv.org/abs/2305.14758) | [Method](#methods) |[IMLTR Dataset](#imltr-dataset) | [Getting Started](#getting-started) | [Citation](#citation)

</div>

It started as code for the paper:

**MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition**
(Accepted by ICCV 2023)

This project is a toolkit for the novel scenario of Incremental Multilingual Text Recognition (IMLTR), the project supports many incremental learning methods and proposes a more applicable method for IMLTR: Multiplexed Routing Network (MRN) and the corresponding dataset. The project provides an efficient framework to assist in developing new methods and analyzing existing ones under the IMLTR task, and we hope it will advance the IMLTR community.

<div align="center">
    
![image](https://github.com/simplify23/MRN/assets/39580716/b865e4c3-e1a4-4fb7-a0d2-91ebc959af46)

</div>


## Methods

* [x] Base: Baseline method which simply updates parameters on new tasks.
* [x] Joint: Bound method: data for all tasks are trained at once, an upper bound for the method.
* [x] [EWC](https://arxiv.org/abs/1612.00796) `[PNAS2017]`: Overcoming catastrophic forgetting in neural networks. 
* [x] [LwF](https://arxiv.org/abs/1911.07053) `[ECCV2016]`:  Learning without Forgetting.
* [x] [WA](https://arxiv.org/abs/1911.07053) `[CVPR2020]`: Maintaining Discrimination and Fairness in Class Incremental Learning. 
* [x] [DER](https://arxiv.org/abs/2103.16788) `[CVPR2021]`: DER: Dynamically Expandable Representation for Class Incremental Learning. 
* [x] [MRN](https://arxiv.org/abs/2305.14758) `[ICCV2023]`: MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition. 

### IMLTR Dataset

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

### Training
```
python3 tiny_train.py --config=config/crnn_3090.py --exp_name CRNN_real
```
<h3 id="pretrained_models"> Run demo with pretrained model <a href="https://colab.research.google.com/github/ku21fan/STR-Fewer-Labels/blob/master/demo_in_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> </h3> 

1. [Download pretrained model](https://www.dropbox.com/sh/23adceu2i85c4x1/AACLmaiL43Jy8eYIVVUkZ344a?dl=0) <br>
There are 2 models (CRNN or TRBA) and 5 different settings of each model.

    Setting | Description
    -- | --
    Baseline-synth | Model trained on 2 synthetic datasets (MJSynth + SynthText)
    Baseline-real | Model trained on 11 real datasets (Real-L in Table 1 of our paper)
    Aug | Best augmentation setting in our experiments
    PL | Combination of Aug and Pseudo-Label (PL)
    PR | Combination of Aug, PL and RotNet

2. Add image files to test into `demo_image/`
3. Run demo.py
   ```
   CUDA_VISIBLE_DEVICES=0 python3 demo.py --model_name TRBA --image_folder demo_image/ \
   --saved_model TRBA-Baseline-real.pth
   ```

### Arguments
train.py (as a default, evaluate trained model on 6 benchmark datasets at the end of training.)
* `--train_data`: folder path to training lmdb dataset. default: `data_CVPR2021/training/label/`
* `--valid_data`: folder path to validation lmdb dataset. default: `data_CVPR2021/validation/`
* `--select_data`: select training data. default is 'label' which means 11 real labeled datasets.
* `--batch_ratio`: assign ratio for each selected data in the batch. default is '1 / number of datasets'.
* `--model_name`: select model 'CRNN' or 'TRBA'.
* `--Aug`: whether to use augmentation |None|Blur|Crop|Rot|
* `--saved_model`: assign saved model to use pretrained model such as RotNet and MoCo.


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
