<div align="center">

# MRN: Multiplexed Routing Network <br/> for Incremental Multilingual Text Recognition
[![Apache License 2.0](https://img.shields.io/github/license/baudm/parseq)](https://github.com/baudm/parseq/blob/main/LICENSE)
[![arXiv preprint](http://img.shields.io/badge/arXiv-2207.06966-b31b1b)](https://arxiv.org/abs/2305.14758)
[![In Proc. ICCV 2023](http://img.shields.io/badge/ECCV-2022-6790ac)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/556_ECCV_2022_paper.php)
[![Gradio demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Gradio-ff7c00)](https://huggingface.co/spaces/baudm/PARSeq-OCR)


[Method](#method-tldr) | [Sample Results](#sample-results) | [Getting Started](#getting-started) | [FAQ](#frequently-asked-questions) | [Training](#training) | [Evaluation](#evaluation) | [Citation](#citation)

</div>

Scene Text Recognition (STR) models use language context to be more robust against noisy or corrupted images. Recent approaches like ABINet use a standalone or external Language Model (LM) for prediction refinement. In this work, we show that the external LM&mdash;which requires upfront allocation of dedicated compute capacity&mdash;is inefficient for STR due to its poor performance vs cost characteristics. We propose a more efficient approach using **p**ermuted **a**uto**r**egressive **seq**uence (PARSeq) models. View our ECCV [poster](https://drive.google.com/file/d/19luOT_RMqmafLMhKQQHBnHNXV7fOCRfw/view) and [presentation](https://drive.google.com/file/d/11VoZW4QC5tbMwVIjKB44447uTiuCJAAD/view) for a brief overview.


Official PyTorch implementation of MRN (Accepted by ICCV 2023)
MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition


## To Do List
* [x] base
* [x] lwf
* [x] ewc
* [x] wa
* [x] der
* [x] mrn


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
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
```
### Download lmdb dataset for traininig and evaluation 
See [`data.md`](https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md)
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

### Training
1. Train CRNN model with only real data.

       CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name CRNN --exp_name CRNN_real

2. Train CRNN with augmentation (For TRBA, use `--Aug Blur5-Crop99`)
   ```
   CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name CRNN --exp_name CRNN_aug --Aug Crop90-Rot15
   ```
   
5. Train with PL + RotNet (PR).
   ```
   CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name CRNN --exp_name CRNN_PR \
   --saved_model saved_models/NV_Pretrain_RotNet/best_score.pth --Aug Crop90-Rot15 \
   --semi Pseudo --model_for_PseudoLabel saved_models/CRNN_NVInitRotNet/best_score.pth
   ```

Try our best accuracy model [TRBA_PR](https://www.dropbox.com/s/s0c26oe8dvk7tsg/TRBA-PR.pth?dl=0) by replacing CRNN to TRBA and `--Aug Crop90-Rot15` to `--Aug Blur5-Crop99`.

### Evaluation
Test CRNN model.
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_type benchmark --model_name CRNN \
--saved_model saved_models/CRNN_real/best_score.pth
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
* `--self_pre`: whether to use self-supversied pretrained model |RotNet|MoCo|. default: RotNet


test.py
* `--eval_data`: folder path to evaluation lmdb dataset. As a default, when you use `eval_type`, this will be set to `data_CVPR2021/evaluation/benchmark/` or `data_CVPR2021/evaluation/addition/`
* `--eval_type`: select 'benchmark' to evaluate 6 evaluation datasets. select 'addition' to evaluate 7 additionally collected datasets (used in Table 6 in our supplementary material).
* `--model_name`: select model 'CRNN' or 'TRBA'.
* `--saved_model`: assign saved model to evaluation.

demo.py
* `--image_folder`: path to image_folder which contains text images. default: `demo_image/`
* `--model_name`: select model 'CRNN' or 'TRBA'.
* `--saved_model`: assign saved model to use.


## When you need to train on your own dataset or Non-Latin language datasets.
1. Create your own lmdb dataset. You may need `pip3 install opencv-python` to `import cv2`.

       python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/

   At this time, `gt.txt` should be `{imagepath}\t{label}\n` <br>
   For example
   ```
   test/word_1.png Tiredness
   test/word_2.png kills
   test/word_3.png A
   ...
   ```
2. Modify `--select_data`, `--batch_ratio`, and `opt.character`, see [this issue](https://github.com/clovaai/deep-text-recognition-benchmark/issues/85).


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
