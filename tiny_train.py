import os
import sys
import time
import random
import argparse

import il_modules
from data.data_manage import Dataset_Manager
from il_modules.base import BaseLearner
from il_modules.der import DER
from il_modules.ewc import EWC
from il_modules.lwf import LwF
from il_modules.wa import WA

print(os.getcwd()) #打印出当前工作路径
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import numpy as np
from mmcv import Config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tools.utils import CTCLabelConverter, AttnLabelConverter, Averager, adjust_learning_rate
from data.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modules.model import Model
from test import validation, benchmark_all_eval
from modules.semi_supervised import PseudoLabelLoss, MeanTeacherLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_data_log(line,name=None):
    '''

    :param name:
    :param line: list of the string [a,b,c]
    :return:
    '''
    with open(f"data_any.txt", "a+") as log:
        log.write(line)

def load_dict(path,char,tmp_char):
    ch_list = []
    character = []
    f = open(path + "/dict.txt")
    line = f.readline()
    while line:
        ch_list.append(line.strip("\n"))
        # print(line)
        line = f.readline()
    f.close()

    for ch in ch_list:
        if char.get(ch, None) == None:
            char[ch] = 1
        if tmp_char.get(ch, None) == None:
            tmp_char[ch] = 1
        else:
            tmp_char[ch] +=1
    for key, value in char.items():
        character.append(key)
    print("dict has {} number characters\n".format(len(character)))
    return character,tmp_char

def count_char_score(tmp_char):
    beta = 0.9999
    gamma = 2.0
    weights = []
    for key, value in tmp_char.items():
        if value > 2:
            print("values:{} {}".format(key,value))
        effective_num = 1.0 - np.power(beta, value)
        weights.append((1.0 - beta) / np.array(effective_num))
    weights = weights / np.sum(weights) * len(tmp_char)
    return weights

def build_arg(parser):
    parser.add_argument(
        "--config",
        default="config/crnn.py",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--train_data",
        # default="data_CVPR2021/training/label/",
        default="../dataset/MLT2017/val_gt/mlt_2017_val",
        help="path to training dataset",
    )
    parser.add_argument(
        "--valid_data",
        default="../dataset/MLT2017/val_gt/mlt_2017_val",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument(
        "--num_iter", type=int, default=200000, help="number of iterations to train for"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=2000,
        help="Interval between each validation",
    )
    parser.add_argument(
        "--log_multiple_test", action="store_true", help="log_multiple_test"
    )
    parser.add_argument(
        "--FT", type=str, default="init", help="whether to do fine-tuning |init|freeze|"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    """ Optimizer """
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer |sgd|adadelta|adam|"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="learning rate, default=1.0 for Adadelta, 0.0005 for Adam",
    )
    parser.add_argument(
        "--sgd_momentum", default=0.9, type=float, help="momentum for SGD"
    )
    parser.add_argument(
        "--sgd_weight_decay", default=0.000001, type=float, help="weight decay for SGD"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.95,
        help="decay rate rho for Adadelta. default=0.95",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="eps for Adadelta. default=1e-8"
    )
    parser.add_argument(
        "--schedule",
        default="super",
        nargs="*",
        help="(learning rate schedule. default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER",
    )
    parser.add_argument(
        "--lr_drop_rate",
        type=float,
        default=0.1,
        help="lr_drop_rate. default is the same setting with ASTER",
    )
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, required=False, help="CRNN|TRBA")
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    """ Data processing """
    parser.add_argument(
        "--select_data",
        type=str,
        # default="label",
        default="../dataset/MLT2017/val_gt/mlt_2017_val",
        help="select training data. default is `label` which means 11 real labeled datasets",
    )
    parser.add_argument(
        "--batch_ratio",
        type=str,
        default="1.0",
        help="assign ratio for each selected data in the batch",
    )
    parser.add_argument(
        "--total_data_usage_ratio",
        type=str,
        default="1.0",
        help="total data usage ratio, this ratio is multiplied to total number of data.",
    )
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        help="character label",
    )
    parser.add_argument(
        "--NED", action="store_true", help="For Normalized edit_distance"
    )
    parser.add_argument(
        "--Aug",
        type=str,
        default="None",
        help="whether to use augmentation |None|Blur|Crop|Rot|",
    )
    """ Semi-supervised learning """
    parser.add_argument(
        "--semi",
        type=str,
        default="None",
        help="whether to use semi-supervised learning |None|PL|MT|",
    )
    parser.add_argument(
        "--MT_C", type=float, default=1, help="Mean Teacher consistency weight"
    )
    parser.add_argument(
        "--MT_alpha", type=float, default=0.999, help="Mean Teacher EMA decay"
    )
    parser.add_argument(
        "--model_for_PseudoLabel", default="", help="trained model for PseudoLabel"
    )
    parser.add_argument(
        "--self_pre",
        type=str,
        default="RotNet",
        help="whether to use `RotNet` or `MoCo` pretrained model.",
    )
    """ exp_name and etc """
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument(
        "--manual_seed", type=int, default=111, help="for random seed setting"
    )
    parser.add_argument(
        "--saved_model", default="", help="path to model to continue training"
    )
    return parser

def train(opt, log):
    # ["Latin", "Chinese", "Arabic", "Japanese", "Korean", "Bangla","Hindi","Symbols"]
    # train_datasets = ["mlt_2017_train_Latin", "mlt_2017_train_Chinese", "mlt_2017_train_Arabic", "mlt_2017_train_Japanese", "mlt_2017_train_Korean", "mlt_2017_train_Bangla", "mlt_2017_train_Symbols"]
    # valid_datasets = ["mlt_2017_val_Latin", "mlt_2017_val_Chinese", "mlt_2017_val_Arabic", "mlt_2017_val_Japanese", "mlt_2017_val_Korean", "mlt_2017_val_Bangla", "mlt_2017_val_Symbols"]
    # train_datasets = [opt.root_pefix + "_train_" + lan for lan in opt.lan_list]
    # valid_datasets = [opt.root_pefix + "_test_" + lan for lan in opt.lan_list]
    write_data_log(f"---- {opt.exp_name} ----\n")

    if opt.ch_list!=None:
        train_datasets = [ch+"/train" for ch in opt.ch_list]
        valid_datasets = [ch+"/test" for ch in opt.ch_list]
    else:
        train_datasets = [lan for lan in opt.lan_list]
        valid_datasets = [lan for lan in opt.lan_list]

    best_scores = []
    ned_scores = []
    valid_datas = []
    char = dict()
    """ final options """
    # print(opt)
    opt_log = "------------ Options -------------\n"
    args = vars(opt)
    for k, v in args.items():
        if str(k) == "character" and len(str(v)) > 500:
            opt_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
    opt_log += "---------------------------------------\n"
    # print(opt_log)
    log.write(opt_log)
    if opt.il =="lwf":
        learner = LwF(opt)
    elif opt.il == "wa":
        learner = WA(opt)
    elif opt.il =="ewc":
        learner = EWC(opt)
    elif opt.il == "der":
        learner = DER(opt)
    else:
        learner = BaseLearner(opt)

    data_manager = Dataset_Manager()
    
    for taski in range(len(train_datasets)):
        train_data = os.path.join(opt.train_data, train_datasets[taski])
        valid_data = os.path.join(opt.valid_data, valid_datasets[taski])
        valid_datas.append(valid_data)

        tmp_char = dict()
        """dataset preparation"""
        select_data = opt.select_data

        # # set batch_ratio for each data.
        # if opt.batch_ratio:
        #     batch_ratio = opt.batch_ratio.split("-")
        #     # batch_ratio = opt.batch_ratio
        # else:
        #     batch_ratio = [round(1 / len(select_data), 3)] * len(select_data)
        #
        # train_loader = Batch_Balanced_Dataset(
        #     opt, train_data, select_data, batch_ratio, log,taski
        # )
        if taski == 0:
            data_manager.init_start(opt, train_data, select_data, log, taski, memory=None)
        train_loader = data_manager

        #-------load char to dict --------#
        for data_path in opt.select_data:
            if data_path=="/":
                opt.character = load_dict(train_data,char)
            else:
                opt.character,tmp_char = load_dict(data_path+f"/{opt.lan_list[taski]}",char,tmp_char)
        char_score = count_char_score(tmp_char)
        AlignCollate_valid = AlignCollate(opt, mode="test")
        valid_dataset, valid_dataset_log = hierarchical_dataset(
            root=valid_data, opt=opt, mode="test"
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid,
            pin_memory=False,
        )
        log.write(valid_dataset_log)
        print("-" * 80)
        log.write("-" * 80 + "\n")
        # log.close()

        # ----- incremental model start -------

        learner.incremental_train(taski, opt.character, train_loader, valid_loader)

        # ----- incremental model end -------
        """ Evaluation at the end of training """
        best_scores,ned_scores = learner.test(AlignCollate_valid,valid_datas,best_scores,ned_scores, taski)
        learner.after_task()

    print(
            'ALL Average Incremental Accuracy: {:.2f} '.format(sum(best_scores)/len(best_scores))
        )

def val(model, criterion, valid_loader, converter, opt,optimizer,best_score,start_time,iteration,train_loss_avg,taski):
    with open(f"./saved_models/{opt.exp_name}/log_train.txt", "a") as log:
        model.eval()
        with torch.no_grad():
            (
                valid_loss,
                current_score,
                ned_score,
                preds,
                confidence_score,
                labels,
                infer_time,
                length_of_data,
            ) = validation(model, criterion, valid_loader, converter, opt)
        model.train()

        # keep best score (accuracy or norm ED) model on valid dataset
        # Do not use this on test datasets. It would be an unfair comparison
        # (training should be done without referring test set).
        if current_score > best_score:
            best_score = current_score
            if opt.ch_list!=None:
                name = opt.ch_list[taski]
            else:
                name = opt.lan_list[taski]
            torch.save(
                model.state_dict(),
                f"./saved_models/{opt.exp_name}/{name}_{taski}_best_score.pth",
            )

        # validation log: loss, lr, score (accuracy or norm ED), time.
        lr = optimizer.param_groups[0]["lr"]
        elapsed_time = time.time() - start_time
        valid_log = f"\n[{iteration}/{opt.num_iter}] Train_loss: {train_loss_avg.val():0.5f}, Valid_loss: {valid_loss:0.5f} \n "
        # valid_log += f", Semi_loss: {semi_loss_avg.val():0.5f}\n"
        valid_log += f'{"":9s}Current_score: {current_score:0.2f},   Ned_score: {ned_score:0.2f}\n'
        valid_log += f'{"":9s}Current_lr: {lr:0.7f}, Best_score: {best_score:0.2f}\n'
        valid_log += f'{"":9s}Infer_time: {infer_time:0.2f},     Elapsed_time: {elapsed_time:0.2f}\n'

        # show some predicted results
        dashed_line = "-" * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
        for gt, pred, confidence in zip(
                labels[:5], preds[:5], confidence_score[:5]
        ):
            if "Attn" in opt.Prediction:
                gt = gt[: gt.find("[EOS]")]
                pred = pred[: pred.find("[EOS]")]

            predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
        predicted_result_log += f"{dashed_line}"
        valid_log = f"{valid_log}\n{predicted_result_log}"
        print(valid_log)
        log.write(valid_log + "\n")
        write_data_log(f"Task {opt.lan_list[taski]} [{iteration}/{opt.num_iter}] : Score:{current_score:0.2f} LR:{lr:0.7f}\n")



def test(AlignCollate_valid,valid_datas,model,criterion,converter,opt,best_scores,taski,log):
    print("---Start evaluation on benchmark testset----")
    """ keep evaluation model and result logs """
    os.makedirs(f"./result/{opt.exp_name}", exist_ok=True)
    os.makedirs(f"./evaluation_log", exist_ok=True)
    if opt.ch_list != None:
        name = opt.ch_list[taski]
    else:
        name = opt.lan_list[taski]
    saved_best_model = f"./saved_models/{opt.exp_name}/{name}_{taski}_best_score.pth"
    # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
    model.load_state_dict(torch.load(f"{saved_best_model}"))

    task_accs = []
    for val_data in valid_datas:
        valid_dataset, valid_dataset_log = hierarchical_dataset(
            root=val_data, opt=opt, mode="test")
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid,
            pin_memory=False,
        )

        model.eval()
        with torch.no_grad():
            (
                valid_loss,
                current_score,
                ned_score,
                preds,
                confidence_score,
                labels,
                infer_time,
                length_of_data,
            ) = validation(model, criterion, valid_loader, converter, opt)

        task_accs.append(current_score)

    best_scores.append(sum(task_accs) / len(task_accs))

    acc_log= f'Task {taski} Test Average Incremental Accuracy: {best_scores[taski]} \n Task {taski} Incremental Accuracy: {task_accs}'
    # acc_log = f'Task {taski} Test Average Incremental Accuracy: {best_scores[taski]} \n '
    # acc_log += f'Task {taski} Incremental Accuracy: {task_accs:.2f}'
    write_data_log(f'Task {taski} Avg Acc: {best_scores[taski]:0.2f} \n  {task_accs}\n')
    print(acc_log)
    log.write(acc_log)
    return best_scores,log

def change_model(opt, model):
    """ model configuration """
    # model.module.reset_class(opt, device)
    reset_class(model.module, opt, device)
    # data parallel for multi-GPU
    model.train()
    return model


def build_model(opt, log):
    """ model configuration """

    model = Model(opt)

    # weight initialization
    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"Skip {name} as it is already initialized")
            continue
        try:
            if "bias" in name:
                init.constant_(param, 0.0)
            elif "weight" in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if "weight" in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    return model,log


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = build_arg(parser)

    arg = parser.parse_args()
    cfg = Config.fromfile(arg.config)

    opt={}
    opt.update(cfg.common)
    opt.update(cfg.test)
    opt.update(cfg.model)
    opt.update(cfg.train)
    opt.update(cfg.optimizer)

    opt = argparse.Namespace(**opt)

    # if opt.model_name == "CRNN":  # CRNN = NVBC
    #     opt.Transformation = "None"
    #     opt.FeatureExtraction = "VGG"
    #     opt.SequenceModeling = "BiLSTM"
    #     opt.Prediction = "CTC"
    #
    # elif opt.model_name == "TRBA":  # TRBA
    #     opt.Transformation = "TPS"
    #     opt.FeatureExtraction = "ResNet"
    #     opt.SequenceModeling = "BiLSTM"
    #     opt.Prediction = "Attn"
    #
    # elif opt.model_name == "RBA":  # RBA
    #     opt.Transformation = "None"
    #     opt.FeatureExtraction = "ResNet"
    #     opt.SequenceModeling = "BiLSTM"
    #     opt.Prediction = "Attn"

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # It fasten training.
    cudnn.deterministic = True

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience
    opt.num_gpu = torch.cuda.device_count()
    # if opt.num_gpu > 1:
    #     print(
    #         "We recommend to use 1 GPU, check your GPU number, you would miss CUDA_VISIBLE_DEVICES=0 or typo"
    #     )
    #     print("To use multi-gpu setting, remove or comment out these lines")
    #     sys.exit()

    if sys.platform == "win32":
        opt.workers = 0

    """ directory and log setting """
    if not opt.exp_name:
        opt.exp_name = f"Seed{opt.manual_seed}-{opt.model_name}"

    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)
    log = open(f"./saved_models/{opt.exp_name}/log_train.txt", "a")
    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )
    log.write(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}\n"
    )
    os.makedirs(f"./tensorboard", exist_ok=True)
    # opt.writer = SummaryWriter(log_dir=f"./tensorboard/{opt.exp_name}")

    train(opt, log)
