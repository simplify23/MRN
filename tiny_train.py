import os
import sys
import time
import random
import argparse
from data.data_manage import Dataset_Manager, Val_Dataset
from il_modules.base import BaseLearner
from il_modules.der import DER
from il_modules.mrn import MRN
from il_modules.ewc import EWC
from il_modules.joint import JointLearner
from il_modules.lwf import LwF
from il_modules.wa import WA

print(os.getcwd())
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from mmcv import Config

from data.dataset import hierarchical_dataset, AlignCollate
from test import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_data_log(line):
    '''

    :param name:
    :param line: list of the string [a,b,c]
    :return:
    '''
    with open(f"data_any.txt", "a+") as log:
        log.write(line)

def load_dict(path,char):
    ch_list = []
    character = []
    f = open(path + "/dict.txt")
    line = f.readline()
    while line:
        ch_list.append(line.strip("\n"))
        line = f.readline()
    f.close()

    for ch in ch_list:
        if char.get(ch, None) == None:
            char[ch] = 1
    for key, value in char.items():
        character.append(key)
    print("dict has {} number characters\n".format(len(character)))
    return character,char


def build_arg(parser):
    parser.add_argument(
        "--config",
        default="config/crnn_mrn.py",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--valid_datas",
        default=[" ../dataset/MLT17_IL/test_2017", "../dataset/MLT19_IL/test_2019"],
        help="path to testing dataset",
    )
    parser.add_argument(
        "--select_data",
        type=str,
        default=[" ../dataset/MLT17_IL/train_2017", "../dataset/MLT19_IL/train_2019"],
        help="select training data.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument(
        "--num_iter", type=int, default=20000, help="number of iterations to train for"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=5000,
        help="Interval between each validation",
    )
    parser.add_argument(
        "--log_multiple_test", action="store_true", help="log_multiple_test"
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
        "--NED", action="store_true", help="For Normalized edit_distance"
    )
    parser.add_argument(
        "--Aug",
        type=str,
        default="None",
        help="whether to use augmentation |None|Blur|Crop|Rot|",
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
    write_data_log(f"----------- {opt.exp_name} ------------\n")
    print(f"----------- {opt.exp_name} ------------\n")

    valid_datasets = train_datasets = [lan for lan in opt.lan_list]

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
    if opt.il == "lwf":
        learner = LwF(opt)
    elif opt.il == "wa":
        learner = WA(opt)
    elif opt.il == "ewc":
        learner = EWC(opt)
    elif opt.il == "der":
        learner = DER(opt)
    elif opt.il == "mrn":
        learner = MRN(opt)
    elif opt.il == "joint_mix" or opt.il == "joint_loader":
        learner = JointLearner(opt)
    else:
        learner = BaseLearner(opt)

    data_manager = Dataset_Manager(opt)
    for taski in range(len(train_datasets)):
        # train_data = os.path.join(opt.train_data, train_datasets[taski])
        for valid_data in opt.valid_datas:
            val_data = os.path.join(valid_data, valid_datasets[taski])
            valid_datas.append(val_data)

        valid_loader = Val_Dataset(valid_datas,opt)
        """dataset preparation"""
        select_data = opt.select_data
        AlignCollate_valid = AlignCollate(opt, mode="test")

        if opt.il =="joint_loader" or opt.il == "joint_mix":
            valid_datas = []
            char = {}
            for taski in range(len(train_datasets)):
                # char={}
                # train_data = os.path.join(opt.train_data, train_datasets[taski])
                for val_data in opt.valid_datas:
                    valid_data = os.path.join(val_data, valid_datasets[taski])
                    valid_datas.append(valid_data)
                data_manager.joint_start(opt, select_data, log, taski, len(train_datasets))
                for data_path in opt.select_data:
                    opt.character, char = load_dict(data_path + f"/{opt.lan_list[taski]}", char)
            print(len(opt.character))
            best_scores,ned_scores = learner.incremental_train(0,opt.character, data_manager, valid_loader,AlignCollate_valid,valid_datas)
            """ Evaluation at the end of training """
            best_scores, ned_scores = learner.test(AlignCollate_valid, valid_datas, best_scores, ned_scores, 0)
            break
        if taski == 0:
            data_manager.init_start(opt, select_data, log, taski)
        train_loader = data_manager

        #-------load char to dict --------#
        for data_path in opt.select_data:
            if data_path=="/":
                opt.character = load_dict(data_path+f"/{opt.lan_list[taski]}",char)
            else:
                opt.character,tmp_char = load_dict(data_path+f"/{opt.lan_list[taski]}",char)
        # ----- incremental model start -------

        learner.incremental_train(taski, opt.character, train_loader, valid_loader)

        # ----- incremental model end -------
        """ Evaluation at the end of training """
        best_scores,ned_scores = learner.test(AlignCollate_valid,valid_datas,best_scores,ned_scores, taski)
        learner.after_task()

    write_data_log(f"----------- {opt.exp_name} ------------\n")
    print(f"----------- {opt.exp_name} ------------\n")
    if len(opt.valid_datas) == 1:
        print(
                'ALL Average Incremental Accuracy: {:.2f} \n'.format(sum(best_scores)/len(best_scores))
            )
        write_data_log('ALL Average Acc: {:.2f} \n'.format(sum(best_scores)/len(best_scores)))
    elif len(opt.valid_datas) == 2:
        print(
            'ALL Average 17 Acc: {:.2f} \n'.format(sum(best_scores) / len(best_scores))
        )
        print(
            'ALL Average 19 Acc: {:.2f} \n'.format(sum(ned_scores) / len(ned_scores))
        )
        write_data_log('ALL 17 Acc: {:.2f} \n'.format(sum(best_scores) / len(best_scores)))
        write_data_log('ALL 19 Acc: {:.2f} \n'.format(sum(ned_scores) / len(ned_scores)))

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
            # if opt.ch_list!=None:
            #     name = opt.ch_list[taski]
            # else:
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
    # if opt.ch_list != None:
    #     name = opt.ch_list[taski]
    # else:
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = build_arg(parser)

    arg = parser.parse_args()
    cfg = Config.fromfile(arg.config)

    opt={}
    opt.update(cfg.common)
    # opt.update(cfg.test)
    opt.update(cfg.model)
    opt.update(cfg.train)
    opt.update(cfg.optimizer)

    opt = argparse.Namespace(**opt)

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
