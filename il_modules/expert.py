import logging
import os
import time
import os

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch.autograd import Variable

from data.dataset import hierarchical_dataset
from il_modules.base import BaseLearner
from il_modules.der import DER
from modules.model import DERNet, Ensemble, Expert_Gate, Expert_Gatev2
from test import validation
from tools.utils import Averager, adjust_learning_rate

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class Expert(BaseLearner):

    def __init__(self, opt):
        super().__init__(opt)
        # self.model = Ensemblev2(opt)
        self.model = Expert_Gatev2(opt)
        self.criterion2 = nn.MSELoss()

    def after_task(self):
        # will we need this line ? (AB Study)
        self.model = self.model.module
        self._known_classes = self._total_classes
        self._old_network = self.model.copy().freeze()
        # logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def model_eval_and_train(self,taski):
        self.model.train()
        self.model.module.model[-1].train()
        if taski >= 1:
            for i in range(taski):
                self.model.module.model[i].eval()

    def build_custom_optimizer(self,filtered_parameters,optimizer="adam", schedule="super",scale=1.0):
        if optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filtered_parameters,
                lr=self.opt.lr * scale,
                momentum=self.opt.sgd_momentum,
                weight_decay=self.opt.sgd_weight_decay,
            )
        elif optimizer == "adadelta":
            optimizer = torch.optim.Adadelta(
                filtered_parameters, lr=self.opt.lr * scale, rho=self.opt.rho, eps=self.opt.eps
            )
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(filtered_parameters, lr=self.opt.lr * scale)
        # print("optimizer:")
        # print(optimizer)
        self.optimizer = optimizer
        self.write_log(repr(optimizer) + "\n")

        if "super" in schedule:
            if optimizer == "sgd":
                cycle_momentum = True
            else:
                cycle_momentum = False

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.opt.lr * scale,
                cycle_momentum=cycle_momentum,
                div_factor=20,
                final_div_factor=1000,
                total_steps=self.opt.num_iter,
            )
            # print("Scheduler:")
            # print(scheduler)
            self.scheduler = scheduler
            self.write_log(repr(scheduler) + "\n")
        elif schedule == "mlr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.opt.milestones, gamma=self.opt.lrate_decay
            )
            self.scheduler = scheduler
            self.write_log(repr(scheduler) + "\n")

    def change_model(self,):
        """ model configuration """
        # model.module.reset_class(opt, device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        self.model.update_fc(self.opt.hidden_size, self._total_classes)
        # self.model.update_fc(self.opt.output_channel, self._total_classes)
        self.model.build_prediction(self.opt, self._total_classes)
        # reset_class(self.model.module, self.device)
        # data parallel for multi-GPU
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.train()
        # return self.model

    def build_model(self):
        """ model configuration """

        self.model.build_fc(self.opt.hidden_size, self._total_classes)
        # self.model.build_fc(self.opt.output_channel, self._total_classes)
        self.model.build_prediction(self.opt, self._total_classes)

        # weight initialization
        for name, param in self.model.named_parameters():
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
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.train()

    def incremental_train(self, taski, character, train_loader, valid_loader):

        # pre task classes for know classes
        # self._known_classes = self._total_classes
        self.character = character
        self.converter = self.build_converter()

        if taski > 0:
            self.change_model()
        else:
            self.criterion = self.build_criterion()
            self.build_model()

        # ignore [PAD] token
        self.taski_criterion = torch.nn.CrossEntropyLoss(reduction="mean").to(
            self.device
        )
        # self.taski_criterion = FocalLoss().to(self.device)

        if taski > 0:
            for i in range(taski):
                for p in self.model.module.model[i].parameters():
                    p.requires_grad = False
                for p in self.model.module.gate[i].parameters():
                    p.requires_grad = False

        # filter that only require gradient descent
        filtered_parameters = self.count_param(self.model,False)

        # setup optimizer
        self.build_optimizer(filtered_parameters)

        self._train(0, taski, train_loader, valid_loader,step=0)
        # if taski >0:
        #     self._train(0, taski, train_loader, valid_loader, step=1)


    def build_rehearsal_memory(self,train_loader,taski):
        # Calculate the means of old classes with newly trained network
        memory_num = self.opt.memory_num
        if memory_num >= 5000:
            num_i = memory_num
        else:
            num_i = int(memory_num / (taski))
        # self.build_queue_bag_memory(num_i, taski, train_loader)
        if self.opt.memory == "rehearsal" or self.opt.memory == "loss_max" or self.opt.memory == "cof_max":
            self.build_current_memory(num_i, taski, train_loader)
        elif self.opt.memory == "bag":
            self.build_queue_bag_memory(num_i, taski, train_loader)
        elif self.opt.memory == "score":
            self.dataset_label_score(num_i, taski, train_loader)
        else:
            self.build_random_current_memory(num_i, taski, train_loader)
        if memory_num < 5000:
            if len(self.memory_index) != 0 and len(self.memory_index)*len(self.memory_index[0]) > memory_num:
                self.reduce_samplers(taski,taski_num =num_i)
        train_loader.get_dataset(taski,memory=self.opt.memory,index_list=self.memory_index)
        print("Is using rehearsal memory, has {} prev datasets, each has {}\n".format(len(self.memory_index),self.memory_index[0].size))


    def _train(self, start_iter,taski, train_loader, valid_loader,step=0):

        if self.opt.start_task > taski :
            name = self.opt.lan_list[taski]
            saved_best_model = f"./saved_models/{self.opt.exp_name}/{name}_{taski}_best_score.pth"
            self.model.load_state_dict(torch.load(f"{saved_best_model}"), strict=True)
            print(
                'Task {} load checkpoint from {}.'.format(taski, saved_best_model)
            )

            if taski > 0 and step == 0:
                train_loader.get_dataset(taski, memory=None)
                self.freeze_step1(taski)
                # self.update_step1(0, taski, train_loader, valid_loader.create_dataset())
            elif taski > 0 and step ==1:
                if self.opt.memory != None:
                    self.build_rehearsal_memory(train_loader, taski)
                else:
                    train_loader.get_dataset(taski, memory=self.opt.memory)

        else:
            print(
                'Task {} start training for model ------{}------'.format(taski, self.opt.exp_name)
            )
            """ start training """
            self._init_train(start_iter, taski, train_loader, valid_loader.create_dataset(), cross=False)
            # treadness = 1.0
            # self.train_autoencoder(start_iter, taski, train_loader, valid_loader.create_dataset(), cross=False)
            # self.freeze_step1(taski)
            # if treadness>0.85:
            #     self._init_train(start_iter, taski, train_loader, valid_loader.create_dataset(), cross=False)
            # else:
            #     self._update_representation(start_iter, taski, train_loader, valid_loader.create_dataset(), cross=True)
            # if taski == 0:
            #     # valid_loader = valid_loader.create_dataset()
            #     self._init_train(start_iter,taski, train_loader, valid_loader.create_dataset(),cross=False)
            # elif step == 0:
            #     train_loader.get_dataset(taski, memory=None)
            #     # valid_loader = valid_loader.create_dataset()
            #     self.update_step1(start_iter,taski, train_loader, valid_loader.create_dataset())
            # elif step == 1:
            #     if self.opt.memory != None:
            #         self.build_rehearsal_memory(train_loader, taski)
            #     else:
            #         train_loader.get_dataset(taski, memory=self.opt.memory)
            #     self._update_representation(start_iter,taski, train_loader, valid_loader.create_list_dataset())
                # self.model.module.weight_align(self._total_classes - self._known_classes)

    def _init_train(self,start_iter,taski, train_loader, valid_loader,cross=False):
        # fine-tuning
        # loss averager
        train_loss_avg = Averager()
        train_taski_loss_avg = Averager()
        start_time = time.time()
        best_score = -1

        # training loop
        for iteration in tqdm(
                range(start_iter + 1, self.opt.num_iter + 1),
                total=self.opt.num_iter,
                position=0,
                leave=True,
        ):
            image_tensors, labels = train_loader.get_batch()

            image = image_tensors.to(self.device)
            labels_index, labels_length = self.converter.encode(
                labels, batch_max_length=self.opt.batch_max_length
            )
            batch_size = image.size(0)

            # default recognition loss part
            if "CTC" in self.opt.Prediction:
                output = self.model(image,cross)
                preds = output["logits"]
                taski_loss = self.criterion2(image,output["gate_feature"])
                # preds = self.model(image)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                output = self.model(image, cross,labels_index[:, :-1])  # align with Attention.forward
                preds = output["logits"]
                taski_loss = self.criterion2(image, output["gate_feature"])
                # taski_loss = output["loss"]
                target = labels_index[:, 1:]  # without [SOS] Symbol
                loss = self.criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )
            # loss = loss + taski_loss + taski_loss2
            loss = loss + 10 * taski_loss
            self.model.zero_grad()
            loss.backward()
            # taski_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.grad_clip
            )  # gradient clipping with 5 (Default)
            self.optimizer.step()
            train_loss_avg.add(loss)
            train_taski_loss_avg.add(taski_loss)

            if "super" in self.opt.schedule:
                self.scheduler.step()
            else:
                adjust_learning_rate(self.optimizer, iteration, self.opt)

            # validation part.
            # To see training progress, we also conduct validation when 'iteration == 1'
            if iteration % self.opt.val_interval == 0 or iteration ==int(self.opt.num_iter)-1:
                # for validation log
                # print("66666666")
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, train_taski_loss_avg, taski,0,"FF")
                train_loss_avg.reset()
                # train_taski_loss_avg.reset()

    def update_step1(self,start_iter,taski, train_loader, valid_loader):
        self.model_eval_and_train(taski)
        self._init_train(start_iter, taski, train_loader, valid_loader,cross=False)
        self.model.train()
        for p in self.model.module.model[-1].parameters():
            p.requires_grad = False
        self.model.module.model[-1].eval()

    def freeze_step1(self,  taski):
        self.model_eval_and_train(taski)
        # self._init_train(start_iter, taski, train_loader, valid_loader, cross=False)
        self.model.train()
        for p in self.model.module.model[-1].parameters():
            p.requires_grad = False
        self.model.module.model[-1].eval()

    def _update_representation(self,start_iter, taski, train_loader, valid_loader, cross=False):
        # kd loss
        # loss averager
        train_loss_avg = Averager()
        # train_taski_loss_avg = Averager()
        start_time = time.time()
        best_score = -1

        filtered_parameters = self.count_param(self.model)

        # setup optimizer
        self.build_custom_optimizer(filtered_parameters,optimizer="adam",schedule="super",scale=1)

        # training loop
        for iteration in tqdm(
                range(start_iter + 1, int(self.opt.num_iter) + 1),
                total=int(self.opt.num_iter),
                position=0,
                leave=True,
        ):
            image_tensors, labels = train_loader.get_batch()

            image = image_tensors.to(self.device)
            labels_index, labels_length = self.converter.encode(
                labels, batch_max_length=self.opt.batch_max_length
            )
            batch_size = image.size(0)

            # default recognition loss part
            if "CTC" in self.opt.Prediction:
                start_index = 0
                preds = self.model(image)["logits"]
                old_preds = self._old_network(image)["logits"]
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # B，T，C(max) -> T, B, C
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss_clf = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                start_index = 1
                preds = self.model(image, labels_index[:, :-1], True)["logits"]  # align with Attention.forward
                old_preds = self._old_network(image, labels_index[:, :-1], True)["logits"]
                target = labels_index[:, 1:]  # without [SOS] Symbol
                loss_clf = self.criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

            # fake_targets = self._total_classes - self._known_classes
            # loss_clf = F.cross_entropy(
            #     preds_log_softmax[:, self._known_classes:], fake_targets
            # )
            loss_kd = _KD_loss(
                preds.view(-1, preds.shape[-1])[:, start_index: self._known_classes],
                old_preds.view(-1, old_preds.shape[-1])[:, start_index: self._known_classes],
                T,
            )
            loss = loss_kd + loss_clf
            self.model.zero_grad()
            loss.backward()
            # taski_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.grad_clip
            )  # gradient clipping with 5 (Default)
            self.optimizer.step()
            train_loss_avg.add(loss)
            # train_taski_loss_avg.add(taski_loss)

            if "super" in self.opt.schedule:
                self.scheduler.step()
            else:
                adjust_learning_rate(self.optimizer, iteration, self.opt)

            # validation part.
            # To see training progress, we also conduct validation when 'iteration == 1'
            if iteration % self.opt.val_interval == 0 or iteration == self.opt.num_iter:
                # for validation log
                # print("66666666")
                self.val(valid_loader, self.opt, best_score, start_time, iteration,
                         train_loss_avg, None, taski, 0, "FF")
                train_loss_avg.reset()
                # train_taski_loss_avg.reset()

    def val(self, valid_loader, opt, best_score, start_time, iteration,
            train_loss_avg, train_taski_loss_avg, taski, step,val_choose="val"):
        self.model.eval()
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
            ) = validation(self.model, self.criterion, valid_loader, self.converter, opt,val_choose=val_choose)
        self.model.train()

        # keep best score (accuracy or norm ED) model on valid dataset
        # Do not use this on test datasets. It would be an unfair comparison
        # (training should be done without referring test set).
        if current_score > best_score:
            best_score = current_score
            if opt.ch_list != None:
                name = opt.ch_list[taski]
            else:
                name = opt.lan_list[taski]
            torch.save(
                self.model.state_dict(),
                f"./saved_models/{opt.exp_name}/{name}_{taski}_best_score.pth",
            )

        # validation log: loss, lr, score (accuracy or norm ED), time.
        lr = self.optimizer.param_groups[0]["lr"]
        elapsed_time = time.time() - start_time
        valid_log = f"\n[{iteration}/{opt.num_iter}] Train_loss_clf: {train_loss_avg.val():0.5f}, Valid_loss: {valid_loss:0.5f} \n "
        if train_taski_loss_avg != None:
            valid_log += f'{"":9s}Train_taski_loss: {train_taski_loss_avg.val():0.5f}\n'
        # valid_log += f", Semi_loss: {semi_loss_avg.val():0.5f}\n"
        valid_log += f'{"":9s}Current_score: {current_score:0.2f}, Ned_score: {ned_score:0.2f}\n'
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
        self.write_log(valid_log + "\n")
        # self.write_data_log(
        #     f"Task {opt.lan_list[taski]} [{iteration}/{opt.num_iter}] : Score:{current_score:0.2f} LR:{lr:0.7f}\n")


    def test(self, AlignCollate_valid,valid_datas,best_scores,ned_scores,taski,val_choose="test"):
        print("---Start evaluation on benchmark testset----")
        if taski == 0:
            val_choose = "FF"
            step=0
        else:
            val_choose = "TF"
            step=1
        """ keep evaluation model and result logs """
        os.makedirs(f"./result/{self.opt.exp_name}", exist_ok=True)
        os.makedirs(f"./evaluation_log", exist_ok=True)
        if self.opt.ch_list != None:
            name = self.opt.ch_list[taski]
        else:
            name = self.opt.lan_list[taski]
        saved_best_model = f"./saved_models/{self.opt.exp_name}/{name}_{taski}_best_score.pth"
        # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
        self.model.load_state_dict(torch.load(f"{saved_best_model}"),strict=True)

        task_accs = []
        ned_accs = []
        for val_data in valid_datas:
            valid_dataset, valid_dataset_log = hierarchical_dataset(
                root=val_data, opt=self.opt, mode="test")
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.opt.batch_size,
                shuffle=True,  # 'True' to check training progress with validation function.
                num_workers=int(self.opt.workers),
                collate_fn=AlignCollate_valid,
                pin_memory=False,
            )

            self.model.eval()
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
                ) = validation(self.model, self.criterion, valid_loader, self.converter, self.opt,val_choose=val_choose)


            task_accs.append(round(current_score,2))
            ned_accs.append(round(ned_score,2))


        self.write_data_log(f"----------- {self.opt.exp_name} Task {taski}------------\n")

        if (taski+1) * 2 == len(task_accs):
            score17,score19 = self.double_write(taski,task_accs)
            best_scores.append(score17)
            ned_scores.append(score19)
            acc_log = f'Task {taski} Avg Incremental Acc:  17: {best_scores[taski]}    19: {ned_scores[taski]}\n Task {taski} 17 Acc: {score17}\n 19 Acc: {score19}\n'
            self.write_log(acc_log)
            print(acc_log)
        else:
            best_scores.append(round(sum(task_accs) / len(task_accs), 2))
            ned_scores.append(round(sum(ned_accs) / len(ned_accs), 2))
            acc_log = f'Task {taski} Test Average Incremental Accuracy: {best_scores[taski]} \n Task {taski} Incremental Accuracy: {task_accs}\n ned_acc: {ned_accs}\n'
            self.write_log(acc_log)
            print(acc_log)
            self.write_data_log(f'{taski} Avg Acc: {best_scores[taski]:0.2f} \n  acc: {task_accs}\n')
        return best_scores,ned_scores


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


