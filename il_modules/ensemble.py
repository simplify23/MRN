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
from modules.model import DERNet, Ensemble
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


class Ensem(BaseLearner):

    def __init__(self, opt):
        super().__init__(opt)
        self.model = Ensemble(opt)

    def after_task(self):
        # will we need this line ? (AB Study)
        self.model = self.model.module

        self._known_classes = self._total_classes
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
        self.model.build_prediction(self.opt, self._total_classes)
        # reset_class(self.model.module, self.device)
        # data parallel for multi-GPU
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.train()
        # return self.model

    def build_model(self):
        """ model configuration """

        self.model.update_fc(self.opt.hidden_size, self._total_classes)
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

    def incremental_train(self, taski, character, train_loader, valid_loader, opt):

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

        # filter that only require gradient descent
        filtered_parameters = self.count_param(self.model,False)

        # setup optimizer
        self.build_optimizer(filtered_parameters)
        
        if opt.start_task > taski:

            if taski > 0:
                train_loader.get_dataset(taski, memory=None)
                # valid_loader = valid_loader.create_dataset()

                # self.update_step1(0, taski, train_loader, valid_loader.create_dataset())
                if self.opt.memory != None:
                    self.build_rehearsal_memory(train_loader, taski)
                else:
                    train_loader.get_dataset(taski, memory=self.opt.memory)

            if opt.ch_list!=None:
                name = opt.ch_list[taski]
            else:
                name = opt.lan_list[taski]
            saved_best_model = f"./saved_models/{opt.exp_name}/{name}_{taski}_best_score.pth"
            # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
            self.model.load_state_dict(torch.load(f"{saved_best_model}"), strict=False)
            print(
            'Task {} load checkpoint from {}.'.format(taski, saved_best_model)
            )

        else:
            print(
            'Task {} start training.'.format(taski)
            )
            """ start training """
            self._train(0, taski, train_loader, valid_loader)


    def build_rehearsal_memory(self,train_loader,taski):
        # Calculate the means of old classes with newly trained network
        memory_num = self.opt.memory_num
        num_i = int(memory_num / (taski))
        # self.build_queue_bag_memory(num_i, taski, train_loader)
        self.build_random_current_memory(num_i, taski, train_loader)
        if len(self.memory_index) != 0 and len(self.memory_index)*len(self.memory_index[0]) > memory_num:
            self.reduce_samplers(taski,taski_num =num_i)
        train_loader.get_dataset(taski,memory=self.opt.memory,index_list=self.memory_index)
        print("Is using rehearsal memory, has {} prev datasets, each has {}\n".format(len(self.memory_index),self.memory_index[0].size))


    def _train(self, start_iter,taski, train_loader, valid_loader):
        if taski == 0:
            # valid_loader = valid_loader.create_dataset()
            self._init_train(start_iter,taski, train_loader, valid_loader.create_dataset(),cross=False)
        else:
            train_loader.get_dataset(taski, memory=None)
            # valid_loader = valid_loader.create_dataset()

            self.update_step1(start_iter,taski, train_loader, valid_loader.create_dataset())
            if self.opt.memory != None:
                self.build_rehearsal_memory(train_loader, taski)
            else:
                train_loader.get_dataset(taski, memory=self.opt.memory)
            self._update_representation(start_iter,taski, train_loader, valid_loader.create_list_dataset())
            # self.model.module.weight_align(self._total_classes - self._known_classes)

    def _init_train(self,start_iter,taski, train_loader, valid_loader,cross=False):
        # loss averager
        train_loss_avg = Averager()
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
                preds = self.model(image,cross)['logits']
                # preds = self.model(image)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                preds = self.model(image, labels_index[:, :-1])['logits']  # align with Attention.forward
                target = labels_index[:, 1:]  # without [SOS] Symbol
                loss = self.criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.grad_clip
            )  # gradient clipping with 5 (Default)
            self.optimizer.step()
            train_loss_avg.add(loss)

            if "super" in self.opt.schedule:
                self.scheduler.step()
            else:
                adjust_learning_rate(self.optimizer, iteration, self.opt)

            # validation part.
            # To see training progress, we also conduct validation when 'iteration == 1'
            if iteration % self.opt.val_interval == 0 or iteration ==self.opt.num_iter:
                # for validation log
                # print("66666666")
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, taski,"FF")
                train_loss_avg.reset()

    def update_step1(self,start_iter,taski, train_loader, valid_loader):
        self.model_eval_and_train(taski)
        self._init_train(start_iter, taski, train_loader, valid_loader,cross=False)
        self.model.train()
        for p in self.model.module.model[-1].parameters():
            p.requires_grad = False
        self.model.module.model[-1].eval()


    def _update_representation(self,start_iter, taski, train_loader, valid_loader,pi=5):
        # loss averager
        train_loss_avg = Averager()

        # train_taski_loss_avg = Averager()
        # loss_taski = nn.MSELoss()
        #
        # self.model_eval_and_train(taski)
        # self._init_train(start_iter, taski, train_loader, valid_loader)
        # filtered_parameters = self.count_param(self.model, False)
        self.criterion = self.build_criterion()
        filtered_parameters = self.count_param(self.model)

        # setup optimizer
        self.build_custom_optimizer(filtered_parameters,optimizer="adam",schedule="mlr",scale=1)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)


        start_time = time.time()
        best_score = -1

        # training loop
        for iteration in tqdm(
                range(start_iter + 1, int(self.opt.num_iter//2) + 1),
                total=int(self.opt.num_iter//2),
                position=0,
                leave=True,
        ):
            image_tensors, labels, indexs = train_loader.get_batch2()
            indexs = torch.LongTensor(indexs).squeeze().to(self.device)
            image = image_tensors.to(self.device)
            labels_index, labels_length = self.converter.encode(
                labels, batch_max_length=self.opt.batch_max_length
            )
            batch_size = image.size(0)

            # default recognition loss part
            if "CTC" in self.opt.Prediction:
                output= self.model(image,True)
                # [B,T,C,I], [B,I]
                preds = output["logits"]
                taski_loss = self.taski_criterion(output["index"],indexs)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # B，T，C(max) -> T, B, C
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss_clf = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                output = self.model(image, labels_index[:, :-1])  # align with Attention.forward
                preds = output["logits"]
                aux_logits = output["aux_logits"]
                aux_targets = labels_index.clone()[:, 1:]
                target = labels_index[:, 1:]  # without [SOS] Symbol
                loss_clf = self.criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )
                loss_aux = self.criterion(
                    aux_logits.view(-1, aux_logits.shape[-1]), aux_targets.contiguous().view(-1)
                )
            # loss = loss_clf + loss_aux
            loss = loss_clf + pi * taski_loss
            # loss.requires_grad_(True)
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.grad_clip
            )  # gradient clipping with 5 (Default)
            self.optimizer.step()
            train_loss_avg.add(loss)

            self.scheduler.step()
            # if "super" in self.opt.schedule:
            #     self.scheduler.step()
            # else:
            #     adjust_learning_rate(self.optimizer, iteration, self.opt)

            # validation part.
            # To see training progress, we also conduct validation when 'iteration == 1'
            if iteration % (self.opt.val_interval//5)== 0 or iteration == int(self.opt.num_iter//2) or iteration == 1:
                # for validation log
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, taski,"TF")
                train_loss_avg.reset()

    def val(self, valid_loader, opt, best_score, start_time, iteration,
            train_loss_avg, taski, val_choose="val"):
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
        self.write_log(valid_log + "\n")
        # self.write_data_log(
        #     f"Task {opt.lan_list[taski]} [{iteration}/{opt.num_iter}] : Score:{current_score:0.2f} LR:{lr:0.7f}\n")


    def test(self, AlignCollate_valid,valid_datas,best_scores,ned_scores,taski,val_choose="test"):
        print("---Start evaluation on benchmark testset----")
        if taski == 0:
            val_choose = "FF"
        else:
            val_choose = "TF"
        """ keep evaluation model and result logs """
        os.makedirs(f"./result/{self.opt.exp_name}", exist_ok=True)
        os.makedirs(f"./evaluation_log", exist_ok=True)
        if self.opt.ch_list != None:
            name = self.opt.ch_list[taski]
        else:
            name = self.opt.lan_list[taski]
        saved_best_model = f"./saved_models/{self.opt.exp_name}/{name}_{taski}_best_score.pth"
        # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
        self.model.load_state_dict(torch.load(f"{saved_best_model}"),strict=False)

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


        best_scores.append(round(sum(task_accs) / len(task_accs),2))
        ned_scores.append(round(sum(ned_accs) / len(ned_accs),2))

        acc_log= f'Task {taski} Test Average Incremental Accuracy: {best_scores[taski]} \n Task {taski} Incremental Accuracy: {task_accs}\n ned_acc: {ned_accs}\n'
        self.write_data_log(f'{taski} Avg Acc: {best_scores[taski]:0.2f} \n  acc: {task_accs}\n')

        print(acc_log)
        self.write_log(acc_log)
        return best_scores,ned_scores


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


