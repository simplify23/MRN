import logging
import time

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
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
        self.model = self.model.module
        self._known_classes = self._total_classes
        # logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def model_eval_and_train(self,taski):
        self.model.train()
        self.model.module.model[-1].train()
        if taski >= 1:
            for i in range(taski):
                self.model.module.model[i].eval()

    def change_model(self,):
        """ model configuration """
        # model.module.reset_class(opt, device)
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

        if taski > 0:
            for i in range(taski):
                for p in self.model.module.model[i].parameters():
                    p.requires_grad = False

        # filter that only require gradient descent
        filtered_parameters = self.count_param(self.model,False)

        # setup optimizer
        self.build_optimizer(filtered_parameters)

        """ start training """
        self._train(0, taski, train_loader, valid_loader)

    def build_rehearsal_memory(self,train_loader,taski):
        # Calculate the means of old classes with newly trained network
        memory_num = self.opt.memory_num
        num_i = int(memory_num / (taski))
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
            if iteration % self.opt.val_interval == 0 or iteration ==1:
                # for validation log
                # print("66666666")
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, taski)
                train_loss_avg.reset()

    def update_step1(self,start_iter,taski, train_loader, valid_loader):
        self.model_eval_and_train(taski)
        self._init_train(start_iter, taski, train_loader, valid_loader,cross=False)
        self.model.train()
        for p in self.model.module.model[-1].parameters():
            p.requires_grad = False
        self.model.module.model[-1].eval()


    def _update_representation(self,start_iter, taski, train_loader, valid_loader,pi=1):
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
        self.build_optimizer(filtered_parameters,scale=1)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)


        start_time = time.time()
        best_score = -1

        # training loop
        for iteration in tqdm(
                range(start_iter + 1, int(self.opt.num_iter//5) + 1),
                total=int(self.opt.num_iter//5),
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

            if "super" in self.opt.schedule:
                self.scheduler.step()
            else:
                adjust_learning_rate(self.optimizer, iteration, self.opt)

            # validation part.
            # To see training progress, we also conduct validation when 'iteration == 1'
            if iteration % (self.opt.val_interval//10)== 0 or iteration == 1:
                # for validation log
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, taski)
                train_loss_avg.reset()




