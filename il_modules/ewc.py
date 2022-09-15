import logging
import time

import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F

from il_modules.base import BaseLearner
from tools.utils import Averager, adjust_learning_rate

init_epoch=200
init_lr=0.1
init_milestones=[60,120,170]
init_lr_decay=0.1
init_weight_decay=0.0005



epochs = 180
lrate = 0.1
milestones = [70, 120,150]
lrate_decay = 0.1
batch_size = 128
weight_decay=2e-4
num_workers=4
T=2
lamda=1000
fishermax=0.0001
alpha = 0.5
num_iter = 1000

class EWC(BaseLearner):
    def __init__(self, opt):
        super().__init__(opt)
        self.fisher = None

    def after_task(self):
        self.model = self.model.module
        # self._old_network = self.model.copy().freeze()
        self._known_classes = self._total_classes

    def _train(self, start_iter, taski, train_loader, valid_loader):
        if taski == 0:
            self._init_train(start_iter,taski, train_loader, valid_loader)
        else:
            if self.opt.memory == "rehearsal" or self.opt.memory == "random":
                self.build_rehearsal_memory(train_loader, taski)
            else:
                train_loader.get_dataset(taski, memory=self.opt.memory)
            self._update_representation(start_iter,taski, train_loader, valid_loader)
            # self._update_representation(start_iter,taski, train_loader, valid_loader)
        if self.fisher is None:
            self.fisher=self.getFisherDiagonal(train_loader)
        else:
            # alpha=self._known_classes/self._total_classes
            new_finsher=self.getFisherDiagonal(train_loader)
            f_list = list(self.fisher.values())
            for i,(n,p) in enumerate(new_finsher.items()):
                new_finsher[n][:len(f_list[i])] = alpha * f_list[i] + (1 - alpha) * new_finsher[n][:len(f_list[i])]
                # new_finsher[n][:len(self.fisher[n])]=alpha*self.fisher[n]+(1-alpha)*new_finsher[n][:len(self.fisher[n])]
            self.fisher=new_finsher
        self.mean={n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def _update_representation(self,start_iter, taski, train_loader, valid_loader):
        # loss averager
        train_loss_avg = Averager()
        # semi_loss_avg = Averager()

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
                preds = self.model(image)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # B，T，C(max) -> T, B, C
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss_clf = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                preds = self.model(image, labels_index[:, :-1])  # align with Attention.forward
                target = labels_index[:, 1:]  # without [SOS] Symbol
                loss_clf = self.criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

            loss_ewc = self.compute_ewc()
            loss = loss_clf + lamda * loss_ewc

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
            if iteration % self.opt.val_interval == 0 or iteration == 1:
                # for validation log
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, taski)
                train_loss_avg.reset()


    def compute_ewc(self):
        loss = 0
        # if len(self._multiple_gpus) > 1:
        for n, p in self.model.module.named_parameters():
            if n in self.fisher.keys():
                loss += torch.sum((self.fisher[n]) * (p[:len(self.mean[n])] - self.mean[n]).pow(2)) / 2
        return loss

    def getFisherDiagonal(self,train_loader):
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        self.model.train()
        # optimizer = optim.SGD(self.model.parameters(),lr=lrate)
        for iteration in tqdm(
                range( 1, num_iter + 1),
                total= num_iter,
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
                preds = self.model(image)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # B，T，C(max) -> T, B, C
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                preds = self.model(image, labels_index[:, :-1])  # align with Attention.forward
                target = labels_index[:, 1:]  # without [SOS] Symbol
                loss= self.criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n,p in fisher.items():
            fisher[n]=p/num_iter
            fisher[n]=torch.min(fisher[n],torch.tensor(fishermax))
        return fisher
