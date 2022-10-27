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


EPSILON = 1e-8


init_epoch=200
init_lr=0.1
init_milestones=[60,120,170]
init_lr_decay=0.1
init_weight_decay=0.0005


epochs = 170
lrate = 0.1
milestones = [60, 100,140]
lrate_decay = 0.1
batch_size = 128
weight_decay=2e-4
num_workers=8
T=2


class WA(BaseLearner):
    def __init__(self, opt):
        super().__init__(opt)
        self.taski = 0

    def after_task(self):
        if self.taski >0:
            self.model.module.weight_align(self._total_classes-self._known_classes)
        self.model = self.model.module
        self._old_network = self.model.copy().freeze()
        self._known_classes = self._total_classes

    def _update_representation(self,start_iter, taski, train_loader, valid_loader):
        self.taski = taski
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
                start_index = 0
                preds = self.model(image)["predict"]
                old_preds = self._old_network(image)["predict"]
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # B，T，C(max) -> T, B, C
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss_clf = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                start_index = 1
                preds = self.model(image, labels_index[:, :-1],True)["predict"]  # align with Attention.forward
                old_preds = self._old_network(image, labels_index[:, :-1],True)["predict"]
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

            loss=loss_clf+ 2*loss_kd

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
        self.model.module.weight_align(self._total_classes - self._known_classes)

def _KD_loss(pred, soft, T):
   pred = torch.log_softmax(pred / T, dim=1)
   soft = torch.softmax(soft / T, dim=1)
   return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
