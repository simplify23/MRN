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
from modules.model import DERNet
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


class DER(BaseLearner):

    def __init__(self, opt):
        super().__init__(opt)
        self.model = DERNet(opt)

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
        # self.model.update_fc(self.opt.output_channel, self._total_classes)
        self.model.update_fc(self.opt.hidden_size, self._total_classes)
        self.model.build_prediction(self.opt, self._total_classes)
        self.model.build_aux_prediction(self.opt, self._total_classes)
        # reset_class(self.model.module, self.device)
        # data parallel for multi-GPU
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.train()
        # return self.model

    def build_model(self):
        """ model configuration """
        # self.model.update_fc(self.opt.output_channel, self._total_classes)
        self.model.update_fc(self.opt.hidden_size, self._total_classes)
        self.model.build_prediction(self.opt, self._total_classes)
        self.model.build_aux_prediction(self.opt, self._total_classes)

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
        valid_loader = valid_loader.create_dataset()

        if taski > 0:
            self.change_model()
        else:
            self.criterion = self.build_criterion()
            self.build_model()

        # print opt config
        # self.print_config(self.opt)
        if taski > 0:
            for i in range(taski):
                for p in self.model.module.model[i].parameters():
                    p.requires_grad = False

        # filter that only require gradient descent
        filtered_parameters = self.count_param(self.model,False)

        # setup optimizer
        self.build_optimizer(filtered_parameters)

        if self.opt.start_task > taski:

            if taski > 0:
                if self.opt.memory != None:
                    self.build_rehearsal_memory(train_loader, taski)
                else:
                    train_loader.get_dataset(taski, memory=self.opt.memory)

            if self.opt.ch_list!=None:
                name = self.opt.ch_list[taski]
            else:
                name = self.opt.lan_list[taski]
            saved_best_model = f"./saved_models/{self.opt.exp_name}/{name}_{taski}_best_score.pth"
            # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
            self.model.load_state_dict(torch.load(f"{saved_best_model}"), strict=True)
            print(
            'Task {} load checkpoint from {}.'.format(taski, saved_best_model)
            )

        else:
            print(
            'Task {} start training for model ------{}------'.format(taski,self.opt.exp_name)
            )
            """ start training """
            self._train(0, taski, train_loader, valid_loader)


    def _train(self, start_iter,taski, train_loader, valid_loader):
        if taski == 0:
            self._init_train(start_iter,taski, train_loader, valid_loader)
        else:
            if self.opt.memory != None:
                self.build_rehearsal_memory(train_loader, taski)
            else:
                train_loader.get_dataset(taski, memory=self.opt.memory)
            self._update_representation(start_iter,taski, train_loader, valid_loader)
            self.model.module.weight_align(self._total_classes - self._known_classes)

    def _init_train(self,start_iter,taski, train_loader, valid_loader):
        # loss averager
        train_loss_avg = Averager()
        train_clf_loss = Averager()
        train_aux_loss = Averager()
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
                preds = self.model(image)['logits']
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
                    train_loss_avg, train_clf_loss, train_aux_loss, taski)
                train_loss_avg.reset()

    def _update_representation(self,start_iter, taski, train_loader, valid_loader):
        # loss averager
        train_loss_avg = Averager()
        train_clf_loss = Averager()
        train_aux_loss = Averager()

        self.model_eval_and_train(taski)


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
                output = self.model(image)
                preds = output["logits"]
                aux_logits = output["aux_logits"]
                aux_targets = labels_index.clone()
                # aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
                #                           aux_targets - self._known_classes + 1, 0)

                aux_preds_size = torch.IntTensor([aux_logits.size(1)] * batch_size)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # B，T，C(max) -> T, B, C
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                aux_preds_log_softmax = aux_logits.log_softmax(2).permute(1, 0, 2)

                loss_clf = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
                loss_aux = self.criterion(aux_preds_log_softmax, aux_targets, aux_preds_size, labels_length)
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
            loss = loss_clf + loss_aux
            # loss = loss_clf

            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.grad_clip
            )  # gradient clipping with 5 (Default)
            self.optimizer.step()
            train_loss_avg.add(loss)
            train_clf_loss.add(loss_clf)
            train_aux_loss.add(loss_aux)

            if "super" in self.opt.schedule:
                self.scheduler.step()
            else:
                adjust_learning_rate(self.optimizer, iteration, self.opt)

            # validation part.
            # To see training progress, we also conduct validation when 'iteration == 1'
            if iteration % self.opt.val_interval == 0 or iteration == 1:
                # for validation log
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, train_clf_loss, train_aux_loss, taski)
                train_loss_avg.reset()
                train_clf_loss.reset()
                train_aux_loss.reset()

    def val(self, valid_loader, opt, best_score, start_time, iteration,
            train_loss_avg,train_clf_loss, train_aux_loss, taski):
        self.model.eval()
        start_time = time.time()
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
            ) = validation(self.model, self.criterion, valid_loader, self.converter, opt)
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
        if train_clf_loss !=None:
            valid_log += f"CLF_loss: {train_clf_loss.val():0.5f} , Aux_loss: {train_aux_loss.val():0.5f}\n"
        valid_log += f'{"":9s}Current_score: {current_score:0.2f},   Ned_score: {ned_score:0.2f}\n'
        valid_log += f'{"":9s}Current_lr: {lr:0.7f}, Best_score: {best_score:0.2f}\n'
        valid_log += f'{"":9s}Infer_time: {infer_time:0.2f},     Elapsed_time: {elapsed_time/length_of_data:0.4f}\n'

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
