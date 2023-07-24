import os
import time

import torch
import torch.nn.init as init
import torch.utils.data
import numpy as np
from tqdm import tqdm
from tools.utils import CTCLabelConverter, AttnLabelConverter, Averager, adjust_learning_rate
from data.dataset import hierarchical_dataset
from modules.model import Model
from test import validation


class Label():
    def __init__(self,label,index,score):
        self.label = label
        self.index = index
        self.score = score
        # self.len_label = len(self.label)
    def __lt__(self, other):
        if self.score !=other.score:
            return self.score > other.score
        return len(self.label) < len(other.label)

class BaseLearner(object):
    def __init__(self, opt):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.opt = opt
        self.character = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.converter = None
        self.memory_index = []
        self._old_network = None
        # opt.num_class = self._total_classes
        self.model = Model(opt)


    def build_model(self):
        """ model configuration """

        # self.model.update_fc(self.opt.output_channel, self._total_classes)
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
        # return model

    def build_optimizer(self,filtered_parameters,scale=1.0):
        if self.opt.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filtered_parameters,
                lr=self.opt.lr * scale,
                momentum=self.opt.sgd_momentum,
                weight_decay=self.opt.sgd_weight_decay,
            )
        elif self.opt.optimizer == "adadelta":
            optimizer = torch.optim.Adadelta(
                filtered_parameters, lr=self.opt.lr * scale, rho=self.opt.rho, eps=self.opt.eps
            )
        elif self.opt.optimizer == "adam":
            optimizer = torch.optim.Adam(filtered_parameters, lr=self.opt.lr * scale)
        # print("optimizer:")
        # print(optimizer)
        self.optimizer = optimizer
        self.write_log(repr(optimizer) + "\n")

        if "super" in self.opt.schedule:
            if self.opt.optimizer == "sgd":
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
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.opt.milestones, gamma=self.opt.lrate_decay
            )
            self.scheduler = scheduler
            self.write_log(repr(scheduler) + "\n")
        # return scheduler,optimizer


    def build_converter(self):
        if "CTC" in self.opt.Prediction:
            converter = CTCLabelConverter(self.character)
        else:
            converter = AttnLabelConverter(self.character)
            self.sos_token_index = converter.dict["[SOS]"]
            self.eos_token_index = converter.dict["[EOS]"]
        self._total_classes = len(converter.character)
        return converter

    def build_criterion(self,reduction="mean"):
        """ setup loss """
        if "CTC" in self.opt.Prediction:
            criterion = torch.nn.CTCLoss(reduction=reduction,zero_infinity=True).to(self.device)
        else:
            # ignore [PAD] token
            criterion = torch.nn.CrossEntropyLoss(reduction=reduction,ignore_index=self.converter.dict["[PAD]"]).to(
                self.device
            )
        return criterion

    def change_model(self,):
        """ model configuration """
        # model.module.reset_class(opt, device)
        # self.model.update_fc(self.opt.output_channel, self._total_classes)
        self.model.update_fc(self.opt.hidden_size, self._total_classes)
        self.model.build_prediction(self.opt, self._total_classes)
        # reset_class(self.model.module, self.device)
        # data parallel for multi-GPU
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.train()
        # return self.model

    def after_task(self):
        self.model = self.model.module
        self._old_network = self.model.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self,taski, character, train_loader, valid_loader):

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

        # filter that only require gradient descent
        filtered_parameters = self.count_param()

        # setup optimizer
        self.build_optimizer(filtered_parameters)

        # print opt config
        # self.print_config(self.opt)
        if self.opt.start_task > taski:

            if taski > 0:
                if self.opt.memory != None:
                    self.build_rehearsal_memory(train_loader, taski)
                else:
                    train_loader.get_dataset(taski, memory=self.opt.memory)

            # if self.opt.ch_list!=None:
            #     name = self.opt.ch_list[taski]
            # else:
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
            start_iter = 0
            self._train(start_iter,taski, train_loader, valid_loader)

    def _train(self, start_iter,taski, train_loader, valid_loader):
        if taski == 0:
            self._init_train(start_iter,taski, train_loader, valid_loader)
        else:
            if self.opt.memory != None:
                self.build_rehearsal_memory(train_loader, taski)
            else:
                train_loader.get_dataset(taski, memory=self.opt.memory)
            self._update_representation(start_iter,taski, train_loader, valid_loader)


    def _init_train(self,start_iter,taski, train_loader, valid_loader):
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
                preds = self.model(image)["predict"]
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss = self.criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                preds = self.model(image, labels_index[:, :-1],True)["predict"]  # align with Attention.forward
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
            if iteration % self.opt.val_interval == 0 or iteration == 1:
                # for validation log
                self.val(valid_loader, self.opt,  best_score, start_time, iteration,
                    train_loss_avg, taski)
                train_loss_avg.reset()
                # semi_loss_avg.reset()

    def _update_representation(self,start_iter, taski, train_loader, valid_loader):
        self._init_train(start_iter, taski, train_loader, valid_loader)

    def build_rehearsal_memory(self,train_loader,taski):
        # Calculate the means of old classes with newly trained network
        memory_num = self.opt.memory_num
        num_i = int(memory_num / (taski))
        self.build_random_current_memory(num_i, taski, train_loader)
        if len(self.memory_index) != 0 and len(self.memory_index)*len(self.memory_index[0]) > memory_num:
            # if self.opt.memory == "rehearsal":
            #     self.reduce_div_samplers(taski, taski_num=num_i)
            # else:
            self.reduce_samplers(taski,taski_num =num_i)
        train_loader.get_dataset(taski,memory=self.opt.memory,index_list=self.memory_index)
        print("Is using rehearsal memory, has {} prev datasets, each has {}\n".format(len(self.memory_index),self.memory_index[0].size))


    def build_random_current_memory(self, taski_num, taski, train_loader):
        prev_loader, len_data = train_loader.rehearsal_prev_model(taski)
        index_list = np.random.choice(range(len_data), taski_num, replace=False)
        self.memory_index.append(index_list)

    def reduce_samplers(self,taski,taski_num):
        div = taski_num
        for i in range(taski):
            index = self.memory_index[i][:div]
            self.memory_index[i] = index
            print("----using memory {}".format(self.memory_index[i].size))

    def val(self, valid_loader, opt, best_score, start_time, iteration,
            train_loss_avg, taski):
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
            ) = validation(self.model, self.criterion, valid_loader, self.converter, opt,val_choose="val")
        self.model.train()

        # keep best score (accuracy or norm ED) model on valid dataset
        # Do not use this on test datasets. It would be an unfair comparison
        # (training should be done without referring test set).
        if current_score > best_score:
            best_score = current_score
            # if opt.ch_list != None:
            #     name = opt.ch_list[taski]
            # else:
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


    def test(self, AlignCollate_valid,valid_datas,best_scores,ned_scores,taski):
        print("---Start evaluation on benchmark testset----")
        """ keep evaluation model and result logs """
        os.makedirs(f"./result/{self.opt.exp_name}", exist_ok=True)
        os.makedirs(f"./evaluation_log", exist_ok=True)
        # if self.opt.ch_list != None:
        #     name = self.opt.ch_list[taski]
        # else:
        name = self.opt.lan_list[taski]
        saved_best_model = f"./saved_models/{self.opt.exp_name}/{name}_{taski}_best_score.pth"
        # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
        self.model.load_state_dict(torch.load(f"{saved_best_model}"))

        task_accs = []
        ned_accs = []
        for i,val_data in enumerate(valid_datas):
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
                ) = validation(self.model, self.criterion, valid_loader, self.converter, self.opt,val_choose="test")


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

    def double_write(self,taski, best_score):
        list17 = []
        list19 = []
        for i in range(taski+1):
            list17.append(best_score[i*2])
            list19.append(best_score[i*2+1])
        score17 = round(sum(list17) / len(list17), 2)
        score19 = round(sum(list19) / len(list19), 2)
        print(f'Task{taski} : 2017: {score17:0.2f} 2019: {score19:0.2f} \n ')
        self.write_data_log(f'Task{taski} : 2017: {score17:0.2f} 2019: {score19:0.2f}\n')
        self.write_data_log(f'17 acc: {list17}\n19 acc: {list19}\n')
        return score17,score19


    def count_param(self):
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print("Trainable params num: {:.2f} M\n".format(sum(params_num) / 1000000))
        print("Total paramerters: {:.2f} M\n".format(sum(x.numel() for x in self.model.parameters()) / 1e6))
        self.write_log("Trainable params num: {:.2f} M\n".format(sum(params_num) / 1000000))
        return filtered_parameters


    def print_config(self,opt):
        self_log = "------------ selfions -------------\n"
        args = vars(opt)
        for k, v in args.items():
            if str(k) == "character" and len(str(v)) > 500:
                self_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        self_log += "---------------------------------------\n"
        # print(self_log)
        self.write_log(self_log)

    def write_log(self,line):
        with open(f"./saved_models/{self.opt.exp_name}/log_train.txt", "a") as log:
            log.write(line)

    def write_data_log(self, line):
        with open(f"data_any.txt", "a+") as log:
            log.write(line)

