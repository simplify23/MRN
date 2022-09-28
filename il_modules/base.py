import os
import time
from collections import defaultdict
from queue import Queue,PriorityQueue

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import numpy as np
from mmcv import Config
from tqdm import tqdm

from tools.utils import CTCLabelConverter, AttnLabelConverter, Averager, adjust_learning_rate
from data.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modules.model import Model
from test import validation, benchmark_all_eval
class bag_value():
    def __init__(self,bag):
        self.label, = bag
        self.bag = bag
        self.index, = bag.values()
        self.len_label = len(self.label)
    def __lt__(self, other):
        if self.len_label !=other.len_label:
            return self.len_label < other.len_label
        return self.label > other.label

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
        filtered_parameters = self.count_param(self.model)

        # setup optimizer
        self.build_optimizer(filtered_parameters)

        # print opt config
        # self.print_config(self.opt)

        """ start training """
        start_iter = 0
        if self.opt.saved_model != "":
            try:
                start_iter = int(self.saved_model.split("_")[-1].split(".")[0])
                print(f"continue to train, start_iter: {start_iter}")
            except:
                pass

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
        semi_loss_avg = Averager()

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
                preds = self.model(image, labels_index[:, :-1])  # align with Attention.forward
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
                semi_loss_avg.reset()

    def _update_representation(self,start_iter, taski, train_loader, valid_loader):
        self._init_train(start_iter, taski, train_loader, valid_loader)

    def build_rehearsal_memory(self,train_loader,taski):
        # Calculate the means of old classes with newly trained network
        memory_num = self.opt.memory_num
        num_i = int(memory_num / (taski))
        if self.opt.memory == "rehearsal" or self.opt.memory == "loss_max" or self.opt.memory == "cof_max":
            self.build_current_memory(num_i,taski,train_loader)
        elif self.opt.memory == "bag":
            self.build_queue_bag_memory(num_i, taski, train_loader)
        elif self.opt.memory == "score":
            self.dataset_label_score(num_i, taski, train_loader)
        else:
            self.build_random_current_memory(num_i, taski, train_loader)
        if len(self.memory_index) != 0 and len(self.memory_index)*len(self.memory_index[0]) > memory_num:
            if self.opt.memory == "rehearsal":
                self.reduce_div_samplers(taski, taski_num=num_i)
            else:
                self.reduce_samplers(taski,taski_num =num_i)
        train_loader.get_dataset(taski,memory=self.opt.memory,index_list=self.memory_index)
        print("Is using rehearsal memory, has {} prev datasets, each has {}\n".format(len(self.memory_index),self.memory_index[0].size))


    def build_random_current_memory(self, taski_num, taski, train_loader):
        prev_loader, len_data = train_loader.rehearsal_prev_model(taski)
        index_list = np.random.choice(range(len_data), taski_num, replace=False)
        self.memory_index.append(index_list)

    def build_current_memory(self, taski_num, taski, train_loader):
        prev_loader, len_data = train_loader.rehearsal_prev_model(taski)
        # criterion = self.build_criterion("none")
        loss = []
        seq = []
        for i, (image_tensors, labels) in enumerate(prev_loader):
            image = image_tensors.to(self.device)
            # labels_index, labels_length = self.converter.encode(
            #     labels, batch_max_length=self.opt.batch_max_length
            # )
            # batch_size = image.size(0)
            preds = self._old_network(image)
            preds_list = torch.max(preds,-1)[0].cpu()
            for pred_seq in preds_list:
                cof = 0.0
                for ch_cof in pred_seq:
                    # cof *=ch_cof
                    cof += ch_cof
                seq.append(cof/len(pred_seq))
            # default recognition loss part
        #     if "CTC" in self.opt.Prediction:
        #         preds = self._old_network(image)
        #         preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        #         # B，T，C(max) -> T, B, C
        #         preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
        #         loss_clf = criterion(preds_log_softmax, labels_index, preds_size, labels_length)
        #     else:
        #         preds = self.model(image, labels_index[:, :-1])  # align with Attention.forward
        #         target = labels_index[:, 1:]  # without [SOS] Symbol
        #         loss_clf = criterion(
        #             preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
        #         )
        #     loss.append(loss_clf.cpu())
        # loss = torch.cat(loss)
        max_v, max_i = torch.topk(torch.Tensor(seq), k=int(taski_num), sorted=True, largest=True)
        # min_v, min_i = torch.topk(loss, k=int(taski_num/2), sorted=True, largest=False)
        # max_v, max_i = torch.topk(loss, k=int(taski_num), sorted=True, largest=True)
        # index = torch.cat([max_i,min_i]).numpy(),0)
        # self.memory_index.append(torch.cat([max_i,min_i]).numpy())
        self.memory_index.append((max_i).numpy())

    def reduce_div_samplers(self,taski,taski_num):
        div = taski_num//2
        for i in range(taski):
            list_num = len(self.memory_index[i])
            maxi = self.memory_index[i][:div]
            mini = self.memory_index[i][list_num//2:list_num//2 + div]
            self.memory_index[i] = np.concatenate([maxi,mini],-1)
            print("----using memory {}".format(self.memory_index[i].size))

    def reduce_samplers(self,taski,taski_num):
        div = taski_num
        for i in range(taski):
            index = self.memory_index[i][:div]
            self.memory_index[i] = index
            print("----using memory {}".format(self.memory_index[i].size))

    def dataset_label_score(self, taski_num, taski, train_loader):
        prev_dataset, len_data = train_loader.rehearsal_prev_dataset(taski)
        char = {}
        # queue = PriorityQueue()
        # max_length = 0
        index_array = []
        for index in range(len_data):
            (image_tensor, label) = prev_dataset[index]
            for ch in label:
                if char.get(ch, None) == None:
                    char[ch] = 1
                else:
                    char[ch] +=1
        # print(char)
        for index in range(len_data):
            (image_tensor, label) = prev_dataset[index]
            labels_length = len(label)
            if labels_length == 0:
                continue
            label_score = 0.0
            for ch in label:
                if char.get(ch, None) != None:
                    label_score += pow(char[ch],-1.5)
            label_score = label_score / labels_length
            index_array.append(Label(label,index,label_score))

        # queue = [Queue() for i in range(max_length)]
        index_array = sorted(index_array)[:taski_num]
        data_list = [label_c.index for label_c in index_array]
        print("samples get array {}--------".format(len(data_list)))
        self.memory_index.append(np.array(data_list))


    def build_queue_bag_memory(self, taski_num, taski, train_loader):
        prev_dataset, len_data = train_loader.rehearsal_prev_dataset(taski)
        data_len = defaultdict(list)
        char = {}
        max_length = 0
        index_array = []
        for index in range(len_data):
            (image_tensors, labels) = prev_dataset[index]
            # labels_index, labels_length = self.converter.encode(
            #     labels, batch_max_length=self.opt.batch_max_length
            # )
            labels_length = len(labels)
            data_len[labels_length].append({labels:index})
            if labels_length > max_length:
                max_length = labels_length
        # queue = [Queue() for i in range(max_length)]
        queue = [PriorityQueue() for i in range(max_length)]
        for i in range(max_length):
            # max-(0:max-1)  -> max : 1
            len_label = max_length - i
            # print("starting {}--------".format(len_label))
            if i!=0:
                for j in range(queue[len_label].qsize()):
                    label = queue[len_label].get().bag
                    # label = queue[len_label].get()
                    char,queue,index_array = self.if_put_label(label,char, queue,len_label,index_array)

            if data_len[len_label] == []:
                continue
            for label in data_len[len_label]:
                char,queue,index_array= self.if_put_label(label, char, queue,len_label,index_array)
            if len(index_array) > taski_num:
                break
        # print("starting {}--------".format(0))
        print("task need {}, the lan need {}\n".format(taski_num,len(index_array)))
        for j in range(queue[0].qsize()):
            if len(index_array) > taski_num:
                break
            # label = queue[0].get()
            label = queue[0].get().bag
            char, queue,index_array= self.if_put_label(label, char, queue,0,index_array)
        print("samples get array {}--------".format(len(index_array)))
        self.memory_index.append(np.array(index_array[:taski_num]))


    def if_put_label(self, label,char,queue,len_value,index_array):
        label_v = 0
        string, = label
        index, = label.values()
        tmp_char = {}
        for s in string:
            if char.get(s, False) == False and tmp_char.get(s, False) == False:
                tmp_char[s] = True
                label_v += 1
        # choose this index & bag is True
        if label_v == len_value:
            index_array.append(index)
            # index_array.append({string: label_v})
            char.update(tmp_char)
        else:
            queue[label_v].put(bag_value(label))
            # queue[label_v].put(label)
        return char,queue,index_array

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


    def test(self, AlignCollate_valid,valid_datas,best_scores,ned_scores,taski):
        print("---Start evaluation on benchmark testset----")
        """ keep evaluation model and result logs """
        os.makedirs(f"./result/{self.opt.exp_name}", exist_ok=True)
        os.makedirs(f"./evaluation_log", exist_ok=True)
        if self.opt.ch_list != None:
            name = self.opt.ch_list[taski]
        else:
            name = self.opt.lan_list[taski]
        saved_best_model = f"./saved_models/{self.opt.exp_name}/{name}_{taski}_best_score.pth"
        # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
        self.model.load_state_dict(torch.load(f"{saved_best_model}"))

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
                ) = validation(self.model, self.criterion, valid_loader, self.converter, self.opt,val_choose="test")


            task_accs.append(round(current_score,2))
            ned_accs.append(round(ned_score,2))


        best_scores.append(round(sum(task_accs) / len(task_accs),2))
        ned_scores.append(round(sum(ned_accs) / len(ned_accs),2))

        acc_log= f'Task {taski} Test Average Incremental Accuracy: {best_scores[taski]} \n Task {taski} Incremental Accuracy: {task_accs}\n ned_acc: {ned_accs}\n'
        self.write_data_log(f'{taski} Avg Acc: {best_scores[taski]:0.2f} \n  acc: {task_accs}\n')

        print(acc_log)
        self.write_log(acc_log)
        return best_scores,ned_scores

    # def count_param(self):
    #     filtered_parameters = []
    #     params_num = []
    #     for p in filter(lambda p: p.requires_grad, self.model.parameters()):
    #         filtered_parameters.append(p)
    #         params_num.append(np.prod(p.size()))
    #     print("Trainable params num: {:.2f} M".format(sum(params_num) / 1000000))
    #     self.write_log("Trainable params num: {:.2f} M\n".format(sum(params_num) / 1000000))
    #     return filtered_parameters

    def count_param(self,model,trainable=True):
        filtered_parameters = []
        params_num = []
        if trainable == False:
            params = sum(p.numel() for p in model.parameters())
            print("All params num: {:.2f} M".format(params / 1000000))
            self.write_log("All params num: {:.2f} M\n".format(params/ 1000000))
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print("Trainable params num: {:.2f} M".format(sum(params_num) / 1000000))
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

    def write_data_log(self, line, name=None):
        with open(f"data_any.txt", "a+") as log:
            log.write(line)

