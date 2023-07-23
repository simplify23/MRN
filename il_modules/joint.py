import time
import torch
import torch.utils.data
from tqdm import tqdm

from il_modules.base import BaseLearner
from tools.utils import Averager, adjust_learning_rate

class JointLearner(BaseLearner):

    def incremental_train(self,taski, character, train_loader, valid_loader,AlignCollate_valid,valid_datas):

        # pre task classes for know classes
        # self._known_classes = self._total_classes
        self.character = character
        self.converter = self.build_converter()
        valid_loader = valid_loader.create_list_dataset(valid_datas=valid_datas)

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

        """ start training """
        start_iter = 0
        if self.opt.saved_model != "":
            try:
                start_iter = int(self.saved_model.split("_")[-1].split(".")[0])
                print(f"continue to train, start_iter: {start_iter}")
            except:
                pass

        return self._init_train(start_iter,taski, train_loader, valid_loader,AlignCollate_valid,valid_datas)

    def _init_train(self,start_iter,taski, train_loader, valid_loader,AlignCollate_valid,valid_datas):
        # loss averager
        train_loss_avg = Averager()
        best_scores = []
        ned_scores = []


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
                preds = self.model(image, labels_index[:, :-1])["predict"]  # align with Attention.forward
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
                if iteration != 1:
                    best_scores,ned_scores = self.test(AlignCollate_valid,valid_datas,best_scores,ned_scores,taski)
                    self.model.train()
                train_loss_avg.reset()
        return best_scores,ned_scores