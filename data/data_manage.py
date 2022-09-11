import os

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

from data.dataset import concat_dataset, AlignCollate, LmdbDataset


class Dataset_Manager(object):
    def __int__(self):
        self.data_list = []
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.select_data = None

    def init_start(
        self, opt, dataset_root, select_data, batch_ratio, log, taski):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        self.opt = opt
        self.select_data = select_data
        dashed_line = "-" * 80
        print(dashed_line)
        log.write(dashed_line + "\n")
        print(
            f"dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}"
        )
        log.write(
            f"dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}\n"
        )
        assert len(select_data) == len(batch_ratio)

        # _AlignCollate = AlignCollate(self.opt)
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(select_data, batch_ratio):
            _batch_size = max(round(self.opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + "\n")
            _dataset, _dataset_log = self.create_dataset(
                    data_root=selected_d,
                    taski=taski,
                )
            log.write(_dataset_log)

            selected_d_log = f"num samples of {selected_d} per batch: {self.opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}"
            print(selected_d_log)
            log.write(selected_d_log + "\n")
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            # for faster training, we multiply small datasets itself.
            if len(_dataset) < 50000:
                multiple_times = int(50000 / len(_dataset))
                dataset_self_multiple = [_dataset] * multiple_times
                _dataset = ConcatDataset(dataset_self_multiple)

            self.data_list.append(_dataset)
            self.create_data_loader(_dataset, _batch_size)

        Total_batch_size_log = f"{dashed_line}\n"
        batch_size_sum = "+".join(batch_size_list)
        Total_batch_size_log += (
            f"Total_batch_size: {batch_size_sum} = {Total_batch_size}\n"
        )
        Total_batch_size_log += f"{dashed_line}"
        self.opt.Total_batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + "\n")

    def create_dataset(self, data_root="/", taski=0, mode="train"):
        """select_data='/' contains all sub-directory of root directory"""
        dataset_log = f"dataset_root: {data_root}"
        # print(dataset_log)
        dataset_log += "\n"
        dataset = LmdbDataset(data_root + "/" + self.opt.lan_list[taski], self.opt, mode=mode)
        dataset_log += f"num samples: {len(dataset)}"
        print(dataset_log)
        return dataset, dataset_log

    def create_data_loader(self, _dataset, _batch_size, ):
        _data_loader = torch.utils.data.DataLoader(
            _dataset,
            batch_size=_batch_size,
            shuffle=True,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate(self.opt),
            pin_memory=False,
            drop_last=False,
        )
        self.data_loader_list.append(_data_loader)
        self.dataloader_iter_list.append(iter(_data_loader))

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_labels = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, label = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, label = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_labels
