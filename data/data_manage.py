import os

import numpy as np
import numpy.random
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

from data.dataset import concat_dataset, AlignCollate, LmdbDataset


class Dataset_Manager(object):
    def __int__(self):
        # self.data_list = []
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.select_data = None

    def init_start(
        self, opt, dataset_root, select_data, log, taski,memory=None):
        self.opt = opt
        self.select_data = select_data
        self.data_loader_list = []
        self.dataloader_iter_list = []
        dashed_line = "-" * 80
        print(dashed_line)
        log.write(dashed_line + "\n")
        print(
            f"dataset_root: {dataset_root}\n select_data: {select_data}\n"
        )
        log.write(
            f"dataset_root: {dataset_root}\n select_data: {select_data}\n"
        )

        dataset = self.create_dataset(data_list=select_data,taski=taski)
        if memory != None:
            pass
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate(self.opt),
            pin_memory=False,
            drop_last=False,
        )
        self.data_loader_list.append(data_loader)
        self.dataloader_iter_list.append(iter(data_loader))
        # self.data_list.append(dataset)
        # self.create_data_loader(dataset)

    def memory_dataset(self,select_data, taski, random=True,total_num=2000,index_list=None):
        data_list = []
        num_i = int(total_num/taski)
        for i in range(taski-1):
            dataset = self.create_dataset(data_list=select_data,taski=i,repeat=False)
            if random:
                index_list = numpy.random.choice(range(len(dataset)),num_i,replace=False)
            # print(random)
            split_dataset = Subset(dataset,list(index_list))
            data_list.append(split_dataset)
        return ConcatDataset(data_list)


    def create_dataset(self, data_list="/", taski=0, mode="train", repeat=True):
        """select_data is list for all dataset"""
        dataset_list = []
        for data_root in data_list:
            dataset_log = f"dataset_root: {data_root}"
            # print(dataset_log)
            dataset_log += "\n"
            dataset = LmdbDataset(data_root + "/" + self.opt.lan_list[taski], self.opt, mode=mode)
            dataset_log += f"num samples: {len(dataset)}"
            print(dataset_log)

            # for faster training, we multiply small datasets itself.
            if len(dataset) < 50000 and repeat:
                multiple_times = int(50000 / len(dataset))
                dataset_self_multiple = [dataset] * multiple_times
                dataset = ConcatDataset(dataset_self_multiple)
            dataset_list.append(dataset)
        # if memory !=None:
        #     dataset_list.append(memory_dataset)

        return ConcatDataset(dataset_list)

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
