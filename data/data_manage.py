import bisect
import os

import numpy as np
import numpy.random
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

from data.dataset import concat_dataset, AlignCollate, LmdbDataset, AlignCollate2, hierarchical_dataset


class Dataset_Manager(object):
    def __init__(self,opt):
        self.data_list = []
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.select_data = None
        self.opt = opt


    def get_dataset(self, taski, memory="random_memory",index_list=None):
        self.data_loader_list = []
        self.dataloader_iter_list = []
        memory_num = self.opt.memory_num

        dataset = self.create_dataset(data_list=self.select_data,taski=taski)

        if memory != None and self.opt.il=="ems":
            # curr: num/(taski-1) mem: num/(taski-1)
            index_current = numpy.random.choice(range(len(dataset)),int(self.opt.memory_num/(taski)),replace=False)
            split_dataset = Subset(dataset,index_current.tolist())
            memory_data,index_list = self.rehearsal_memory(taski, random=False,total_num=self.opt.memory_num,index_array=index_list)
            self.create_dataloader_mix(IndexConcatDataset([memory_data,split_dataset]),self.opt.batch_size)
            print("taski is {} current dataset chose {}\n now dataset chose {}".format(taski,int(self.opt.memory_num/taski),len(memory_data)))
        elif memory == "test_ch":
            # curr: total  mem: num/(taski-1) (repeat)
            # index_current = numpy.random.choice(range(len(dataset)),int(self.opt.memory_num/taski),replace=False)
            # split_dataset = Subset(dataset,index_current.tolist())
            memory_data,index_list = self.rehearsal_memory(taski, random=False,total_num=self.opt.memory_num,index_array=index_list,repeat=True)
            self.create_dataloader_mix(IndexConcatDataset([memory_data,dataset]),self.opt.batch_size)
            print("taski is {} current dataset chose {}\n now dataset chose {}".format(taski,int(self.opt.memory_num/taski),len(memory_data)))
        elif memory == "large":
            # curr: num  mem: num
            index_current = numpy.random.choice(range(len(dataset)), memory_num, replace=False)
            split_dataset = Subset(dataset, index_current.tolist())
            memory_data, index_list = self.rehearsal_memory(taski, random=False, total_num=memory_num*taski, index_array=index_list)
            self.create_dataloader_mix(IndexConcatDataset([memory_data, split_dataset]), self.opt.batch_size)
            print("taski is {} current dataset chose {}\n now dataset chose {}".format(taski, int(memory_num),
                                                                                       len(memory_data)))
        elif memory == "total":
            # curr : total  mem : total(repeat)
            total_data_list = []
            total_data_list.append(dataset)
            for i in range(taski):
                dataset = self.create_dataset(data_list=self.select_data, taski=i)
                total_data_list.append(dataset)
            self.create_dataloader_mix(IndexConcatDataset(total_data_list), self.opt.batch_size)
            print("taski is {} current dataset chose {} lenth dataset\n now dataset chose {}".format(taski, len(total_data_list),
                                                                                       len(dataset)))
        elif memory != None:
            memory_data,index_list = self.rehearsal_memory(taski, random=False,total_num=memory_num,index_array=index_list)
            self.create_dataloader(memory_data,(self.opt.batch_size)//2)
            self.create_dataloader(dataset,(self.opt.batch_size)//2)
        # elif memory == "rehearsal":
        #     memory_data, index_list = self.rehearsal_memory(taski, random=False,total_num=2000,index_array=index_list)
        #     self.create_dataloader(memory_data,(self.opt.batch_size)//2)
        #     self.create_dataloader(dataset,(self.opt.batch_size)//2)
        else:
            self.create_dataloader(dataset)
        return index_list

    def joint_start(
        self, opt, dataset_root, select_data, log, taski,total_task):
        self.opt = opt
        self.select_data = select_data
        dashed_line = "-" * 80
        print(dashed_line)
        log.write(dashed_line + "\n")
        # print(
        #     f"dataset_root: {dataset_root}\n select_data: {select_data}\n"
        # )
        # log.write(
        #     f"dataset_root: {dataset_root}\n select_data: {select_data}\n"
        # )


        dataset = self.create_dataset(data_list=self.select_data, taski=taski)
        if opt.il == "joint_mix":
            self.data_list.append(dataset)
            if taski == total_task-1:
                self.create_dataloader(ConcatDataset(self.data_list), int(self.opt.batch_size))
        elif opt.il == "joint_loader":
            self.create_dataloader(dataset, int(self.opt.batch_size // total_task))


    def init_start(
        self, opt, dataset_root, select_data, log, taski,memory="random"):
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
        self.get_dataset(taski, memory=None)
        # dataset = self.create_dataset(data_list=select_data,taski=taski)
        # self.create_dataloader(dataset)
        # if memory == "random_memory":
        #     memory_data = self.memory_dataset(select_data, taski, random=True,total_num=2000)
        #     self.create_dataloader(memory_data)

        # self.data_list.append(dataset)
        # self.create_data_loader(dataset)

    def memory_dataset(self,select_data, taski, random=True,total_num=2000,index_list=None):
        data_list = []
        num_i = int(total_num/taski)
        for i in range(taski):
            dataset = self.create_dataset(data_list=select_data,taski=i,repeat=False)
            if random:
                index_list = numpy.random.choice(range(len(dataset)),num_i,replace=False)
            # print(random)
            split_dataset = Subset(dataset,list(index_list))
            data_list.append(split_dataset)
        return ConcatDataset(data_list)

    def rehearsal_memory(self,taski, random=False,total_num=2000,index_array=None,repeat=False):
        data_list = []
        select_data = self.select_data
        num_i = int(total_num/(taski))
        print("memory size is {}\n".format(num_i))
        for i in range(taski):
            dataset = self.create_dataset(data_list=select_data,taski=i,repeat=repeat)
            if random:
                index_list = numpy.random.choice(range(len(dataset)),num_i,replace=repeat)
            # print(random)
            else:
                index_list = index_array[i]
            split_dataset = Subset(dataset,index_list.tolist())
            data_list.append(split_dataset)
        return ConcatDataset(data_list), index_array

    def rehearsal_prev_model(self,taski,):
        select_data = self.select_data
        dataset = self.create_dataset(data_list=select_data,taski=taski-1,repeat=False)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate(self.opt),
            pin_memory=False,
            drop_last=False,
        )
        return data_loader,len(dataset)

    def rehearsal_prev_dataset(self,taski,):
        select_data = self.select_data
        dataset = self.create_dataset(data_list=select_data,taski=taski-1,repeat=False)
        return dataset,len(dataset)

    def create_dataset(self, data_list="/", taski=0, mode="train", repeat=True):
        """select_data is list for all dataset"""
        dataset_list = []
        for data_root in data_list:
            # dataset_log = f"dataset_root: {data_root}"
            # print(dataset_log)
            # dataset_log += "\n"
            dataset = LmdbDataset(data_root + "/" + self.opt.lan_list[taski], self.opt, mode=mode)
            dataset_log = f"num samples: {len(dataset)}"
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

    def create_dataloader(self,dataset,batch_size=None):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size if batch_size==None else batch_size,
            shuffle=True,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate(self.opt),
            pin_memory=False,
            drop_last=False,
        )
        self.data_loader_list.append(data_loader)
        self.dataloader_iter_list.append(iter(data_loader))

    def create_dataloader_mix(self,dataset,batch_size=None):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size if batch_size==None else batch_size,
            shuffle=True,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate2(self.opt),
            pin_memory=False,
            drop_last=False,
        )
        self.data_loader_list.append(data_loader)
        self.dataloader_iter_list.append(iter(data_loader))

    def get_batch2(self):
        balanced_batch_images = []
        balanced_batch_labels = []
        balanced_batch_index = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, label,index = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
                balanced_batch_index.append(index)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, label, index = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
                balanced_batch_index.append(index)
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_labels, balanced_batch_index

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

class Val_Dataset(object):
    def __init__(self,val_datas,opt):
        self.data_loader_list = []
        self.dataset_list = []
        self.current_data = val_datas[-1]
        self.val_datas = val_datas
        self.opt = opt
        self.AlignCollate_valid = AlignCollate(self.opt, mode="test")


    def create_dataset(self,val_data=None):
        if val_data == None:
            val_data = self.current_data
        valid_dataset, valid_dataset_log = hierarchical_dataset(
            root=val_data, opt=self.opt, mode="test"
        )
        print(valid_dataset_log)
        print("-" * 80)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_valid,
            pin_memory=False,
        )
        return valid_loader

    def create_list_dataset(self,valid_datas=None):
        if valid_datas==None:
            valid_datas = self.val_datas
        concat_data = []
        for val_data in valid_datas:
            valid_dataset, valid_dataset_log = hierarchical_dataset(
                root=val_data, opt=self.opt, mode="test")
            if len(valid_dataset) > 700:
                index_current = numpy.random.choice(range(len(valid_dataset)),700,replace=False)
                valid_dataset = Subset(valid_dataset,index_current.tolist())
            concat_data.append(valid_dataset)
            print(valid_dataset_log)
            print("-" * 80)
        val_data = ConcatDataset(concat_data)
        valid_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_valid,
            pin_memory=False,
        )
        return valid_loader


class IndexConcatDataset(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx],dataset_idx

class DummyDataset(Dataset):
    def __init__(self, images, labels):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        # self.trsf = trsf
        # self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return (image, label)