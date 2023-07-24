import os
import sys
import six
import random

from natsort import natsorted
import PIL
import lmdb
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as transforms

from data.transform import CVGeometry, CVDeterioration, CVColorJitter

def hierarchical_dataset(root, opt, select_data="/", data_type="label", mode="train"):
    """select_data='/' contains all sub-directory of root directory"""
    dataset_list = []
    dataset_log = f"dataset_root:  {root}\t dataset: {select_data}"
    print(dataset_log)
    dataset_log += "\n"
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                # if data_type == "label":
                dataset = LmdbDataset(dirpath, opt, mode=mode)
                # else:
                #     dataset = LmdbDataset_unlabel(dirpath, opt)
                sub_dataset_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
                print(sub_dataset_log)
                dataset_log += f"{sub_dataset_log}\n"
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    def __init__(self, root, opt, mode="train"):

        self.root = root
        skip = 0
        self.opt = opt
        self.mode = mode
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            print(self.nSamples)
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                # print(label_key)
                if txn.get(label_key)==None:
                    skip+=1
                    print("skip --- {}\n".format(skip))
                    continue
                label = txn.get(label_key).decode("utf-8")
                # print(label)

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key).decode("utf-8")
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGBA")

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGBA", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"

        return (img, label)


class RawDataset(Dataset):
    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            img = PIL.Image.open(self.image_path_list[index]).convert("RGBA")

        except IOError:
            print(f"Corrupted image for {index}")
            # make dummy image and dummy label for corrupted image.
            img = PIL.Image.new("RGBA", (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])

class AlignCollate2(object):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.mode = mode

        if opt.Aug == "None" or mode != "train":
            self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        elif opt.Aug == "ABINet" and mode == "train":
            self.transform = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
                transforms.Resize(
                        (self.opt.imgH, self.opt.imgW), interpolation=PIL.Image.BICUBIC
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = Text_augment(opt)

    def __call__(self, batch):
        b_info, index = zip(*batch)
        images, labels = zip(*b_info)
        image_tensors = [self.transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels , index

class AlignCollate(object):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.mode = mode

        if opt.Aug == "None" or mode != "train":
            self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        elif opt.Aug == "ABINet" and mode == "train":
            self.transform = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
                transforms.Resize(
                        (self.opt.imgH, self.opt.imgW), interpolation=PIL.Image.BICUBIC
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = Text_augment(opt)

    def __call__(self, batch):
        images, labels = zip(*batch)
        image_tensors = [self.transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = image.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return image


class RandomCrop(object):
    """RandomCrop,
    RandomResizedCrop of PyTorch 1.6 and torchvision 0.7.0 work weird with scale 0.90-1.0.
    i.e. you can not always make 90%~100% cropped image scale 0.90-1.0, you will get central cropped image instead.
    so we made RandomCrop (keeping aspect ratio version) then use Resize.
    """

    def __init__(self, scale=[1, 1]):
        self.scale = scale

    def __call__(self, image):
        width, height = image.size
        crop_ratio = random.uniform(self.scale[0], self.scale[1])
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        x_start = random.randint(0, width - crop_width)
        y_start = random.randint(0, height - crop_height)
        image_crop = image.crop(
            (x_start, y_start, x_start + crop_width, y_start + crop_height)
        )
        return image_crop


class ResizeNormalize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        # CAUTION: it should be (width, height). different from size of transforms.Resize (height, width)
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image):
        image = image.resize(self.size, self.interpolation)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)
        return image


class Text_augment(object):
    """Augmentation for Text recognition"""

    def __init__(self, opt):
        self.opt = opt
        augmentation = []
        aug_list = self.opt.Aug.split("-")
        for aug in aug_list:
            if aug.startswith("Blur"):
                maximum = float(aug.strip("Blur"))
                augmentation.append(
                    transforms.RandomApply([GaussianBlur([0.1, maximum])], p=0.5)
                )

            if aug.startswith("Crop"):
                crop_scale = float(aug.strip("Crop")) / 100
                augmentation.append(RandomCrop(scale=(crop_scale, 1.0)))

            if aug.startswith("Rot"):
                degree = int(aug.strip("Rot"))
                augmentation.append(
                    transforms.RandomRotation(
                        degree, resample=PIL.Image.BICUBIC, expand=True
                    )
                )

        augmentation.append(
            transforms.Resize(
                (self.opt.imgH, self.opt.imgW), interpolation=PIL.Image.BICUBIC
            )
        )
        augmentation.append(transforms.ToTensor())
        self.Augment = transforms.Compose(augmentation)
        print("Use Text_augment", augmentation)

    def __call__(self, image):
        image = self.Augment(image)
        image.sub_(0.5).div_(0.5)

        return image


class MoCo_augment(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, opt):
        self.opt = opt

        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(
                (opt.imgH, opt.imgW), scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

        self.Augment = transforms.Compose(augmentation)
        print("Use MoCo_augment", augmentation)

    def __call__(self, x):
        q = self.Augment(x)
        k = self.Augment(x)
        q.sub_(0.5).div_(0.5)
        k.sub_(0.5).div_(0.5)

        return [q, k]
