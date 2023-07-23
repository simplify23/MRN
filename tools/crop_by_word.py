# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import re

import mmcv
import numpy as np
from shapely.geometry import Polygon

def crop_img(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    """Crop text region with their bounding box.

    Args:
        src_img (np.array): The original image.
        box (list[float | int]): Points of quadrangle.
        long_edge_pad_ratio (float): Box pad ratio for long edge
            corresponding to font size.
        short_edge_pad_ratio (float): Box pad ratio for short edge
            corresponding to font size.
    """
    # assert utils.is_type_list(box, (float, int))
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0

    h, w = src_img.shape[:2]
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    font_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * font_size
        vertical_pad = short_edge_pad_ratio * font_size
    else:
        horizontal_pad = short_edge_pad_ratio * font_size
        vertical_pad = long_edge_pad_ratio * font_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    dst_img = src_img[top:bottom, left:right]

    return dst_img

def test_crop_img(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    pts1 = np.float32([[wordBB[0][0], wordBB[1][0]],
                       [wordBB[0][3], wordBB[1][3]],
                       [wordBB[0][1], wordBB[1][1]],
                       [wordBB[0][2], wordBB[1][2]]])
    height = math.sqrt((wordBB[0][0] - wordBB[0][3]) ** 2 + (wordBB[1][0] - wordBB[1][3]) ** 2)
    width = math.sqrt((wordBB[0][0] - wordBB[0][1]) ** 2 + (wordBB[1][0] - wordBB[1][1]) ** 2)

    # Coord validation check
    if (height * width) <= 0:
        err_log = 'empty file : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
        err_file.write(err_log)
        # print(err_log)
        continue
    elif (height * width) > (img_height * img_width):
        err_log = 'too big box : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
        err_file.write(err_log)
        # print(err_log)
        continue
    else:
        valid = True
        for i in range(2):
            for j in range(4):
                if wordBB[i][j] < 0 or wordBB[i][j] > img.shape[1 - i]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            err_log = 'invalid coord : {}\t{}\t{}\t{}\t{}\n'.format(
                image_name, txt[word_indx], wordBB, (width, height), (img_width, img_height))
            err_file.write(err_log)
            # print(err_log)
            continue

    pts2 = np.float32([[0, 0],
                       [0, height],
                       [width, 0],
                       [width, height]])

    x_min = np.int(round(min(wordBB[0][0], wordBB[0][1], wordBB[0][2], wordBB[0][3])))
    x_max = np.int(round(max(wordBB[0][0], wordBB[0][1], wordBB[0][2], wordBB[0][3])))
    y_min = np.int(round(min(wordBB[1][0], wordBB[1][1], wordBB[1][2], wordBB[1][3])))
    y_max = np.int(round(max(wordBB[1][0], wordBB[1][1], wordBB[1][2], wordBB[1][3])))
    # print(x_min, x_max, y_min, y_max)
    # print(img.shape)
    # assert 1<0
    if len(img.shape) == 3:
        img_cropped = img[y_min:y_max:1, x_min:x_max:1, :]
    else:
        img_cropped = img[y_min:y_max:1, x_min:x_max:1]

def list_to_file(filename, lines):
    """Write a list of strings to a text file.

    Args:
        filename (str): The output filename. It will be created/overwritten.
        lines (list(str)): Data to be written.
    """
    mmcv.mkdir_or_exist(os.path.dirname(filename))
    with open(filename, 'w', encoding='utf-8') as fw:
        for line in lines:
            fw.write(f'{line}\n')

def list_from_file(filename, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings. The
    trailing "\\r" and "\\n" of each line will be removed.

    Note:
        This will be replaced by mmcv's version after it supports encoding.

    Args:
        filename (str): Filename.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list


def load_img_info(file):
    """Load the information of one image.

    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)
        dataset(str): Dataset name, icdar2015 or icdar2017

    Returns:
        img_info(dict): The dict of the img and annotation information
    """
#     assert isinstance(files, tuple)
#     assert isinstance(dataset, str)
#     assert dataset

#     img_file, gt_file = files
    # read imgs with ignoring orientations
    # img = mmcv.imread(img_file, 'unchanged')
    gt_file = file[1]
    img_file = file[0]
    img = mmcv.imread(img_file, 'unchanged')

    split_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=img_file,
        height=img.shape[0],
        width=img.shape[1],)
    # img_file
    # print("gt_file{}".format(gt_file))
    gt_list = list_from_file(gt_file)

    anno_info = []
    # img_info = {}
    for line in gt_list:
        # each line has one ploygen (4 vetices), and others.
        # e.g., 695,885,866,888,867,1146,696,1143,Latin,9
        line = line.strip()
        strs = line.split(',')
        category_id = 1
        xy = [float(x) for x in strs[0:8]]
        coordinates = np.array(xy).reshape(-1, 2)
        polygon = Polygon(coordinates)

        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
#         bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        bbox = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        # bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        anno = dict(word=strs[9], bbox=bbox)
        anno_info.append(anno)
#         print(anno)
    img_info.update(anno_info=anno_info)
    # print(img_info)
    return img_info


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory
        split(str): The split of dataset. Namely: training or test
    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    # suffixes = ['.png']

    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, '*' + suffix)))

    imgs_list = sorted(imgs_list)
    ann_list = sorted(
        [osp.join(gt_dir, gt_file) for gt_file in os.listdir(gt_dir)])

    files = [(img_file, gt_file)
             for (img_file, gt_file) in zip(imgs_list, ann_list)]
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        nproc(int): The number of process to collect annotations
    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def generate_ann(root_path, image_infos, out_dir):
    """Generate cropped annotations and label txt file.

    Args:
        root_path(str): The relative path of the totaltext file
        split(str): The split of dataset. Namely: training or test
        image_infos(list[dict]): A list of dicts of the img and
        annotation information
    """

    dst_image_root = osp.join(out_dir, 'imgs')
    dst_label_file = osp.join(out_dir, 'label.txt')
    os.makedirs(dst_image_root, exist_ok=True)

    lines = []
    for image_info in image_infos:
        index = 1
        src_img_path = image_info['file_name']
        image = mmcv.imread(src_img_path)
        # src_img_root = osp.splitext(image_info['file_name'])[0].split('/')[1]
        src_img_root = image_info['file_name'].split('/')[-1].split(".")[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            dst_img = crop_img(image, anno['bbox'])

            # Skip invalid annotations
            if min(dst_img.shape) == 0:
                continue

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)
            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{word}')
        # print(lines)
        # print("\n")
    list_to_file(dst_label_file, lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SynthMLT annotations to COCO format')
    parser.add_argument('--root_path', help='SynthMLT root path')
    parser.add_argument('--lan', default="Hindi", help='languang for data')
    parser.add_argument('--out_dir', default="test",help='output path')
    # parser.add_argument(
    #     '--split-list',
    #     nargs='+',
    #     help='a list of splits. e.g., "--split_list training test"')

    parser.add_argument(
        '--nproc', default=10, type=int, help='number of process')
    args = parser.parse_args()
    return args

def unzip(root_path, lan):
    # root_path = "../dataset/SynthMLT/"
    img_path = "{}{}".format(root_path, lan)
    gt_path = "{}{}_gt".format(root_path, lan)

    if not os.path.exists(img_path):
        # os.system(f"rm -r {img_path}")
        cmd = "unzip -d {} {}.zip".format(img_path, img_path)
        os.system(cmd)


    if not os.path.exists(gt_path):
        # os.system(f"rm -r {gt_path}")
        cmd = "unzip -d {} {}.zip".format(gt_path, gt_path)
        os.system(cmd)

def main():
    args = parse_args()
    unzip(args.root_path, args.lan)
    # root_path = args.root_path + args.lan
    # out_dir = args.root_path + args.out_dir if args.out_dir else args.root_path
    out_dir = args.out_dir
    root_path = args.root_path + args.lan
    mmcv.mkdir_or_exist(out_dir)
    out_dir = osp.join(out_dir, args.lan)
    print("save to {}\n".format(out_dir))



    # root_path = "../dataset/SynthMLT/"
    img_dir = "{}/{}/".format(root_path, args.lan)
    gt_dir = "{}_gt/{}/".format(root_path, args.lan)

    print(f'Converting  SynthMLT to TXT\n')
    # print("img dir is {}\n".format(img_dir))
    # print("gt dir is {}\n".format(gt_dir))
    with mmcv.Timer( print_tmpl='It takes {}s to convert txt annotation'):
        files = collect_files(img_dir, gt_dir)
        # print("--------------------start------------\n{}".format(files))
        image_infos = collect_annotations(files, nproc=args.nproc)
        generate_ann(root_path, image_infos,out_dir)
    print(out_dir)


if __name__ == '__main__':
    main()
