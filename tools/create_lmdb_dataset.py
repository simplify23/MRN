import os
import random

import fire
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
lan_list = ["Chinese", "Arabic", "Japanese", "Korean", "Bangla","Hindi","Latin","Symbols"]
lan_list = ["Korean","Bangla","Hindi","Latin","Symbols"]
chi_list = ["ArT","RCTW","ReCTS","LSVT"]

def is_test(cnt,rad_list):
    return rad_list[cnt%10]==1

def from_gt_file(gt_path,img_path):
    lines = open(gt_path, 'r').readlines()
    image_list = []
    label_list = []
    for line in lines:
        line = line.strip()
        # print(line)
        str = line.split(" ",1)
        if len(str)==1:
            continue
        else:
            image, label=str[0],str[1]
        image_list.append(img_path+image)
        label_list.append(label)

    return image_list, label_list

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def write_txt(lexicon, name):
    # -*-coding:utf-8-*-
    file = name+".txt"
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(file):
        os.remove(file)

    # 以写的方式打开文件，如果文件不存在，就会自动创建
    file_write_obj = open(file, 'w')
    for key in lexicon:
#         print(key)
        file_write_obj.writelines(key)
        file_write_obj.write('\n')
    file_write_obj.close()

def create_train_test_Dataset(inputPath, gtFile, outputPath,outputPath2, checkValid=True,lan_lmdb=None):
    """a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    # CAUTION: if outputPath (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(outputPath):
        os.system(f"rm -r {outputPath}")

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=80 * 2 ** 30)
    cache = {}
    cnt = 1
    lexicon=set()

    if os.path.exists(outputPath2):
        os.system(f"rm -r {outputPath2}")

    os.makedirs(outputPath2, exist_ok=True)
    env2 = lmdb.open(outputPath2, map_size=80 * 2 ** 30)
    cache2 = {}
    rad_num = [i for i in range(10)]
    random.shuffle(rad_num)
    cnt_test=1
    cnt_train=1


    with open(gtFile, "r", encoding="utf-8-sig") as data:
        datalist = data.readlines()
    nSamples = len(datalist)

    for i in tqdm(range(nSamples), total=nSamples, position=0, leave=True):
        # imagePath, label = datalist[i].strip("\n").split("\t")
        imagePath, lan, label = datalist[i].strip("\n").split(",", 2)
        if lan_lmdb != None:
            if lan !=lan_lmdb:
                continue
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue


        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt

        if is_test(cnt,rad_num):
            t_image = "image-%09d".encode() % (cnt_test)
            t_label = "label-%09d".encode() % (cnt_test)
            cache2[t_image] = imageBin
            cache2[t_label] = label.encode()
            # cache2[imagepathKey] = imagePath.encode()
            cnt_test+=1
        else:
            t_image = "image-%09d".encode() % (cnt_train)
            t_label = "label-%09d".encode() % (cnt_train)
            cache[t_image] = imageBin
            cache[t_label] = label.encode()
            # cache[imagepathKey] = imagePath.encode()
            cnt_train+=1

        if cnt % 10 == 0:
            random.shuffle(rad_num)

        # print(label)
        # label = label.decode('utf-8')
        for char in label:
            lexicon.add(char)

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            writeCache(env2, cache2)
            cache2 = {}
            # print("test sample {}".format(cnt_test))
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cnt_test =cnt_test - 1
    cnt_train = cnt_train - 1
    cache["num-samples".encode()] = str(cnt_train).encode()
    writeCache(env, cache)
    cache2["num-samples".encode()] = str(cnt_test).encode()
    writeCache(env2, cache2)
    write_txt(lexicon,outputPath+"/dict")
    print(lexicon)
    print("Created dataset with %d train samples" % (cnt_train))
    print("Created dataset with %d test samples" % (cnt_test))

def create_from_lmdb_train_test_Dataset(inputPath, gtFile, outputPath,outputPath2, checkValid=True,lan_lmdb=None):
    """a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    # CAUTION: if outputPath (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    lexicon = set()

    if os.path.exists(outputPath):
        os.system(f"rm -r {outputPath}")

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=80 * 2 ** 30)
    cache = {}
    cnt = 1


    if os.path.exists(outputPath2):
        os.system(f"rm -r {outputPath2}")

    os.makedirs(outputPath2, exist_ok=True)
    env2 = lmdb.open(outputPath2, map_size=80 * 2 ** 30)
    cache2 = {}
    rad_num = [i for i in range(10)]
    random.shuffle(rad_num)
    cnt_test=1
    cnt_train=1

    env_in = lmdb.open(inputPath, readonly=True) # 打开文件
    txn_in = env_in.begin() # 生成处理句柄
    # cur_in = txn_in.cursor() # 生成迭代器指针
    nSamples = int(txn_in.get('num-samples'.encode()))
    print("total sampler:{}".format(nSamples))


    for i in tqdm(range(nSamples), total=nSamples, position=0, leave=True):
        # imagePath, label = datalist[i].strip("\n").split("\t")
        #label_key = 'label-%09d' % (index + 1)
        # text = cur.get(label_key.encode())
        #label = str(txn.get(label_key.encode()).decode('utf-8'))

        imageKey = "image-%09d".encode() % (i+1)
        # imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % (i+1)

        label = txn_in.get(labelKey)
        image = txn_in.get(imageKey)

        if is_test(cnt,rad_num):
            t_image = "image-%09d".encode() % (cnt_test)
            t_label = "label-%09d".encode() % (cnt_test)
            # print(label.decode("utf-8"))
            # print(t_label.decode("utf-8"))
            cache2[t_image] = image
            cache2[t_label] = label
            # cache2[imagepathKey] = imagePath.encode()
            cnt_test+=1
        else:
            t_image = "image-%09d".encode() % (cnt_train)
            t_label = "label-%09d".encode() % (cnt_train)
            cache[t_image] = image
            cache[t_label] = label
            # cache[imagepathKey] = imagePath.encode()
            cnt_train+=1

        if cnt % 10 == 0:
            random.shuffle(rad_num)

        # print(label)
        # label = label.decode('utf-8')
        for char in label.decode('utf-8'):
            lexicon.add(char)

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            writeCache(env2, cache2)
            cache2 = {}
            # print("test sample {}".format(cnt_test))
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    cnt_test = cnt_test - 1
    cnt_train = cnt_train - 1
    cache["num-samples".encode()] = str(cnt_train).encode()
    writeCache(env, cache)
    cache2["num-samples".encode()] = str(cnt_test).encode()
    writeCache(env2, cache2)
    print("Created dataset with %d train samples" % (cnt_train))
    print("Created dataset with %d test samples" % (cnt_test))

    print(lexicon)
    write_txt(lexicon,outputPath+"/dict")
    # print(lexicon)

def createDataset(inputPath, gtFile, outputPath, checkValid=True,lan_lmdb=None):
    """a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    # CAUTION: if outputPath (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(outputPath):
        os.system(f"rm -r {outputPath}")

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=80 * 2 ** 30)
    cache = {}
    cnt = 1
    lexicon=set()

    with open(gtFile, "r", encoding="utf-8-sig") as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in tqdm(range(nSamples), total=nSamples, position=0, leave=True):
        # imagePath, label = datalist[i].strip("\n").split("\t")
        imagePath, lan, label = datalist[i].strip("\n").split(",", 2)
        if lan_lmdb != None:
            if lan !=lan_lmdb:
                continue
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[imagepathKey] = imagePath.encode()

        # print(label)
        # label = label.decode('utf-8')
        for char in label:
            lexicon.add(char)

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    write_txt(lexicon,outputPath+"/dict")
    print(lexicon)
    print("Created dataset with %d samples" % nSamples)

def createSynthMLTDataset(inputPath, gtFile, outputPath, checkValid=True):
    """a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    # CAUTION: if outputPath (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(outputPath):
        os.system(f"rm -r {outputPath}")

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=80 * 2 ** 30)
    cache = {}
    cnt = 1
    lexicon=set()

    gt_list = gtFile

    nSamples = len(gt_list)
    for i in tqdm(range(nSamples), total=nSamples, position=0, leave=True):
        # imagePath, label = datalist[i].strip("\n").split("\t")
        # imagePath, label = datalist[i].strip("\n").split(" ", 1)
        label = gt_list[i]
        imagePath = inputPath[i]

        # imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[imagepathKey] = imagePath.encode()

        # print(label)
        # label = label.decode('utf-8')
        for char in label:
            lexicon.add(char)

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    write_txt(lexicon,outputPath+"/dict")
    print(lexicon)
    print("Created dataset with %d samples" % nSamples)


def createDataset_with_ValidTestset(
    inputPath,
    gtFile,
    outputPath,
    dataset_name,
    validset_percent=10,
    testset_percent=0,
    random_seed=1111,
    checkValid=True,
):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    train_path = os.path.join(outputPath, "training", dataset_name)
    valid_path = os.path.join(outputPath, "validation", dataset_name)

    # CAUTION: if train_path (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(train_path):
        os.system(f"rm -r {train_path}")

    if os.path.exists(valid_path):
        os.system(f"rm -r {valid_path}")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    gt_train_path = gtFile.replace(".txt", "_train.txt")
    gt_valid_path = gtFile.replace(".txt", "_valid.txt")
    data_log = open(gt_train_path, "w", encoding="utf-8")

    if testset_percent != 0:
        test_path = os.path.join(outputPath, "evaluation", dataset_name)
        if os.path.exists(test_path):
            os.system(f"rm -r {test_path}")
        os.makedirs(test_path, exist_ok=True)
        gt_test_path = gtFile.replace(".txt", "_test.txt")

    env = lmdb.open(train_path, map_size=30 * 2 ** 30)
    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8-sig") as data:
        datalist = data.readlines()

    random.seed(random_seed)
    random.shuffle(datalist)

    nSamples = len(datalist)
    num_valid_dataset = int(nSamples * validset_percent / 100.0)
    num_test_dataset = int(nSamples * testset_percent / 100.0)
    num_train_dataset = nSamples - num_valid_dataset - num_test_dataset
    print(
        f"# Train dataset: {num_train_dataset}, # valid datast: {num_valid_dataset}, and # test datast: {num_test_dataset}"
    )

    for i in tqdm(range(nSamples), total=nSamples, position=0, leave=True):
        data_log.write(datalist[i])
        imagePath, label = datalist[i].strip("\n").split("\t")
        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[imagepathKey] = imagePath.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))

        # Finish train dataset and Start validation dataset
        if i + 1 == num_train_dataset:
            print(f"# Train dataset: {num_train_dataset} is finished")
            cache["num-samples".encode()] = str(num_train_dataset).encode()
            writeCache(env, cache)
            data_log.close()

            # start validation set
            env = lmdb.open(valid_path, map_size=30 * 2 ** 30)
            cache = {}
            cnt = 0  # not 1 at this time
            data_log = open(gt_valid_path, "w", encoding="utf-8")

        # Finish train/valid dataset and Start test dataset
        if (i + 1 == num_train_dataset + num_valid_dataset) and num_test_dataset != 0:
            print(f"# Valid dataset: {num_valid_dataset} is finished")
            cache["num-samples".encode()] = str(num_valid_dataset).encode()
            writeCache(env, cache)
            data_log.close()

            # start test set
            env = lmdb.open(test_path, map_size=30 * 2 ** 30)
            cache = {}
            cnt = 0  # not 1 at this time
            data_log = open(gt_test_path, "w", encoding="utf-8")

        cnt += 1

    if testset_percent == 0:
        cache["num-samples".encode()] = str(num_valid_dataset).encode()
        writeCache(env, cache)
        print(f"# Valid datast: {num_valid_dataset} is finished")
    else:
        cache["num-samples".encode()] = str(num_test_dataset).encode()
        writeCache(env, cache)
        print(f"# Test datast: {num_test_dataset} is finished")


if __name__ == "__main__":

    # root_path = "../dataset/MLT2019/"
    # root_path = "../dataset/chinese/"
    # # print(root_path)
    # # for lan in lan_list:
    # #     create_train_test_Dataset(inputPath=root_path+"train", gtFile=root_path+"train/gt.txt",
    # #                               outputPath=root_path+"train_2019/mlt_2019_train_{}".format(lan),
    # #                               outputPath2=root_path + "test_2019/mlt_2019_test_{}".format(lan),
    # #                               checkValid=True,lan_lmdb=lan)
    #     # createDataset(inputPath=root_path+"train", gtFile=root_path+"train/gt.txt", outputPath=root_path+"mlt_2019_train_{}".format(lan), checkValid=True,lan_lmdb=lan)
    # # fire.Fire(createDataset)
    # # chi_list=["CTW"]
    # for path in chi_list:
    #     total_path = root_path + path
    #     create_from_lmdb_train_test_Dataset(inputPath=total_path+"_train",gtFile=None,outputPath=total_path+"/train",outputPath2=total_path+"/test", checkValid=True,lan_lmdb=None)

    # SynthMLT
    for lan in lan_list:
        root_path = "/home/ztl/dataset/SynthMLT/"
        gt_path= "{}txt/{}/label.txt".format(root_path, lan)
        img_path="{}txt/{}/".format(root_path, lan)
        imgList, labelList = from_gt_file(gt_path,img_path)
        print("The length of the list is ", len(imgList))

        '''Input the address you want to generate the lmdb file.'''
        createSynthMLTDataset(imgList, labelList,root_path+"lmdb/"+lan)
    # inputPath, gtFile, outputPath,