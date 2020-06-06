import os
import sys
import random

sys.path.append(os.getcwd())

dataset_path = "../Datasets/CS"
ori_path = "split_list_full"
ori_split_name = "cityscapes_train_extra_list.txt"
split_name = "cityscapes_train_extra_list_shuf.txt"

split_keep = []
with open(os.path.join(dataset_path, ori_path, ori_split_name), "r") as f:
    split_list = f.readlines()
    list_len = len(split_list)
    split_all = [x for x in range(list_len)]
    random.shuffle(split_all)
    keep_idx = split_all

    for idx in keep_idx:
        split_keep.append(split_list[idx])

with open(os.path.join(dataset_path, "split_list", split_name), "w") as f:
    f.writelines(split_keep)

