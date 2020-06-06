import os
import sys
import random

sys.path.append(os.getcwd())

dataset_path = "kitti/kitti_full"
full_path = "split_list"
full_split_name = "train_files_all.txt"
split_name = "train_files.txt"
keep_num = 4000

split_keep = []
with open(os.path.join(dataset_path, full_path, full_split_name), "r") as f:
    split_list = f.readlines()
    list_len = len(split_list)
    split_all = [x for x in range(list_len)]
    random.shuffle(split_all)
    keep_idx = split_all[0:keep_num]

    for idx in keep_idx:
        split_keep.append(split_list[idx])

with open(os.path.join(dataset_path, "split_list", split_name), "w") as f:
    f.writelines(split_keep)

