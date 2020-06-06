import os
import json
import cv2 as cv
import numpy as np

log_path = "/data/Train_Log/"
compare = []
compare.append("2020-05-08_10h07m42s")
compare.append("2020-05-06_23h47m23s")
compare_data = "eval/abs_rel"


for i in range(2):
    dir_path = os.path.join(log_path, compare[i])
    for r, ds ,fs in os.walk(dir_path):
        for d in ds:
            if "KITTI" in d or "CS" in d:
                print("->find metric")
                compare[i] = os.path.join(dir_path, d, "metric_list.json")

count = []
num = 0
with open(compare[0], "r") as f1:
    json_data1 = json.load(f1)
    with open(compare[1], "r") as f2:
        json_data2 = json.load(f2)
        for k, _ in json_data1.items():
            m1 = json_data1[k]["metric"][compare_data]
            m2 = json_data2[k]["metric"][compare_data]
            if m1 < m2:
                count.append(str(num) + ": " + json_data1[k]["path"] + "\n")
            num += 1

with open(os.path.join(log_path, "N3-N0", "better.txt"), "w") as f:
    f.writelines(count)