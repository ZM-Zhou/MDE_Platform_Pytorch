import os
import json
import cv2 as cv
import numpy as np

log_path = "/data/Train_Log/"
compare = []
compare.append("2020-05-20_02h44m02s")
compare.append("2020-05-20_04h22m02s")
compare.append("2020-05-19_04h59m35s")
compare.append("2020-05-06_23h47m23s")

compare_data = "eval/abs_rel"

def condition(data1, data2, data3, data4):
    if data1 <= data2 and data2 <= data3 and data3 <= data4:
        return True
    else:
        return False


for i in range(len(compare)):
    dir_path = os.path.join(log_path, compare[i])
    for r, ds ,fs in os.walk(dir_path):
        for d in ds:
            if "KITTI" in d: #or "CS" in d:
                print("->find metric")
                compare[i] = os.path.join(dir_path, d, "metric_list.json")

count = []
num = 0
with open(compare[0], "r") as f1:
    json_data1 = json.load(f1)
    with open(compare[1], "r") as f2:
        json_data2 = json.load(f2)
        with open(compare[2], "r") as f3:
            json_data3 = json.load(f3)
            with open(compare[3], "r") as f4:
                json_data4 = json.load(f4)
                for k, _ in json_data1.items():
                    m1 = json_data1[k]["metric"][compare_data]
                    m2 = json_data2[k]["metric"][compare_data]
                    m3 = json_data3[k]["metric"][compare_data]
                    m4 = json_data4[k]["metric"][compare_data]
                    if condition(m1 ,m2, m3, m4):
                        count.append([m4 - m1, json_data1[k]["path"] + "\n"])
                    num += 1
                count.sort(key=lambda x:x[0])
                count = count[-60:]
                for idx, c in enumerate(count):
                    count[idx] = c[1]

with open(os.path.join("./split_list", "compare_files.txt"), "w") as f:
    f.writelines(count)