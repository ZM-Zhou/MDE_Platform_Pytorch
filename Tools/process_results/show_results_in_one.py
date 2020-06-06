import os
import cv2 as cv
import numpy as np

log_path = "/data/Train_Log"
# result_dict = {"monov2":"2020-05-06_23h47m23s",
#                "aspp": "2020-05-19_04h59m35s",
#                "fuse_loss": "2020-05-20_02h44m02s"}
#choice_img = ["1", "5", "7", "12", "21"]  # kitti
choice_img = [str(k) for k in range(60)]
#choice_img = ["1", "6", "12", "14", "16"]  # CS
output_name = "Q_15"

def find_img(dir_path):
    for r, ds ,fs in os.walk(dir_path):
        for d in ds:
            if "KITTI" in d : #or "CS" in d:
                print("->find metric")
                # return os.path.join(dir_path, d, "Choice")
                return os.path.join(dir_path, d, "Compare")

def get_img(img_dir, img):
    for name in choice_img:
        img_path = os.path.join(img_dir, name + ".png")
        metric_img = cv.imread(img_path)
        h, w, _ = metric_img.shape
        img_h = int(h / 4)
        img_w = int(w / 2)
        if not metric_img is None:
            print("-->find {}.png".format(name))
            if img[name] == []:
                ori_img = metric_img[: img_h, : img_w, :]
                board = metric_img[: img_h, img_w:, :]
                ori_img = np.vstack([board, ori_img, board])
                img[name].append(ori_img)
            valid_img = metric_img[img_h: 4*img_h, : img_w, :]
            img[name].append(valid_img)

        
            


output_path = os.path.join(log_path, output_name)
if not os.path.isdir(output_path):
            os.makedirs(output_path)
walk_path = os.path.join(log_path)
img = {}
for i in choice_img:
    img[i] = []

for k, v in result_dict.items():
    for r, ds, fs in os.walk(walk_path):
        for d in ds:
            if d == v:
                print("find {}".format(k))
                dir_path = os.path.join(r, d)
                img_dir = find_img(dir_path)
                get_img(img_dir, img)
    
for k, v in img.items():
    output = np.hstack(v)
    output_name = os.path.join(output_path, k + '.png')
    cv.imwrite(output_name, output)

