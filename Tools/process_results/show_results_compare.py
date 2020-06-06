import os
import cv2 as cv
import numpy as np

log_path = "/data/Train_Log/N1"
choice_img = ["5", "6", "7", "11", "15"]  # kitti
# choice_img = [str(k) for k in range(23)]
# choice_img = ["1", "6", "12", "14", "16"]  # CS
output_name = "N1_cp"

def get_img(img_dir, img):
    for name in choice_img:
        img_path = os.path.join(img_dir, name + ".png")
        metric_img = cv.imread(img_path)
        h, w, _ = metric_img.shape
        img_h = int(h / 3)
        if not metric_img is None:
            print("-->find {}.png".format(name))
            valid_img = metric_img[img_h: 2*img_h, ...]
            img.append(valid_img)

img =  []

get_img(log_path, img)
output = np.vstack(img)
output_name = os.path.join(log_path, output_name + '.png')
cv.imwrite(output_name, output)