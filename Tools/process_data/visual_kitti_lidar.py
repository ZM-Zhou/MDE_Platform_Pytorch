import os
import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.cm as cm
import copy
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())
from _Dataset.kitti import KittiColorDataset, Dataset_Options
from _Network.base_module import get_depth_sid

data_dir = "../Datasets/KITTI"
gt_path = os.path.join(data_dir, "gt_depths.npz")
gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

opts = Dataset_Options()
opts.data_dir = data_dir
opts.height = 375
opts.width = 1242
dataset = KittiColorDataset(opts, mode="test")

with open(os.path.join(data_dir, "split_list", "test_files.txt"), "r") as f:
    test_imgs = f.readlines()
side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

u_depth_range = [0 for i in range(100)]
cha_num = 64

ch_m = 10
ch_c = 24
di = 8
max_depth = 100
ranges = torch.arange(0, cha_num, 1).to(torch.float)

a = np.log(10)
b = torch.log(torch.tensor(10.0))

sid_depth_guide = [get_depth_sid(i, min_depth, max_depth, cha_num - 1).item() for i in ranges]
sid_depth_range = [0 for i in range(cha_num)]
all_valid = 0
all_pix = 0
all_depth = 0

def insert_range(d):
    for idx, guide in enumerate(sid_depth_guide):
        if d < guide:
            sid_depth_range[idx] += 1
            return


for idx, depth in enumerate(gt_depths):
    inputs = dataset.__getitem__(idx)
    raw_depth = inputs["depth_gt"].unsqueeze(0)
    # img = inputs[("color", "l")]
    # line = test_imgs[idx].replace("\n", "")
    # line = line.split()
    # folder = line[0]
    # frame_index = int(line[1])
    # side = line[2]
    # f_str = "{:010d}.{}".format(frame_index, "png")
    # image_path = os.path.join(data_dir, folder,
    #                           "image_0{}/data".format(side_map[side]),
    #                           f_str)
    # img = cv.imread(image_path)
    depth_shape = depth.shape
    raw_depth = F.interpolate(raw_depth, size=[depth_shape[0], depth_shape[1]])
    raw_depth = raw_depth.squeeze(0).squeeze(0).numpy()
    # img = img.permute(1, 2, 0).numpy()
    # img *= 255
    # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    mask = depth > 0
    h, w = depth.shape
    crop_mask = np.zeros_like(mask)
    crop_mask[int(0.40810811 * h): int(0.99189189 * h),
            int(0.03594771 * w): int(0.96405229 * w)] = 1
    c_mask = mask * crop_mask

    depth_count = depth[c_mask]
    # max_depth = np.max(depth_count)
    # normal_depth = (depth / max_depth) * 255
    for d in depth_count:
        d_int = int(d)
        u_depth_range[d_int] += 1
        insert_range(d)
        all_depth += d
    all_valid += len(depth_count)
    all_pix += h * w
    
    # depth_sub = depth - raw_depth

    # depth_count = depth[c_mask]
    # max_depth = np.max(depth_count)
    # normal_depth = (depth / max_depth) * 255

    # raw_depth_count = raw_depth[c_mask]
    # raw_max_depth = np.max(raw_depth_count)
    # raw_normal_depth = (raw_depth / raw_max_depth) * 255

    # depth_sub_count = depth_sub[c_mask]
    # min_depth_sub = np.min(depth_sub_count)
    # max_depth_sub = np.max(depth_sub_count)
    # if np.abs(min_depth_sub) > np.abs(max_depth_sub):
    #     normal_v = np.abs(min_depth_sub)
    # else:
    #     normal_v = np.abs(max_depth_sub)

    # img = cv.resize(img, (w, h))
    # normal_depth = cv.applyColorMap(normal_depth.astype(np.uint8), cv.COLORMAP_RAINBOW)
    # raw_normal_depth = cv.applyColorMap(raw_normal_depth.astype(np.uint8), cv.COLORMAP_RAINBOW)

    # normal_sub = mpl.colors.Normalize(vmin=-normal_v, vmax=normal_v)
    # mapper_sub = cm.ScalarMappable(norm=normal_sub , cmap='coolwarm')
    # sub_color = (mapper_sub.to_rgba(depth_sub)[:, :, :3] * 255).astype(np.uint8)
    # normal_depth_sub = cv.cvtColor(sub_color, cv.COLOR_RGB2BGR)
    # # show_1 = copy.deepcopy(normal_depth)
    # # show_2 = normal_depth
    # # show_1[~mask] = (0, 0, 0)
    # # show_2[~c_mask] = (0, 0, 0)
    # normal_depth[~c_mask] = (0, 0, 0)
    # raw_normal_depth[~c_mask] = (0, 0, 0)
    # #normal_depth_sub[~c_mask] = (0, 0, 0)

    # img_depth = img.astype(np.float) + normal_depth_sub.astype(np.float) * 1.5
    # max_img = np.max(img_depth)
    # img_depth = (img_depth / max_img * 255).astype(np.uint8)
    # show = np.vstack([img, img_depth, normal_depth_sub, normal_depth, raw_normal_depth])
    # cv.imwrite("/data/Train_Log/KITTI_visual/{}.png".format(idx),show)
    print(idx, end="\r")

with open("/data/Train_Log/KITTI_visual/count2.txt", "w") as f:
    for i in range(100):
        f.write(str(u_depth_range[i]) + "\n")
    for i in range(cha_num):
        f.write(str(i) + ":" + str(sid_depth_guide[i]) + ":" + str(sid_depth_range[i]) + "\n")
    f.write(str(all_valid) + "\n")
    f.write(str(all_pix) + "\n")

for i in range(100):
    print(u_depth_range[i], end=", ")
print("end")
for i in range(cha_num):
    print(sid_depth_range[i], end=", ")
print("end")
print(all_valid)
print(all_pix)
print(all_depth)