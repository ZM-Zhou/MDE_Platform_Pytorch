import time
import os
import json
import shutil
import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.import_choice import import_module
from Utils.visualization import VisualImage, make_output_img

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


class MetricLog():
    def __init__(self, weights_path, data_name):
        self.weights_path = weights_path
        path_split = weights_path.split("/")
        log_dir = ""
        for name in path_split[:-2]:
            log_dir += name + '/'
        self.log_path = os.path.join(log_dir,
                                     "{}_depth_".format(data_name) + path_split[-1])
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        self.output_file = os.path.join(self.log_path, 'output.txt')
        self.visual_tool = VisualImage()

        self.device = "cpu"
        self.is_train = False
        self.output_count = 0
    
    def log_print(self, arg, *args, **kargs):
        with open(self.output_file, "a+") as f:
            if kargs:
                print(arg, end=kargs["end"])
                f.write(arg + kargs["end"])
            else:
                print(arg)
                f.write(arg + "\n")

    def make_logdir(self, dir_name):
        path = os.path.join(self.log_path, dir_name)
        if not os.path.isdir(path):
            os.makedirs(path)

    def do_visualizion(self, dir_name, imgs, visual_modes, size, name=""):
        if name == "":
            save_path = os.path.join(self.log_path, dir_name,
                                     str(self.output_count) + ".png")
            self.output_count += 1
        else:
            save_path = os.path.join(self.log_path, dir_name,
                                     name + ".png")
        for k, v in imgs.items():
            tar_img = v
            break
        _, _, h, w = tar_img.shape
        for idx, (k, v) in enumerate(imgs.items()):
            v = F.interpolate(v, [h, w], mode="bilinear",
                              align_corners=False)
            v = v[0, ...].cpu().permute(1, 2, 0).numpy()
            imgs[k] = self.visual_tool.do_visualize(v, visual_modes[idx])

        img = make_output_img(imgs, size)
        cv.imwrite(save_path, img)


    # for trainer
    def do_log_all_test(self, use_time, total, data_num):
        self.log_print('Done in {} seconds!'.format(sec_to_hm_str(use_time)))
        for k, v, in total.items():
                total[k] = v / data_num
                self.log_print("-->{}: {}".format(k, total[k]))     

    def load_models(self, networks):
        """Load model(s) from disk
        """
        self.log_print("Loading pretrained opts")
        for n, v in networks.items():
            self.log_print("Loading {} weights...".format(n))
            path = os.path.join(self.weights_path, "{}.pth".format(n))
            model_dict = networks[n].state_dict()
            try:
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict}
                model_dict.update(pretrained_dict)
                networks[n].load_state_dict(model_dict)
            except:
                self.log_print("{} is randomly initialized".format(n))

        return networks
