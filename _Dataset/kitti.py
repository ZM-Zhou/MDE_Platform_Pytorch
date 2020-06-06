import os
import numpy as np
from skimage import transform
import copy
import random
from PIL import Image  # using pillow-simd for increased speed
import PIL.Image as pil

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

from _Dataset.kitti_utils import *


class Dataset_Options():
    def __init__(self):
        self.data_dir = "../../Datasets/KITTI"
        self.width = 1242
        self.height = 375
        self.T_x = 0.1
        self.train_split = ""


class KittiColorDataset(Dataset):
    """
    Superclass for monocular depth estimation by stereo
    The following parameters should be included in opts:
        data_dir: path to dataset(include in base opts)
        height, width: output image size
    mode: dataset mode in "train", "val"， "test", "sample"
    (adapted from https://github.com/nianticlabs/monodepth2)
    """
    def __init__(self, opts, mode):
        super().__init__()
        self.opts = opts
        self.data_dir = self.opts.data_dir
        self.height = self.opts.height
        self.width = self.opts.width
        self.mode = mode
        self.interp = Image.ANTIALIAS  # 1

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1242, 375)

        self.filenames = self.get_filenames()

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize((self.height, self.width),
                                        interpolation=self.interp)
        self.load_depth = self.check_depth()
        if self.mode == "test":
            gt_path = os.path.join(self.opts.data_dir, "gt_depths.npz")
            self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]


    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                f = self.resize(f)
                inputs[(n, im)] = self.to_tensor(f)
                f_aug = color_aug(f)
                inputs[(n + "_aug", im)] = self.to_tensor(f_aug)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <left or right>)          for raw colour images,
            ("color_aug", <left or right>)      for augmented colour images,
            "K" or "inv_K"                      for camera intrinsics,
            "stereo_T"                          for camera extrinsics, and
            "depth_gt"                          for ground truth depth maps.
        """
        inputs = {}

        do_color_aug = self.mode == "train" and random.random() > 0.5
        do_flip = self.mode == "train" and random.random() > 0.5

        line = self.filenames[index]
        # if in mode "test" or "sample" returen the image information for eval
        if self.mode == "test" or self.mode == "sample":
            inputs["img_info"] = line

        line = line.split()
        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        # if do flip exchange the right-left image
        other_side = {"r": "l", "l": "r"}[side]
        inputs[("color", other_side, -1)] = self.get_color(folder, frame_index,
                                                           other_side, do_flip)
        inputs[("color", side, -1)] = self.get_color(folder, frame_index,
                                                     side, do_flip)

        # return original image for visualization
        if self.mode == "sample" or self.mode == "choice" or self.mode == "compare":
            inputs["ori_image"] = np.array(inputs[("color", "l", -1)])

        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)
        inputs["K"] = torch.from_numpy(K)
        inputs["inv_K"] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)

        del inputs[("color", 'l', -1)]
        del inputs[("color", 'r', -1)]

        if self.mode == "test":
            gt_depth = torch.from_numpy(self.gt_depths[index]).unsqueeze(0).to(torch.float)
            gt_depth = F.interpolate(gt_depth.unsqueeze(0), [1242, 375], mode="nearest")
            inputs["depth_gt"] = gt_depth.squeeze(0)
        else:
            if self.load_depth:
                depth_gt = self.get_depth(folder, frame_index, side, do_flip)
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"]
                                                    .astype(np.float32))
        

        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = 1 if do_flip else -1
        stereo_T[0, 3] = baseline_sign * self.opts.T_x

        inputs["stereo_T"] = torch.from_numpy(stereo_T)
        inputs["visual"] = False

        return inputs

    def get_filenames(self):
        fname = "_files{}.txt".format(self.opts.train_split)
        fpath = os.path.join("./_Dataset", "kitti_split", "{}" + fname)
        if self.mode == "train":
            filenames = readlines(fpath.format("train"))
        elif self.mode == "val":
            filenames = readlines(fpath.format("val"))
        elif self.mode == 'test':
            filenames = readlines(fpath.format("test"))
        elif self.mode == 'sample':
            filenames = readlines(fpath.format("sample"))
        elif self.mode == "choice":
            filenames = readlines(fpath.format("choice"))
        else:  # compare
            filenames = readlines(fpath.format("compare"))
        for index, train_filename in enumerate(filenames):
            filenames[index] = train_filename.replace('\n', '')
        return filenames

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}.{}".format(frame_index, "png")
        image_path = os.path.join(self.data_dir, folder,
                                  "image_0{}/data".format(self.side_map[side]),
                                  f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])
        velo_filename = os.path.join(
            self.data_dir,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        return os.path.isfile(velo_filename)

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_dir, folder.split("/")[0])
        velo_filename = os.path.join(
            self.data_dir,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        depth_gt = generate_depth_map(calib_path, velo_filename,
                                      self.side_map[side])
        depth_gt = transform.resize(depth_gt,
                                            self.full_res_shape[::-1],
                                            order=0, preserve_range=True,
                                            mode='constant')
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt

    def evaluation(self, inputs, outputs, losses=None, visual=None):
        depth_pred = outputs[("depth", 0)]
        if depth_pred.size()[1] == 2:
            depth_pred = depth_pred[:, 0, ...].unsqueeze(1)

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0
        n, c, h, w = depth_gt.size()

        depth_pred = F.interpolate(depth_pred, [h, w], mode="bilinear",
                                   align_corners=False)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, int(0.40810811 * h): int(0.99189189 * h),
                  int(0.03594771 * w): int(0.96405229 * w)] = 1
        mask = mask * crop_mask

        depth_pred = depth_pred.detach()
        

        depth_gt_mask = depth_gt[mask]
        depth_pred_mask = depth_pred[mask]
        # depth_k = torch.median(depth_gt_mask) / torch.median(depth_pred_mask)
        # print(depth_k.data)
        depth_k = 0.54 / self.opts.T_x
        depth_pred_mask  *= depth_k
        depth_pred_mask = torch.clamp(depth_pred_mask, 1e-3, 80)

        depth_errors = compute_depth_errors(depth_gt_mask, depth_pred_mask)

        depth_metric_names = ["abs_rel", "sq_rel", "rms", "log_rms",
                              "a1", "a2", "a3"]

        if losses is None:
            losses = {}
        if visual is not None:
            errors_map = (depth_pred * depth_k - depth_gt)\
                * mask.to(torch.float)
            errors_map = errors_map.detach()
            visual["depth"] = errors_map
        for i, metric in enumerate(depth_metric_names):
            losses["eval/" + metric] = np.array(depth_errors[i].cpu())
        

        return depth_errors[0].cpu().data, losses, visual
