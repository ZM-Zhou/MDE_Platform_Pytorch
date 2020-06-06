import os

import numpy as np
import json
import random
from PIL import Image  # using pillow-simd for increased speed
import PIL.Image as pil
import cv2 as cv

import torch
import torch.nn.functional as F
from torchvision import transforms

from _Dataset.cityscapes_utils import *

class Dataset_Options():
    def __init__(self):
        self.data_dir = "../Datasets/CS"
        self.width = 1242
        self.height = 375
        self.height_crop = 768
        self.width_crop = 48
        self.label_used = "depth"
        self.img_aug = True
        self.T_x = -0.022

class CSColorDepthSegDataset:
    """
    Class for self supervised monocular depth estimation with segmentation
    by stereo
    The following parameters should be included in opts:
    data_dir:path to dataset(include in base opts)
    height, width: output iamges' size
    min_depth, max_depth: min and max depth of gt 
    height_crop, width_crop: the height aftr crop
    label_used: segment label or depth label will be included in output
    img_aug: do image augment and flip
    """
    def __init__(self, opts, mode):
        self.opts = opts
        self.data_dir = self.opts.data_dir
        self.height = self.opts.height
        self.width = self.opts.width
        self.label_used = self.opts.label_used
        self.img_aug = self.opts.img_aug

        self.width_crop = self.opts.width_crop
        self.height_crop = self.opts.height_crop

        if self.label_used != "segment":
            self.min_depth = 1
            self.max_depth = 100
            self.K, self.inv_K, self.T = self.get_K_T()
        self.mode = mode
        self.filenames = self.get_filenames()
        if self.img_aug:
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

        self.to_tensor = transforms.ToTensor()
        self.resize_color = transforms.Resize((self.height, self.width),
                                               interpolation=Image.ANTIALIAS)

        if self.mode == "test":
            self.path_mode = "val"
        elif self.mode == "sample" or self.mode == "choice":
            self.path_mode = "val"
        else:
            self.path_mode = self.mode
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        '''
        return inputs:
        ('color(_aug)', 'l/r'): left or right color image with augment[3, H, W]
        "seg_mask": semantic gt include 19+1 channels for 19 classes[20, H, W]
        "seg_label": semantic gt just 1 channel for 19 classes[1, H, W]
        "depth_gt": depth compute from disparity gt in [0.1, 100][1, H, W]
        "K/inv_K": intrinsic matrix
        "stereo_T": extrinsic matrix
        the seg_mask and seg_label are aligned with left color image. if do flip,
        the image left right will not be changed, but stereo_T[0, 3] = -stereo_T[0, 3].
        '''
        inputs = {}

        self.do_color_aug = self.mode == "train" and\
            self.img_aug and random.random() > 0.5
        self.do_flip = self.mode == "train" and\
             self.img_aug and random.random() > 0.5
        file_dict = self.filenames[index]
        img_name = file_dict["img"]

        # create input image(left)
        sence_name = img_name.split("_")[0]
        if self.mode == "train" and self.label_used == "depth":
            train_set = file_dict["train_set"]
            left_img_path = os.path.join(self.data_dir, "leftImg8bit",
                                         train_set, sence_name, 
                                        img_name + "_leftImg8bit.png")
        else:
            left_img_path = os.path.join(self.data_dir, "leftImg8bit",
                                         self.path_mode, sence_name, 
                                        img_name + "_leftImg8bit.png")
        inputs[("color", "l", -1)] = self.get_color(left_img_path,
                                                    self.do_flip)
        
        #create ground-truth depth
        if self.label_used != "segment":
            right_img_path = left_img_path.replace("leftImg8bit",
                                                   "rightImg8bit")
            inputs[("color", "r", -1)] = self.get_color(right_img_path,
                                                        self.do_flip)
            disp_img_path = left_img_path.replace("leftImg8bit", "disparity")
            disp_np = self.get_disp(disp_img_path, self.do_flip)
            disp_max = 32257 #
            disp_map = np.clip((disp_np/disp_max), 0.0, 1.0)
            _, inputs["depth_gt"] = disp_to_depth(disp_map, self.min_depth,
                                                  self.max_depth)
            depth_gt_mask = inputs["depth_gt"] > 97
            inputs["depth_gt"][depth_gt_mask] = 0
            inputs["depth_gt"] = self.to_tensor(inputs["depth_gt"])

            inputs["disp"] = self.to_tensor(disp_map.astype(np.float32))

        # create seg mask and seg label (for semantic)
        if self.label_used != "depth":
            gt_path = os.path.join(self.data_dir, "gtFine", 
                                "{}".format(self.path_mode), sence_name,
                                img_name + "_gtFine_labelTrainIds.png")
            
            gt = self.get_disp(gt_path, self.do_flip)
            ignore_mask = gt == 255
            gt[ignore_mask] = 19
            seg_label = gt
            # seg_mask =  np.zeros(
            #         (gt.shape[0], gt.shape[1], 20), dtype=np.float
            #     )
            # traIDs = np.unique(seg_label)
            # for traID in traIDs:
            #     mask = seg_label == traID
            #     seg_mask[:, :, traID][mask] = 1
            # inputs["seg_mask"] = self.to_tensor(seg_mask)
            inputs["seg_label"] = self.to_tensor(seg_label.astype(np.float))
            
        if self.do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)

        if self.mode == "sample":
            inputs["ori_image"] = inputs[("color", "l", -1)]

        del inputs[("color", 'l', -1)]
        if self.label_used != "segment":
            del inputs[("color", 'r', -1)]

        # create auxiliary inputs
        if self.label_used != "segment":
            inputs["K"] = torch.from_numpy(self.K).to(torch.float)
            inputs["inv_K"] = torch.from_numpy(self.inv_K).to(torch.float)
            inputs["stereo_T"] = torch.from_numpy(self.T).to(torch.float)
            if self.do_flip:
                inputs["stereo_T"][0, 3] = -inputs["stereo_T"][0, 3]

        if self.label_used != "depth":
            inputs["seg_label"] = inputs["seg_label"].squeeze(0)\
                .to(torch.long)
            
        if self.mode == "test":
            inputs["img_info"] = img_name
        inputs["visual"] = False

        return inputs

    
    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                f = cv.resize(f, dsize=(self.width, self.height))
                f = Image.fromarray(f)
                inputs[(n, im)] = self.to_tensor(f).to(torch.float)
                inputs[(n + "_aug", im)] = self.to_tensor(color_aug(f))\
                    .to(torch.float)
            # elif "seg" in k:
            #     f = F.interpolate(f.unsqueeze(0),
            #                       size=(self.height, self.width))\
            #         .to(torch.float)
            #     inputs[k] = f.squeeze(0)
            else:
                f = F.interpolate(f.unsqueeze(0),
                                  size=(384, 1024))\
                    .to(torch.float)
                inputs[k] = f.squeeze(0)
    
    def get_color(self, path, do_flip):
        img = cv.imread(path)[0:self.height_crop, self.width_crop:2048, :]
        color = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if do_flip:
            color = np.ascontiguousarray(np.fliplr(color))
        return color

    def get_disp(self, path, do_flip):
        disp = cv.imread(path, flags=-1)[0:self.height_crop,
                                         self.width_crop:2048]
        if do_flip:
            disp = np.ascontiguousarray(np.fliplr(disp))
        return disp
        
    def get_filenames(self):
        phase_name = "cityscapes_{}_list"
        # use extra datas if just train with depth
        if self.label_used == "depth":
            if self.mode == "train":
                data_path = os.path.join("./_Dataset", "cityscapes_split",
                                         phase_name.format("train_extra"))
                f = open(data_path + ".txt", 'r')
                files = f.readlines()
                filenames = []
                for file in files:
                    file_name, train_set = file.replace("\n", "").split(" ")
                    filenames.append({"img":file_name,
                                      "train_set": train_set})
                return filenames

        if self.mode == "train":
            data_path = os.path.join("./_Dataset", "cityscapes_split",
                                     phase_name.format("train"))
        elif self.mode == "val":
            data_path = os.path.join("./_Dataset", "cityscapes_split",
                                     phase_name.format("val"))
        elif self.mode == "test":
            data_path = os.path.join("./_Dataset", "cityscapes_split",
                                     phase_name.format("val"))
        elif self.mode == "sample":
            data_path = os.path.join("./_Dataset", "cityscapes_split",
                                     phase_name.format("sample"))
        else:  # choice
            data_path = os.path.join("./_Dataset", "cityscapes_split",
                                     phase_name.format("choice"))
 
        f = open(data_path + ".txt", 'r')
        files = f.readlines()
        filenames = []
        for file in files:
            filenames.append({"img":file.replace("\n", "")})
        f.close()
        return filenames

    
    def get_K_T(self):
        camera_info_path = os.path.join(self.data_dir, "camera.json")
        f = open(camera_info_path, "r")
        camera_info = json.load(f)
        intr = camera_info["intrinsic"]
        K = np.zeros((4, 4))
        K[0, 0] = intr["fx"]
        K[0, 2] = intr["u0"] - self.width_crop
        K[1, 1] = intr["fy"]
        K[1, 2] = intr["v0"]
        K[2, 2] = 1.0
        K[3, 3] = 1.0
        K[1,:] *= self.height / (1024 - self.height_crop)
        K[0,:] *= self.width / (2048 - self.width_crop)
        inv_K = np.linalg.pinv(K)
        T = np.eye(4)
        #min depth =0.1 max_depth = 100
        #crop_width=48 -0.008 crop_width=0 -0.0055
        T[0, 3] = self.opts.T_x
        return K, inv_K, T
    
    def evaluation(self, inputs, outputs, losses=None, visual=None):
        if self.opts.label_used == "depth" or\
            self.opts.label_used == "both":
            depth_pred = outputs[("depth", 0)]

            depth_gt = inputs["depth_gt"]
            mask = depth_gt > 0

            # seg_label = inputs["seg_label"]
            # mask1 = seg_label == 5
            # mask2 = seg_label == 6
            # mask3 = seg_label == 7
            # mask4 = seg_label == 8
            # seg_mask = mask1 + mask2 + mask3 + mask4

            depth_pred = depth_pred.detach()
            depth_pred = F.interpolate(depth_pred,
                                       [384, 1024],
                                       mode="bilinear",
                                       align_corners=False)

            depth_gt_mask = depth_gt[mask]
            depth_pred_mask = depth_pred[mask]
            depth_k = torch.median(depth_gt_mask) / torch.median(depth_pred_mask)
            # depth_k = -0.022 / self.opts.T_x
            depth_pred_mask  *= depth_k
            depth_pred = torch.clamp(depth_pred, 1, 60)
            depth_errors = compute_depth_errors(depth_gt_mask, depth_pred_mask)

            # mask = mask * seg_mask
            # depth_gt_mask = depth_gt[mask]
            # depth_pred_mask = depth_pred[mask]
            # # depth_k = torch.median(depth_gt_mask) / torch.median(depth_pred_mask)
            # # depth_k = -0.022 / self.opts.T_x
            # depth_pred_mask  *= depth_k
            # depth_pred = torch.clamp(depth_pred, 1, 60)
            # depth_errors_seg = compute_depth_errors(depth_gt_mask, depth_pred_mask)

            depth_metric_names = ["abs_rel", "sq_rel", "rms", "log_rms",
                                "a1", "a2", "a3"]
            if losses is None:
                losses = {}
            if visual is not None:
                errors_map = (depth_pred * depth_k - depth_gt)\
                    * mask.to(torch.float)
                errors_map = errors_map[0, ...].detach()
                min_errors = torch.min(errors_map)
                max_errors = torch.max(errors_map)
                if torch.abs(min_errors) > max_errors:
                    errors_map -= min_errors
                    errors_map /= 2 * torch.abs(min_errors)
                else:
                    errors_map += max_errors
                    errors_map /= 2 * max_errors
                visual["depth_error"] = errors_map
            for i, metric in enumerate(depth_metric_names):
                losses["eval/" + metric] = np.array(depth_errors[i].cpu())
                # losses["evalseg/" + metric] = np.array(depth_errors_seg[i].cpu())

            return depth_errors[0].cpu().data, losses, visual



# visualize mask
        # depth = inputs["depth_gt"].squeeze(0).detach().numpy()
        # invalid_mask = depth > 98
        # invalid_mask2 = depth == 0
        # invalid_mask = np.logical_or(invalid_mask, invalid_mask2)
        # pic = np.zeros([384, 1024, 3])
        # pic[invalid_mask] = (255, 255, 255)
        # inpic = inputs[("color", "l", -1)][0:768, :, :]
        # inpic = cv.resize(inpic, (1024, 384))
        # inpic[invalid_mask] = inpic[invalid_mask] - (125, 125, 125)
        # show_pic = np.vstack([inpic, pic])
        # cv.imshow("disp_invalid", show_pic.astype(np.uint8))
        # cv.waitKey(0)
            
        



