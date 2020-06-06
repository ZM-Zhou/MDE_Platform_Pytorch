import numpy as np
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from _Costfunc.base_confunc import compute_reprojection_map,\
    compute_smooth_map, SSIM, BackprojectDepth,\
    Project3D, Disp_point
from Utils.visualization import visualize_depth

class CostFuncOpts():
    def __init__(self):
        self.height = 192
        self.width = 640
        self.no_cuda = False
        self.batch_size = 12

        self.use_depth = True
        self.scales = [3, 2, 1, 0]
        # for photometric loss and consis loss
        self.is_automask = False
        self.is_blendmask = False
        self.is_occmask = False
        self.pool_smooth = 0
        # consistency loss
        self.is_lr = False #left-0 right-1
        self.is_consis = False
        # occlusion regularization (True if use occmask)
        # self.is_occreg = False
        # std regularization
        self.is_std = False
        # GAN loss
        self.is_GAN = False
        # rates
        self.a_smooth = 0.001
        self.a_consis = 1
        self.a_occ = 0.1
        self.a_std = 1
        self.a_adv = 0.001
       


class NormalUnsuperCost():
    def __init__(self, opts, logger):
        self.opts = opts
        self.logger = logger
        self.device = self.logger.device

        h = int(self.opts.height)
        w = int(self.opts.width)
       
        self.ssim = SSIM()
        self.ssim.to(self.device)

        if self.opts.use_depth:
            self.backproject_depth = BackprojectDepth(self.opts.batch_size, h, w)
            self.backproject_depth.to(self.device)
            self.project_3d = Project3D(self.opts.batch_size, h, w)
            self.project_3d.to(self.device)
        else:
            self.get_coord = Disp_point(self.opts.batch_size, h, w)
            self.get_coord.to(self.device)

        if self.opts.pool_smooth != 0:
            self.ave_pool = nn.AvgPool2d(self.opts.pool_smooth, 1)
            pad_width = int((self.opts.pool_smooth-1) / 2)
            self.pool_pad = nn.ReplicationPad2d((pad_width, pad_width, pad_width, pad_width))

        self.other_side = {"l": "r", "r": "l"}
        self.side = ["l"]
        if self.opts.is_lr:
            self.side = ["l", "r"]

        self.max_error = 0
        if logger.is_train:           
            self.logger.make_logdir("Error")
        self.logger.make_logdir("Mask")

    def compute_losses(self, inputs, outputs, stage):

        losses = {}
        total_loss = []
        reproj_loss = []
        smooth_loss = []
        consis_loss = []
        occ_loss = []
        std_loss = []
        losses["loss"] = 0

        for idx, side in enumerate(self.side):
            # color_target = inputs[("color_aug", side)]
            # color_source = inputs[("color_aug", self.other_side[side])]
            color_target = inputs[("color", side)]
            color_source = inputs[("color", self.other_side[side])]
            T = inputs["stereo_T"].clone()
            if side == "r":
                T[:, 0, 3] = -T[:, 0, 3]
            
            total_loss.append(torch.zeros([]).to(color_target))
            reproj_loss.append(torch.zeros([]).to(color_target))
            smooth_loss.append(torch.zeros([]).to(color_target))
            if self.opts.is_consis:
                consis_loss.append(torch.zeros([]).to(color_target))
            if self.opts.is_occmask:
                occ_loss.append(torch.zeros([]).to(color_target))
            if self.opts.is_std:
                std_loss.append(torch.zeros([]).to(color_target))

            for scale in self.opts.scales:
                total_losses = 0
                # generate estimated image
                disp = outputs[("disp", scale)]
                disp_target = disp[:, idx,...].unsqueeze(1)
                disp_target = F.interpolate(disp_target,
                                            scale_factor=2**scale,
                                            mode='bilinear')
                depth = outputs[("depth", scale)]
                depth_target = depth[:, idx,...].unsqueeze(1)
                depth_target = F.interpolate(depth_target,
                                             scale_factor=2**scale,
                                             mode='bilinear')
                if self.opts.use_depth:
                    cam_points = self.backproject_depth(depth_target, inputs["inv_K"])
                    pix_coords = self.project_3d(cam_points, inputs["K"], T)
                else:
                    pix_coords = self.get_coord(disp_target, T)
                est_target = F.grid_sample(color_source,
                                           pix_coords,
                                           padding_mode="border")              

                # compute losses
                reprojection_map = compute_reprojection_map(est_target,
                                                            color_target,
                                                            self.ssim)
                if self.opts.pool_smooth != 0:
                    guide_map = self.ave_pool(self.pool_pad(color_target))
                else:
                    guide_map = color_target
                smoothness_map = compute_smooth_map(disp_target, guide_map)
                if self.opts.is_consis:
                    disp_source = disp[:, 1 - idx,...].unsqueeze(1)
                    est_disp = F.grid_sample(disp_source,
                                             pix_coords,
                                             padding_mode="border")
                    consistency_map = compute_reprojection_map(est_disp,
                                                               disp_target)
                    
                    
                # apply masks
                if self.opts.is_blendmask:
                    mask1 = pix_coords < 1
                    mask2 = pix_coords > -1
                    blend_mask = torch.cat([mask1, mask2], dim=3).to(torch.float)
                    blend_mask, _ = torch.min(blend_mask, dim=3, keepdim=True)
                    blend_mask = blend_mask.permute(0, 3, 1, 2)
                    reprojection_map = reprojection_map * blend_mask

                if self.opts.is_occmask:
                    ident_reprojection_map = compute_reprojection_map(est_target,
                                                                      color_source,
                                                                      self.ssim)
                    ident_reprojection_map += torch.randn(
                        ident_reprojection_map.shape).to(self.device) * 0.00001
                    raw_rep_map = reprojection_map
                    combined = torch.cat((ident_reprojection_map.detach(), reprojection_map), dim=1)
                    reprojection_map, idxs = torch.min(combined, dim=1, keepdim=True)
                    outputs["ident_selection/{}".format(scale)] = (
                        idxs > ident_reprojection_map.shape[1] - 1).float()
                    occreg_map = torch.exp(raw_rep_map - ident_reprojection_map) * (raw_rep_map.detach())
                    occreg_losses = occreg_map.mean()

                
                if self.opts.is_automask:
                    ident_reprojection_map = compute_reprojection_map(color_source,
                                                                     color_target,
                                                                     self.ssim)
                    ident_reprojection_map += torch.randn(
                        ident_reprojection_map.shape).to(self.device) * 0.00001
                    combined = torch.cat((ident_reprojection_map, reprojection_map), dim=1)
                    reprojection_map, idxs = torch.min(combined, dim=1, keepdim=True)
                    outputs["ident_selection/{}".format(scale)] = (
                        idxs > ident_reprojection_map.shape[1] - 1).float()

                # compute final loss  
                reprojection_losses = reprojection_map.mean()
                reproj_loss[idx] += reprojection_losses
                smoothness_losses = smoothness_map.mean()
                smooth_loss[idx] += smoothness_losses
                total_losses = reprojection_losses + self.opts.a_smooth * smoothness_losses / (2 ** scale)
                if self.opts.is_consis:
                    consistency_losses = consistency_map.mean()
                    consis_loss[idx] += consistency_losses
                    total_losses +=  self.opts.a_consis * consistency_losses
                

                if self.opts.is_occmask:
                    occ_loss[idx] += occreg_losses
                    total_losses += self.opts.a_occ * occreg_losses
                if self.opts.is_std:
                    std_losses = outputs[("std", scale)].mean()
                    std_losses = -torch.log(std_losses/2)
                    std_loss[idx] += std_losses
                    total_losses += self.opts.a_std * std_losses
                
                total_loss[idx] += total_losses
                losses["loss/{}_{}".format(side, scale)] = total_losses

                # visualization
                if stage.is_visual and scale == 0:
                    visual_dict = OrderedDict()
                    visual_dict["est_left_img"] = est_target.detach()
                    visual_dict["left_img"] = color_target.detach()
                    visual_dict["disp"] = disp_target.detach()
                    if self.opts.is_lr:
                        visual_dict["disp"] = visual_dict["disp"][:, 0, ...].unsqueeze(0)
                    
                    errors_map = reprojection_map
                    errors_map_max = torch.max(errors_map)
                    if self.max_error < errors_map_max:
                        self.max_error = errors_map_max
                    errors_map /= self.max_error
                    visual_dict["photo_error"] = errors_map.detach()

                    visual_dict["depth"] = depth_target.detach()

                    if self.logger.is_train:
                        v_modes = ["img", "img", "disp", "error_heat", "depth"]
                        v_size = [["left_img", "depth"],
                                  ["est_left_img", ["left_img", "est_left_img", 1]],
                                  ["disp", ["left_img", "disp", 0.8]],
                                  ["photo_error", ["left_img", "photo_error", 0.5]]]
                        self.logger.do_visualizion("Error", visual_dict, v_modes, v_size, side)
                    
                    if not self.logger.is_train and side == "l":
                        visual_dict_show = visual_dict

                    if self.opts.is_automask or self.opts.is_occmask:
                        v_mask_dict = OrderedDict()
                        v_mask_dict["idt_map"] = outputs["ident_selection/0"].detach()
                        v_mask_dict["left_img"] = color_target.detach()
                        v_mask_dict["right_img"] = color_source.detach()
                        v_mask_dict["est_left"] = est_target.detach()

                        v_modes = ["mask", "img", "img", "img"]

                        if self.opts.is_occmask:
                            show_reg = occreg_map.detach()
                            reg_max = np.max(show_reg)
                            reg_nor = show_reg / reg_max
                            v_mask_dict["reg_mask"] = reg_nor
                            v_modes.append("error_heat")
                            v_size = [["left_img", "left_img","left_img"],
                                      ["reg_mask", "idt_map", ["left_img", "est_left", 1]],
                                      [["left_img", "reg_mask", 1], ["left_img", "idt_map", 1], ["left_img", "right_img", 1]]]
                           
                        else:
                            v_size = [["left_img","left_img"],
                                      ["idt_map", ["left_img", "est_left", 1]],
                                      [["left_img", "idt_map", 1], ["left_img", "right_img", 1]]]

                        self.logger.do_visualizion("Mask", v_mask_dict, v_modes, v_size)

            total_loss[idx] /= len(self.opts.scales)
            reproj_loss[idx] /= len(self.opts.scales)
            smooth_loss[idx] /= len(self.opts.scales)
            if self.opts.is_consis:
                consis_loss[idx] /= len(self.opts.scales)
            if self.opts.is_occmask:
                occ_loss[idx] /= len(self.opts.scales)
            if self.opts.is_std:
                std_loss[idx] /= len(self.opts.scales)

            losses["loss/{}".format(side)] = total_loss[idx]
            losses["loss/reproj_{}".format(side)] = reproj_loss[idx]
            losses["loss/smooth_{}".format(side)] = smooth_loss[idx]
            if self.opts.is_consis:
                losses["loss/consis_{}".format(side)] = consis_loss[idx]
            if self.opts.is_occmask:
                losses["loss/occreg_{}".format(side)] = occ_loss[idx]
            if self.opts.is_std:
                losses["loss/std_{}".format(side)] = std_loss[idx]
            losses["loss"] += total_loss[idx]

        if self.opts.is_GAN:
            all_val = range(self.opts.batch_size)
            ex_list = torch.rand_like(outputs["real_score"])
            d_real = torch.where(ex_list > 0.95, outputs["fake_score"], outputs["real_score"])
            d_fake = torch.where(ex_list > 0.95, outputs["real_score"], outputs["fake_score"])

            discriminate_losses = -(0.5 * torch.log(d_real) + 
                                    0.5 * torch.log(1 - d_fake))
            generate_losses =  -torch.log(outputs["fake_score"]).mean()
            losses["D_loss"] = discriminate_losses.mean()
            losses["loss"] += self.opts.a_adv * generate_losses
            losses["loss/adv"] = generate_losses

        if stage.is_visual and not self.logger.is_train:
            return losses, visual_dict_show
        else:
            return losses
