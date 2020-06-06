import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from _Network.base_module import Network_Opts_base, Network_base,\
                                 get_depth_sid, disp_to_depth,\
                                 ResNet_Encoder_flex, DepthDecoder_forDilaEncoder,\
                                 ASPP

class Network_Opts(Network_Opts_base):
    def __init__(self):
        super().__init__()
        self.max_depth = 100
        self.min_depth = 0.5

        self.width = 640
        self.height = 192
        self.no_cuda = False
        self.batch_size = 2

        self.use_aspp = False

        self.use_depth = True
        self.fuse_depth = True
        self.depth_range = 64

        self.scales = [3, 2, 1, 0]


class BaseMonoDepthEstNetwork(Network_base):
    """The following parameters should be included in opts:
    min_depth, max_depth: min and max depth in predict
    """
    def __init__(self, opts, *args, **kargs):
        super().__init__(*args, **kargs)
        self.opts = opts
        self.device = self.logger.device

        num_ch_enc = [64, 256, 512, 1024, 2048]
        layer_num = 50

        channel = [int(self.opts.depth_range / 4)]
        self.channel = channel * 4
                    
        if self.opts.use_aspp:
            self.networks["Encoder"] = ResNet_Encoder_flex(50, drop_last=True)
            self.networks["ASPP"] = ASPP("res50-3", 16)
            num_ch_enc = [64, 256, 512, 1024, 256]
            self.networks["Decoder"] = DepthDecoder_forDilaEncoder(num_ch_enc,
                                                                   use_dila = True,
                                                                   num_output_channels=self.channel,
                                                                   is_sigmoid=self.opts.depth_range==4,
                                                                   more_features=False)
        else:
            self.networks["Encoder"] = ResNet_Encoder(layer_num)
            self.networks["Decoder"] = DepthDecoder_forDilaEncoder(num_ch_enc,
                                                                   use_dila = False,
                                                                   num_output_channels=self.channel,
                                                                   is_sigmoid=self.opts.depth_range==4,
                                                                   more_features=False)

        self.all_net = nn.ModuleList(v for k, v in self.networks.items())

        depth_range = [k for k in range(self.opts.depth_range)]
        range_np = np.array(depth_range)
        range_map = np.tile(range_np,(self.opts.height, self.opts.width, 1))
        depth_range_full = torch.from_numpy(range_map).permute(2, 0, 1)\
            .to(self.logger.device)   
        self.depth_range_full = depth_range_full
         
        self.logger.make_logdir("Disp")
        if not self.logger.is_train:
            self.logger.make_logdir("Dist")

    def forward(self, inputs, stage):
        image = inputs[("color_aug", "l")]
        if self.opts.use_aspp:
            features = self.networks["Encoder"](image)
            aspp_feature = self.networks["ASPP"](features[-1])
            features.append(aspp_feature)
            outputs = self.networks["Decoder"](features)
        else:
            features = self.networks["Encoder"](image)
            outputs = self.networks["Decoder"](features)

        if self.opts.fuse_depth:
            depth_fuse = []
            scale_orid = range(3, -1, -1)

            for scale in scale_orid:
                depth_result = outputs[("disp_feature", scale)]
                depth_result = F.interpolate(depth_result, scale_factor=2 ** scale,
                                             mode="bilinear", align_corners=False)
                depth_fuse.append(depth_result)
                

            depth_fuse = F.softmax(torch.cat(depth_fuse, dim=1), dim=1)
            outputs[("std", 0)] = depth_fuse[:, :, :, :].std(dim=1, keepdim=True)
            depth_fuse = depth_fuse * self.depth_range_full
            outputs[("mean", 0)] = depth_fuse[:, :, :, :].mean(dim=1, keepdim=True)
            depth_final = depth_fuse.sum(keepdim=True, dim=1)
            depth_final = get_depth_sid(depth_final, self.opts.min_depth,
                                        self.opts.max_depth, self.opts.depth_range - 1)
            outputs[("depth", 0)] = depth_final
            outputs[("disp", 0)] = 1 / depth_final

            for scale in self.opts.scales:
                if scale != 0:
                    outputs[("depth", scale)] = F.interpolate(depth_final, scale_factor=1 / 2**scale,
                                                              mode="bilinear", align_corners=False)
                    outputs[("disp", scale)] = F.interpolate(outputs[("disp", 0)], scale_factor=1 / 2**scale,
                                                             mode="bilinear", align_corners=False)      
                if scale != 0:
                    outputs[("std", scale)] =  outputs[("std", 0)]
        else:
            for scale in range(3, -1, -1):
                disp = outputs[("disp_feature", scale)]
                if self.opts.depth_range == 4:
                    _, depth = disp_to_depth(disp, self.opts.min_depth,
                                            self.opts.max_depth)
                else:
                    depth_result = F.softmax(disp, dim=1)
                    # outputs[("std", scale)] = depth_result.std(dim=1, keepdim=True)
                    # outputs[("mean", scale)] = depth_result.mean(dim=1, keepdim=True)
                    temp_range = self.depth_range_full[:self.channel[scale],...].unsqueeze(0).to(torch.float)
                    temp_range = F.interpolate(temp_range, scale_factor= 1 / 2 ** scale, mode="nearest")
                    depth_result = depth_result * temp_range
                    depth = depth_result.sum(keepdim=True, dim=1)
                    depth = get_depth_sid(depth, self.opts.min_depth,
                                                self.opts.max_depth, int(self.opts.depth_range/4) - 1)
                    disp = 1 / depth

                outputs[("depth", scale)] = depth
                outputs[("disp", scale)] = disp

        if stage.is_visual:
            if self.opts.fuse_depth:
                v_dict = OrderedDict()
                v_modes = []
                v_size = []

                v_dict["show_img"] = image.detach()
                v_modes.append("img")
                v_size.append(["show_img"])
                ch = int(self.opts.depth_range / 4)
                for scale in range(4):
                    depth = depth_fuse[:, ch * scale: ch * (scale + 1), ...]
                    depth = depth.sum(keepdim=True, dim=1)
                    depth = get_depth_sid(depth, self.opts.min_depth,
                                            self.opts.max_depth, self.opts.depth_range - 1)
                    v_dict["depth{}".format(scale)] = depth.detach()
                    v_modes.append("depth")
                    v_size.append(["depth{}".format(scale)])
                v_dict["depth_full"] = outputs[("depth", 0)].detach() 
                v_modes.append("depth")
                v_size.append(["depth_full"])
               
                self.logger.do_visualizion("Disp", v_dict, v_modes, v_size)

                if not self.logger.is_train:
                    v_dist_dict = OrderedDict()
                    v_modes = []

                    v_dist_dict["show_img"] = image.detach()
                    v_modes.append("img")
                    v_size = [["show_img", "show_img"]]
                    stds = []
                    v_dist_dict["mean"] =outputs[("mean", 0)].detach()
                    v_modes.append("depth")
                    v_dist_dict["std"] = outputs[("std", 0)].detach()
                    v_modes.append("error_heat")
                    v_size.append(["mean", "std"])
                        

                    self.logger.do_visualizion("Dist", v_dist_dict, v_modes, v_size)
                    
            else:
                if self.logger.is_train:
                    self.logger.log_print(self.logger.step)
                v_dict = OrderedDict()
                v_modes = []
                v_size = [["img"]]

                v_dict["img"] = image.detach()
                v_modes.append("img")
                for scale in range(0, 4):
                    show_disp = outputs[("disp", scale)].detach()
                    if self.logger.is_train:
                        self.logger.log_print(show_disp.mean())
                    v_dict["disp_{}".format(scale)] = show_disp
                    v_modes.append("disp")
                    v_size.append(["disp_{}".format(scale)])

                self.logger.do_visualizion("Disp", v_dict, v_modes, v_size, "disp")
                if self.logger.is_train:
                    self.logger.log_print("==========")
                    
        return outputs

    def check_info(self):
        assert self.opts.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opts.width % 32 == 0, "'width' must be a multiple of 32"

    def get_trainable_params(self):
        names = ["Encoder", "Decoder"]
        muls = [1, 1]
        if self.opts.use_aspp:
            names.append("ASPP")
            muls.append(1)
        return self.get_module_params(names, muls)
