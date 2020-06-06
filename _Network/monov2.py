import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


from _Network.base_module import Network_Opts_base, Network_base,\
                                 disp_to_depth, ResnetEncoder_fromV2,\
                                 DepthDecoder
                                 
class Network_Opts(Network_Opts_base):
    def __init__(self):
        super().__init__()
        self.max_depth = 100
        self.min_depth = 0.1
        self.backbone = 18
        self.sample_mode = "nearest"

        self.use_depth = True


class BaseMonoDepthEstNetwork(Network_base):
    """The following parameters should be included in opts:
    min_depth, max_depth: min and max depth in predict
    """
    def __init__(self, opts, *args, **kargs):
        super().__init__(*args, **kargs)
        self.opts = opts

        self.networks["Encoder"] = ResnetEncoder_fromV2(self.opts.backbone,
                                                        True)
        if self.opts.backbone > 34:
            num_ch_enc = [64, 256, 512, 1024, 2048]
        else:
            num_ch_enc = [64, 64, 128, 256, 512]

        self.networks["Decoder"] = DepthDecoder(num_ch_enc,
                                                num_output_channels=1)

        self.all_net = nn.ModuleList(v for k, v in self.networks.items())

        if self.logger.is_train:           
            self.logger.make_logdir("Disp") 

    def forward(self, inputs, stage):
        image = inputs[("color_aug", "l")]
        features = self.networks["Encoder"](image)
        outputs = self.networks["Decoder"](features)

        if not self.opts.min_depth:
            K = inputs["K"]
            T = inputs["stereo_T"]
            w = image.size()[3]
            self.compute_min_depth(K, T, w)

        # compute depth from disp
        for scale in range(3, -1, -1):
            disp = outputs[("disp", scale)]
            if self.opts.use_depth:
                _, depth = disp_to_depth(disp, self.opts.min_depth,
                                        self.opts.max_depth)
            else:
                fx = inputs["K"][0, 0, 0]
                img_width = image.size()[-1]
                depth = fx/(img_width * 0.3 * disp + 1e-10)
            outputs[("depth", scale)] = depth

        if stage.is_visual and self.logger.is_train:
            self.logger.log_print(self.logger.step)
            v_dict = OrderedDict()
            v_modes = []
            v_size = [["img"]]

            v_dict["img"] = image.detach()
            v_modes.append("img")
            for scale in range(0, 4):
                show_disp = outputs[("disp", scale)].detach()
                self.logger.log_print(show_disp.mean())
                v_dict["disp_{}".format(scale)] = show_disp
                v_modes.append("disp")
                v_size.append(["disp_{}".format(scale)])

            self.logger.do_visualizion("Disp", v_dict, v_modes, v_size, "disp")
            self.logger.log_print("==========")

        return outputs

    def check_info(self):
        assert self.opts.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opts.width % 32 == 0, "'width' must be a multiple of 32"

    def get_trainable_params(self):
        names = ["Encoder", "Decoder"]
        muls = [1, 1]
        return self.get_module_params(names, muls)

    def compute_min_depth(self, K, T, w):
        fx = K[0, 0, 0]
        baseline = torch.abs(T[0, 0, 3])
        target_d = fx * baseline / (0.3 * w)
        disp_min = 1 / self.opts.max_depth
        disp_max = (1 - disp_min * 0.5 * target_d) / (0.5 * target_d)
        disp_max = disp_max.data
        self.opts.min_depth = 1 / disp_max 
        print(self.opts.min_depth)
