import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter


class Network_Opts_base():
    def __init__(self):
        self.learning_rate = 1e-4
        self.height = 192
        self.width = 640


class Network_base(nn.Module):
    def __init__(self, logger=None):
        super(Network_base, self).__init__()
        self.networks = {}
        self.logger = logger

    def forward(self, inputs, stage):
        # your flow
        raise NotImplementedError

    def check_info(self):
        # check & assart
        raise NotImplementedError

    def get_networks(self):
        return self.networks

    def get_trainable_params(self):
        # return params dict list
        raise NotImplementedError

    def get_module_params(self, modules_name, lr_multis, is_first=True):
        paras_dict = []
        name_dict = []
        index = 0
        for name, v in self.networks.items():
            if name in modules_name:
                paras_list = list(self.networks[name].parameters())
                for par in paras_list:
                    par.requires_grad = True
            else:
                if is_first:
                    paras_list = list(self.networks[name].parameters())
                    for par in paras_list:
                        par.requires_grad = False

        for name, mul in zip(modules_name, lr_multis):
            paras_list = []
            name_list = []
            paras_temp_list = list(self.networks[name].named_parameters())
            for n, par in paras_temp_list:
                name_list.append(n)
                paras_list.append(par)
            temp_dict = {"params": paras_list,
                         "lr": self.opts.learning_rate * mul}
            paras_dict.append(temp_dict)
            name_dict.append((name, name_list))
        return paras_dict, name_dict

    def set_train(self):
        for m in self.networks.values():
            m.train()

    def set_eval(self):
        for m in self.networks.values():
            m.eval()


##############################################################################
# Baseline Encoder use ResNet50
##############################################################################
class ResnetEncoder_fromV2(nn.Module):
    """Pytorch module for a resnet encoder
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, num_layers, pretrained):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers"
                             .format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        after_pool = self.encoder.maxpool(self.features[-1])
        self.features.append(self.encoder.layer1(after_pool))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class ResNet_Encoder_flex(nn.Module):
    """A more flexible encoder part can adapt to multiple pictures
       and change the conv structure
    """
    def __init__(self, layer_num=18, in_channels=3, dila=False, drop_last=False):
        super().__init__()
        self.drop_last = drop_last
        if layer_num == 18:
            resnet = models.resnet18(True)   
        elif layer_num == 50:
            resnet = models.resnet50(True)
        
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
            self._init_weight()
        else:
            self.conv1 = resnet.conv1

        self.maxpool = resnet.maxpool
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        if not drop_last:
            self.layer4 = resnet.layer4
            if dila:
                for n, m in self.layer4.named_modules():
                    if '0.conv1' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
  
    def forward(self, x):
        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        if not self.drop_last:
            x = self.layer4(x)
            features.append(x)
        return features

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        

##############################################################################
# Depth Decoder from Monodepth2
##############################################################################
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthDecoder(nn.Module):
    """Depth Decoder
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [self.upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
    
    def upsample(self, x):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=2, mode="nearest")


##############################################################################
# Depth Decoder for encdoer with dilated conv or ASPP
##############################################################################
class DepthDecoder_forDilaEncoder(nn.Module):
    def __init__(self, num_ch_enc, use_skips=True, num_output_channels=2, use_dila=True,
                 is_sigmoid=True, more_features=False):
        super().__init__()
        num_ch_dec = np.array([16, 32, 64, 128, 256])
        if more_features:
            if use_dila:
                num_ch_dec = np.array([128, 256, 256, 256, 256])
            else:
                num_ch_dec = np.array([32, 64, 128, 256, 512])
        self.use_skips = use_skips
        self.is_sigmoid = is_sigmoid
        self.use_dila = use_dila
        if not isinstance(num_output_channels, list):
            num_output_channels = [num_output_channels] * 4

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = num_ch_enc[-1] if i == 4 else num_ch_dec[i + 1]
            num_ch_out = num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = num_ch_dec[i]
            if use_skips and i > 0:
                num_ch_in += num_ch_enc[i - 1]
            num_ch_out = num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in range(4):
            self.convs[("dispconv", s)] = Conv3x3(num_ch_dec[s],
                                                    num_output_channels[s])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if i < 4:
                x = [self.upsample(x)]
            else:
                if self.use_dila:
                    x = [x]
                else:
                    x = [self.upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in range(4):
                disp = self.convs[("dispconv", i)](x)
                if self.is_sigmoid:
                    self.outputs[("disp_feature", i)] = self.sigmoid(disp)
                else:
                    self.outputs[("disp_feature", i)] = disp
        return self.outputs

    def upsample(self, x):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=2, mode="nearest")

##############################################################################
#ASPP Module
##############################################################################
class _ASPPModule(nn.Module):
    """from https://github.com/jfzhang95/pytorch-deeplab-xception
    """
    def __init__(self, inplanes, planes, kernel_size,
                 padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes,
                                     kernel_size=kernel_size,
                                     stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    """from https://github.com/jfzhang95/pytorch-deeplab-xception
    """
    def __init__(self, backbone, output_stride, BatchNorm = nn.BatchNorm2d,
                 is_drop=False):
        super(ASPP, self).__init__()
        if backbone == 'drn' or backbone == 'res18':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == "res50-3":
            inplanes = 1024
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0,
                                 dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1],
                                 dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2],
                                 dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3],
                                 dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1,
                                                       stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.aspprelu = nn.ReLU()
        if is_drop:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = DoNothing()
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.aspprelu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


##############################################################################
# Others
##############################################################################
class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the
    'additional considerations' section of the paper.
    from https://github.com/nianticlabs/monodepth2
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def get_depth_sid(depth, min_depth, max_depth, K):
    # print('label size:', labels.size())
    # depth = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * labels / K_)
    depth = min_depth * (max_depth / min_depth) ** (depth / K)
    # print(depth.size())
    return depth.float()