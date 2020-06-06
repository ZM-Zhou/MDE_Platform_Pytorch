import time
import os
import json
import shutil
import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# 需要pip install future,需要tensorboard版本小于1.14

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


class TrainLog():
    def __init__(self, opts):
        self.opts = opts
        stamp = time.time()
        date = time.localtime(stamp)
        self.format_date = time.strftime('%Y-%m-%d_%Hh%Mm%Ss', date)
        self.log_path = os.path.join(self.opts["t"].log_dir, self.format_date)
        os.makedirs(self.log_path)
        self.output_file = os.path.join(self.log_path, 'head_info.txt')
        self.visual_tool = VisualImage()

        self.device = "cpu"
        self.is_train = True
        self.step = 0
    
    def log_print(self, arg, *args, **kargs):
        with open(self.output_file, "a+") as f:
            if kargs:
                print(arg, end=kargs["end"])
                f.write(arg + kargs["end"])
            else:
                print(arg)
                f.write(str(arg) + "\n")

    def make_logdir(self, dir_name):
        path = os.path.join(self.log_path, dir_name)
        os.makedirs(path)

    def do_visualizion(self, dir_name, imgs, visual_modes, size, name=""):
        save_path = os.path.join(self.log_path, dir_name,
                                 "{}".format(self.step) + name + ".png")
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
    def do_log_before_train(self, train_dataset, valid_dataset, describles):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        self.writers = {}
        for mode in ["train", "valid"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path,
                                                            "board_log",
                                                            mode))

        self.log_print("==========")
        self.log_print(describles[0])
        self.log_print(describles[1] + "+" + describles[2] + "+"
                        + describles[3])
        self.log_print('There are {:d} training items'\
            .format(len(train_dataset)), end="")
        self.log_print(' and {:d} validation items'\
            .format(len(valid_dataset)))
        self.log_print("==========")
        for k, v in self.opts.items():
            self.log_print("Options " + k)
            for key, value in vars(v).items():
                self.log_print("{}: {}".format(key, value))
            self.log_print("==========")
        # put train log in a new file
        self.output_file = os.path.join(self.log_path, 'train_log.txt')

    def do_log_epoch(self, optimizer, name_list):
        self.log_print("Training:")
        for index, param in enumerate(optimizer.param_groups):
            self.log_print("for params group {}: lr {:.7f}"
                  .format(name_list[index][0], param['lr']))

    def do_log(self, batch_idx, duration, losses, lr,
               start_time, epoch, process_ratio, is_gan=False):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opts["t"].batch_size / duration
        time_sofar = time.time() - start_time
        training_time_left = process_ratio * time_sofar\
            if process_ratio > 0 else 0
        finish_time = time.strftime('%Y-%m-%d_%Hh%Mm%Ss',
                                    time.localtime(time.time()
                                                   + training_time_left))
        if is_gan:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} /D_loss: {:.5f} | time elapsed: {} | finish time: {}"
            self.log_print(print_string.format(epoch, batch_idx, samples_per_sec,
                                    losses['loss'].cpu().data,
                                    losses['D_loss'].cpu().data,
                                    sec_to_hm_str(time_sofar),
                                    finish_time))
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | finish time: {}"
            self.log_print(print_string.format(epoch, batch_idx, samples_per_sec,
                                    losses['loss'].cpu().data,
                                    sec_to_hm_str(time_sofar),
                                    finish_time))
        # write to tensorboard
        writer = self.writers['train']
        writer.add_scalar("First_pargro_Lr", lr, self.step)
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v.cpu().data, self.step)

    def do_log_valid(self, losses):
        writer = self.writers['valid']
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def do_log_validphase(self, acc):
        self.log_print("Do Valid:")
        self.log_print("Valid Acc: {}".format(acc))


    
    def do_grad_check(self, temp_params, param_names):
        error_count = 0
        param_groups_num = 0
        for index, params in enumerate(temp_params):
            param_groups_num += len(param_names[index][1])
            param = params["params"]
            for idx, p in enumerate(param):
                try:
                    if p.grad.to(torch.bool).any():
                        continue
                    # self.log_print(torch.sum(p.grad))
                    self.log_print("{}->{}:grad all Zero!".format(\
                                   param_names[index][0],\
                                   param_names[index][1][idx]))
                    error_count += 1
                except:
                    self.log_print("{}->{}:grad None!".format(\
                                   param_names[index][0],\
                                   param_names[index][1][idx]))
                    error_count += 1
                    pass
        if error_count != 0:
            self.log_print("step:{} {} in {} params are error!"
                           .format(self.step, error_count,
                                   param_groups_num) + "="*20)
            if error_count == param_groups_num:
                exit()

    def save_models(self, models, epoch, acc, is_best=False):
        """Save model weights to disk
        """
        if is_best:
            for r, d, f in os.walk(os.path.join(self.log_path, "models")):
                for d_name in d:
                    if "best" in d_name:
                        shutil.rmtree(os.path.join(self.log_path, "models",
                                                   d_name))
            save_folder = os.path.join(self.log_path, "models",
                                       "best_weights_{}_{}"
                                       .format(epoch, acc))
            self.log_print("Save Model: New Best: {}".format(acc))
        else:
            save_folder = os.path.join(self.log_path, "models",
                                       "weights_{}_{}"
                                       .format(epoch, acc))
            self.log_print("Save Model: Acc: {}".format(acc))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)
        

    def load_models(self, networks, optimizer):
        """Load model(s) from disk
        """
        self.opts["t"].load_weights_folder = os.path.expanduser(
            self.opts["t"].load_weights_folder)

        assert os.path.isdir(self.opts["t"].load_weights_folder), \
            "Cannot find folder {}".format(self.opts["t"].load_weights_folder)
        self.log_print("loading model from folder {}"
              .format(self.opts["t"].load_weights_folder))
        filename = self.opts["t"].load_weights_folder.split('/')[-1]
        epoch = filename.split('_')[-2]
        acc = filename.split('_')[-1]

        for n, v in networks.items():
            self.log_print("Loading {} weights...".format(n))
            path = os.path.join(self.opts["t"].load_weights_folder,
                                "{}.pth".format(n))
            model_dict = networks[n].state_dict()
            try:
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict}
                model_dict.update(pretrained_dict)
                networks[n].load_state_dict(model_dict)
            except:
                self.log_print("{} is randomly initialized".format(n))

        return networks, int(epoch) + 1, float(acc)


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : Number of params: {}'.format(model._get_name(), para))
    self.log_print('Model {} : params: {:4f}M'
          .format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    # print('Model {} : Number of intermedite variables without backward: {}'
    #       .format(model._get_name(), total_nums))
    # print('Model {} : Number of intermedite variables with backward: {}'
    #       .format(model._get_name(), total_nums*2))
    self.log_print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    self.log_print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))
