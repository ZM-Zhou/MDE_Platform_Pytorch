import numpy as np
import json
import PIL.ImageFont as ImageFont
import PIL.Image as pil
from PIL import ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import cv2 as cv


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import time


import os
import sys
sys.path.append(os.getcwd())
from Utils.import_choice import JsonArg, Stage, json_to_data
from _Dataset.kitti import KittiColorDataset, Dataset_Options
from Metric.logger import *
from Utils.visualization import visualize_depth

class Metric():
    def __init__(self):
        json_arg = JsonArg()
        json_arg.parser.add_argument("--weights_path",
                                     type=str,
                                     required=True)

        j_data = json_arg.parse()
        opts, _, Network, Costfunc, self.describles\
            = json_to_data(j_data.json_path)
        self.dataset_opts = Dataset_Options()
        self.dataset_opts.height = opts["d"].height
        self.dataset_opts.width = opts["d"].width
        self.dataset_opts.T_x = opts["d"].T_x
        for k, v in opts.items():
            if getattr(v, 'batch_size', None):
                opts[k].batch_size = 1
        self.opts = opts

        self.stage = Stage()
        self.logger = MetricLog(j_data.weights_path, "KITTI")
        if torch.cuda.is_available() and not opts["t"].no_cuda:
            self.logger.device = torch.device("cuda")
        else:
            self.logger.device = torch.device("cpu")

        #get opt, model, loss
        net = Network(opts["n"], self.logger)
        self.net = net.to(self.logger.device)
        self.net.set_eval()
        #load model
        self.net.networks = self.logger.load_models(self.net.networks)
        self.loss_fun = Costfunc(opts["c"], self.logger)
    
    def test_all(self):
        #get all test data
        test_dataset = KittiColorDataset(self.dataset_opts, mode='test')
        test_loader = DataLoader(test_dataset, 1, shuffle=False,
                                num_workers=0, drop_last=False)
        
        self.logger.log_print('test {}  in {} traind with {}'
                              .format(self.describles[2],
                                      self.describles[1],
                                      self.describles[3],))
        self.logger.log_print("weights: {}\n{} test_image"
                              .format(self.logger.weights_path,
                                      len(test_dataset)))
        depth_metric_names = ["eval/abs_rel", "eval/sq_rel", "eval/rms",
                            "eval/log_rms", "eval/a1", "eval/a2", "eval/a3"]
        total = {}
        for name in depth_metric_names:
            total[name] = 0

        self.stage.is_visual = False
        board_list = []
        board_dict = {}
        start = time.time()
        for index, inputs in enumerate(test_loader):
            for key, ipt in inputs.items():
                if key != "img_info":
                    inputs[key] = ipt.to(self.logger.device)
            outputs= self.net(inputs, self.stage)

            _, losses, _ = test_dataset.evaluation(inputs, outputs)
            board_list.append((inputs["img_info"], losses))
            losses_data = {}
            for k, v in total.items():
                total[k] = v + losses[k]
                losses_data[k] = float(losses[k])
            res = {}
            res["path"] = inputs["img_info"][0]
            res["metric"] = losses_data
            board_dict[index] = res
            print(index, end="\r")

        use_time = time.time() - start
        self.logger.do_log_all_test(use_time, total, len(test_dataset))

        json_data = json.dumps(board_dict, indent=4, separators=(',', ': '))
        with open(os.path.join(self.logger.log_path, "metric_list.json"), "w") as f:
            f.write(json_data)
        board_list.sort(key=lambda x:x[1]["eval/abs_rel"])

        with open(os.path.join(self.opts["d"].data_dir, "split_list",
                               "sample_files.txt"), "w") as f:
            w_lines = []
            #best
            for i in range(5):            
                w_lines.append(board_list[i][0][0] + '\n')
            #mid
            for i in range(int(len(test_dataset) / 2 - 2),
                        int(len(test_dataset) / 2 + 3)):
                w_lines.append(board_list[i][0][0] + '\n')
            #worst
            for i in range(len(test_dataset) - 5,len(test_dataset)):
                w_lines.append(board_list[i][0][0] + '\n')

            f.writelines(w_lines)

    def test_sample(self):
        sam_dataset = KittiColorDataset(self.dataset_opts, mode='sample')
        sam_loader = DataLoader(sam_dataset, 1, shuffle=False,
                                num_workers=0, drop_last=False)
        self.logger.make_logdir("Auto_Sample")
        pics = []
        title = ["best", "mid", "worst"]
        self.stage.is_visual = True
        for index, inputs in enumerate(sam_loader):
            for key, ipt in inputs.items():
                if key != "img_info":
                    inputs[key] = ipt.to(self.logger.device)
            outputs= self.net(inputs, self.stage)
            losses, visual_dict = self.loss_fun.compute_losses(inputs, 
                                                               outputs,
                                                               self.stage)
            _, losses, visual_dict = sam_dataset.evaluation(inputs, outputs,
                                                            losses,
                                                            visual_dict)

            depth_metric_names = ["eval/abs_rel", "eval/sq_rel", "eval/rms",\
                                "eval/log_rms", "eval/a1",
                                "eval/a2", "eval/a3"]
            self.logger.log_print("{}-{}".format(title[int(index / 5)],
                                                index % 5))

            for name in depth_metric_names:
                self.logger.log_print("->{} = {:.4f}".format(name,
                                                             losses[name]))

            name = "{}-{}".format(title[int(index / 5)], index % 5)
            v_modes = ["img", "img", "disp", "error_pn", "depth"]
            v_size = [["left_img", "disp"],
                      ["est_left_img", ["left_img", "est_left_img", 1]],
                      ["depth", ["left_img", "depth", 0.8]],
                      ["photo_error", ["left_img", "photo_error", 0.5]]]
            self.logger.do_visualizion("Auto_Sample", visual_dict, v_modes, v_size, name)

    def test_choice(self, compare = False):
        if compare:
            choice_dataset = KittiColorDataset(self.dataset_opts, mode='compare')
            self.logger.make_logdir("Compare")
        else:
            choice_dataset = KittiColorDataset(self.dataset_opts, mode='choice')
            self.logger.make_logdir("Choice")
        choice_loader = DataLoader(choice_dataset, 1, shuffle=False,
                                   num_workers=0, drop_last=False)
        self.stage.is_visual = True
        for index, inputs in enumerate(choice_loader):
            for key, ipt in inputs.items():
                if key != "img_info":
                    inputs[key] = ipt.to(self.logger.device)
            outputs= self.net(inputs, self.stage)
            losses, visual_dict = self.loss_fun.compute_losses(inputs,
                                                               outputs,
                                                               self.stage)
            _, losses, visual_dict = choice_dataset.evaluation(inputs,
                                                               outputs,
                                                               losses,
                                                               visual_dict)

            depth_metric_names = ["eval/abs_rel", "eval/sq_rel", "eval/rms",\
                                "eval/log_rms", "eval/a1",
                                "eval/a2", "eval/a3"]
            if not compare:
                self.logger.log_print("{}".format(index))
                for name in depth_metric_names:
                    self.logger.log_print("->{} = {:.4f}".format(name,
                                                                 losses[name]))

            name = "{}".format(index)
            v_modes = ["img", "img", "disp", "error_heat", "error_pn"]
            v_size = [["left_img", "depth"],
                      ["est_left_img", ["left_img", "est_left_img", 1]],
                      ["disp", ["left_img", "disp", 0.8]],
                      ["photo_error", ["left_img", "photo_error", 0.5]]]
            if compare:
                self.logger.do_visualizion("Compare", visual_dict, v_modes, v_size, name)
            else:
                self.logger.do_visualizion("Choice", visual_dict, v_modes, v_size, name)
            print(index, end="\r")
    
if __name__ == "__main__":
    metric = Metric()
    metric.test_all()
    metric.test_sample()
    metric.test_choice()
    # metric.test_choice(True)
    

