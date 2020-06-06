import numpy as np
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import sys
sys.path.append(os.getcwd())
from Train.logger import *
from Utils.import_choice import JsonArg, Stage, json_to_data, setup_seed


class Trainer:
    def __init__(self):
        json_arg = JsonArg()
        json_path = json_arg.parse().json_path
        self.opts, Dataset, Network, Costfunc, describles\
            = json_to_data(json_path)
        self.eval_best = 1e10

        setup_seed(self.opts["t"].rand_seed)

        self.stage = Stage()
        self.logger = TrainLog(self.opts)
        self.logger.device = torch.device("cpu" if self.opts["t"].no_cuda
                                   else "cuda")

        # load network
        self.network = Network(self.opts["n"], self.logger)
        self.network.check_info()
        for k, layers in self.network.networks.items():
            layers.to(self.logger.device)
            self.network.networks[k] = layers

        # load loss-functions
        self.loss_func = Costfunc(self.opts["c"], self.logger)

        # load dataset
        train_dataset = Dataset(self.opts["d"], mode="train")
        self.train_loader = DataLoader(
            train_dataset, self.opts["t"].batch_size,
            shuffle=self.opts["t"].is_shuffle,
            num_workers=self.opts["t"].num_workers, pin_memory=True,
            drop_last=True)

        self.valid_dataset = Dataset(self.opts["d"], mode="val")
        self.valid_loader = DataLoader(
            self.valid_dataset, self.opts["t"].batch_size,
            shuffle=True,
            num_workers=self.opts["t"].num_workers, pin_memory=True,
            drop_last=True)
        self.valid_iter = iter(self.valid_loader)

        # load optimizer
        self.params_sets = self.network.get_trainable_params()
        self.model_optimizer = []
        self.model_lr_scheduler = []
        for sets in self.params_sets:
            trainable_params = sets[0]
            if self.opts["t"].optim == "Adam":
                self.model_optimizer.append(optim.Adam(trainable_params,
                                                self.opts["t"].learning_rate))
            elif self.opts["t"].optim == "SGD":
                self.model_optimizera.append(optim.SGD(trainable_params,
                                                self.opts["t"].learning_rate,
                                                momentum=0.9,
                                                weight_decay=0.0005))

            if self.opts["t"].scheduler == "Step":
                self.model_lr_scheduler.append(optim.lr_scheduler.MultiStepLR(
                    self.model_optimizer[-1], self.opts["t"].scheduler_step_size,
                    self.opts["t"].scheduler_rate))
            elif self.opts["t"].scheduler == "Plateau":
                self.model_lr_scheduler.append(optim.lr_scheduler.ReduceLROnPlateau(
                    self.model_optimizer[-1], factor=self.opts["t"].scheduler_rate,
                    patience=self.opts["t"].scheduler_step_size, min_lr=1e-6,
                    verbose=True))

        # load pretrain model
        self.epoch = 0
        if self.opts["t"].load_weights_folder is not None:
            self.network.networks, self.epoch, self.eval_best\
                = self.logger.load_models(self.network.get_networks(),
                                         self.model_optimizer)
            for i in range(2):
                self.model_lr_scheduler[i].last_epoch = self.epoch - 1

        # compute steps
        num_train_samples = len(train_dataset)
        self.logger.step = self.epoch * num_train_samples\
            // self.opts["t"].batch_size
        self.start_step = self.logger.step
        self.num_total_steps = num_train_samples\
            // self.opts["t"].batch_size * (self.opts["t"].num_epochs
                                            - self.epoch)
        self.visual_stop_step = self.opts["t"].visual_frequency\
            * self.opts["t"].visual_stop + self.start_step

        self.logger.do_log_before_train(train_dataset,
                                        self.valid_dataset, describles)

    def do_train(self):
        """Run the entire training pipeline
        """
        self.start_time = time.time()

        while self.epoch < self.opts["t"].num_epochs:
            self.process_epoch()    
            self.epoch = self.epoch + 1

    def process_epoch(self):
        for i in range(2):
            self.logger.do_log_epoch(self.model_optimizer[i],
                                        self.params_sets[i][1])
        self.network.set_train()
        self.stage.phase = "train"
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            if self.logger.step % 5 == 0:
                train_part = ["loss", "D_loss"]
            else:
                train_part = ["loss"]
            for i, name in enumerate(train_part):
                self.model_optimizer[i].zero_grad()
                if i == 0 and len(train_part) > 1:
                    losses["{}".format(name)].backward(retain_graph=True)
                else:
                    losses["{}".format(name)].backward()
                # check the gard
                temp_params = self.model_optimizer[i].param_groups
                self.logger.do_grad_check(temp_params, self.params_sets[i][1])

                self.model_optimizer[i].step()
                self.model_optimizer[i].zero_grad()

            duration = time.time() - before_op_time
            log_flag = batch_idx % self.opts["t"].log_frequency == 0\
                and batch_idx != 0
            if log_flag:
                self.logger.do_log(batch_idx, duration, losses,
                                     self.model_optimizer[0].state_dict()
                                     ['param_groups'][0]["lr"],
                                     self.start_time, self.epoch,
                                     (self.num_total_steps /
                                      (self.logger.step-self.start_step) - 1.0),
                                      is_gan=True)
                self.do_valid()

            self.logger.step += 1
        
        eval_all = 0
        for i in range(int(500 / self.opts["t"].batch_size)):
            eval_all += self.do_valid(do_log=False)
        eval_all /= int(500 / self.opts["t"].batch_size)
        if eval_all < self.eval_best:
            self.eval_best = eval_all
            self.logger.save_models(self.network.get_networks(),
                                        self.epoch, self.eval_best,
                                        True)
        if (self.epoch + 1) % self.opts["t"].save_frequency == 0:
            self.logger.save_models(self.network.get_networks(),
                                        self.epoch, eval_all)
        else:
            self.logger.do_log_validphase(eval_all)

        if self.opts["t"].scheduler == "Step":
            for i in range(2):
                self.model_lr_scheduler[i].step()
        elif self.opts["t"].scheduler == "Plateau":
            for i in range(2):
                self.model_lr_scheduler[i].step(eval_all)

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.logger.device, non_blocking=True)
        
        visual_flag = self.logger.step % self.opts["t"].visual_frequency == 0\
            and self.logger.step < self.visual_stop_step\
            and self.network.train_phase
        if visual_flag:
            self.stage.is_visual = True
        else:
            self.stage.is_visual = False
            
        outputs = self.network(inputs, self.stage)
        losses = self.loss_func.compute_losses(inputs, outputs, self.stage)

        return outputs, losses

    def do_valid(self, do_log=True):
        self.network.set_eval()
        self.stage.phase = "val"
        try:
            inputs = self.valid_iter.next()
        except StopIteration:
            self.valid_iter = iter(self.valid_loader)
            inputs = self.valid_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            eval_value, losses, _ = self.valid_dataset.evaluation(inputs,
                                                                  outputs,
                                                                  losses)
            if do_log:
                self.logger.do_log_valid(losses)
            del inputs, outputs, losses

        self.network.set_train()
        self.stage.phase = "train"

        return eval_value


if __name__ == '__main__':
    train = Trainer()
    train.do_train()
