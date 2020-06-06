import json
import numpy as np
import random
import argparse

import torch

import _Dataset
import _Network
import _Costfunc
from Train.train_opts import TrainOpts


##############################################################################
# module dictionary
##############################################################################
dataset_module_dict = {"KittiDataset": _Dataset.KittiDataset,
                       "CSDataset": _Dataset.CSDataset}

network_module_dict = {"Monov2": _Network.Monov2,                 
                       "ResFuse": _Network.ResFuse}

costfunc_module_dict = {"NormalMaskUns": _Costfunc.NormalMaskUns,
                        }

##############################################################################
# opts dictionary
##############################################################################
dataset_opt_dict = {"KittiDataset": _Dataset.KittiDatasetOpts,
                    "CSDataset": _Dataset.CSDatasetOpts}

network_opt_dict = {"Monov2": _Network.Monov2Opts,
                    "ResFuse": _Network.ResFuseOpts}

costfunc_opt_dict = {"NormalMaskUns": _Costfunc.NormalMaskUnsOpts,
                     }


class JsonArg:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--json_path",
                                 type=str,
                                 required=True)

    def parse(self):
        return self.parser.parse_args()


class Stage():
    def __init__(self):
        self.phase = "train"
        self.is_visual = False
    
    def set_train_phase(self, network_module):
        self.phase = "train"
        network_module.set_train()
    
    def set_val_phase(self, network_module):
        self.phase = "val"
        network_module.set_eval()
    
    def set_test_phase(self, network_module):
        self.phase = "test"
        network_module.set_eval()



def import_module(dataset_id, network_id, costfunc_id):
    Dataset = dataset_module_dict[dataset_id]
    Network = network_module_dict[network_id]
    Costfunc = costfunc_module_dict[costfunc_id]
    return Dataset, Network, Costfunc


def import_opts(dataset_id, network_id, costfunc_id):
    DatasetOpts = dataset_opt_dict[dataset_id]
    NetworkOpts = network_opt_dict[network_id]
    CostfuncOpts = costfunc_opt_dict[costfunc_id]
    return DatasetOpts, NetworkOpts, CostfuncOpts


def json_to_data(json_path):
    print("read json file...", end="")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    opts_dict = json_data["opts"]
    dataset_id = json_data["dataset_id"]
    network_id = json_data["network_id"]
    costfunc_id = json_data["costfunc_id"]
    describle = json_data["describle"]
    print("Done!")

    print("get Options...", end="")
    DatasetOpts, NetworkOpts, CostfuncOpts = import_opts(dataset_id,
                                                         network_id,
                                                         costfunc_id)
    opts = {}
    opts["t"] = TrainOpts()
    opts["d"] = DatasetOpts()
    opts["n"] = NetworkOpts()
    opts["c"] = CostfuncOpts()
    for k, v in opts.items():
        for key in v.__dict__:
            opts[k].__dict__[key] = opts_dict[key]
    print("Done!")

    print("get Modules...", end="")
    Dataset, Network, Costfunc = import_module(dataset_id,
                                               network_id, costfunc_id)
    print("Done!")

    return opts, Dataset, Network, Costfunc, [describle, dataset_id,
                                              network_id, costfunc_id]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
