
import json
import os
import sys 
sys.path.append(os.getcwd())

import _Dataset
import _Network
import _Costfunc
from Train.train_opts import TrainOpts
from Utils.import_choice import import_opts

json_dir = "./JosnList"

def update_data(json_path):
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
            try:
                opts[k].__dict__[key] = opts_dict[key]
            except:
                opts_dict[key] = opts[k].__dict__[key]

    use_dict = {}
    for k, v in opts.items():
        for key, value in vars(v).items():
            use_dict[key] = value
    for k in list(opts_dict.keys()):
        try:
            _ = use_dict[k]
        except:
            opts_dict.pop(k)
                

    print("Done!")

    json_data = json.dumps(json_data, indent=4, separators=(',', ': '))
    with open(json_path, "w") as f:
        f.write(json_data)


if __name__ == "__main__":
    # for root, dirs, files in os.walk(json_dir):
    # for file in files:
    #     if file.endswith('.json'):
    #         json_path = os.path.append(root, file)
    #         update_data(json_path)
    json_path ="/home/commander/ZZM_temple/DL_Framework_Pytorch/JsonList/K9.json"
    update_data(json_path)