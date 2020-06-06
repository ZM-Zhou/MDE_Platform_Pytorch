import json

import sys
import os
sys.path.append(os.getcwd())
from Train.train_opts import TrainOpts
from Utils.import_choice import import_opts

json_name = "monov2_CS"
dataset_id = "CSDataset"
network_id = "Monov2"
costfunc_id = "NormalMaskUns"
json_text = json_name

DatasetOpts, NetworkOpts, CostfuncOpts = import_opts(dataset_id,
                                                     network_id, costfunc_id)

opts = {}
opts["t"] = TrainOpts()
opts["d"] = DatasetOpts()
opts["n"] = NetworkOpts()
opts["c"] = CostfuncOpts()

json_dict = {}
json_dict["describle"] = json_text
json_dict["opts"] = {}
for k, v in opts.items():
    for key, value in vars(v).items():
        json_dict["opts"][key] = value
json_dict["dataset_id"] = dataset_id
json_dict["network_id"] = network_id
json_dict["costfunc_id"] = costfunc_id

json_data = json.dumps(json_dict, indent=4, separators=(',', ': '))
with open("JsonList/" + json_name + ".json", "w") as f:
    f.write(json_data)