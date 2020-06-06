# MDE_Platform_Pytorch 0.1
A constantly improving deep learning platform for monocular depth estimation with Pytorch.

## What the platform can do?
The main idea of this framwork is "Focus on the model". Although Pytorch is a very simple and flexible framework, it is still not convenient enough to write the corresponding training code every time for a new network in the experiment. We try to build a universal platform to finish all the work except building the model.

### This platform can be used to:
Train most end-to-end models for vision which has prescribed format at 1 GPU or CPU;<br>
With Json files, adjust hyperparameters and automatically train models sequentially;<br>
Evaluate the effectiveness of the network.

### Possible updates in subsequent releases:
Update more efficient and normative code;<br>
Add common metric code;<br>
Support cross-dataset training;<br>
Train with multi-optimizer, e.g. training GAN.<br>
...

## Setup
With Anaconda, you can install the dependencies with:
```
conda install pytorch torchvision -c pytorch
pip install future tensorboard==1.14
pip install opencv
```
We test our platform with Pytorch 1.4.0 , CUDA 10.1, Python 3.7.7 and Ubuntu 18.04. But we think Pytorch >= 1.1.0 is ok.<br>
If you want to try our example, the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php)(Raw, about 167G) or [Cityscapes](https://www.cityscapes-dataset.com/)(LeftImg, RightImg disparity and gtFine, about 91G) dataset is required. 

## Train & Test a network
Its very simple and flexible to train or test a network by this platform. We put two example in the platform -- MonodepthV2 in Stereo mode and a variant network for unsupervised monocular depth esitmation. The following command trains the MonodepthV2 in the defualt settings(at least 6G of GPU Memory):
```
python ./Train/trainer.py --json_path ./JsonList/monov2.json
```
If you want to train our variant, just exchange the Json file:
```
python ./Train/trainer.py --json_path ./JsonList/fuse_depth.json
```
To evaluate this two networks for monocualr depth estimation in KITTI, We follow [MonodepthV2](https://github.com/nianticlabs/monodepth2) prepare the ground truth, and put `.npz` into KITTI's path. The defulat pretrained weights path is in `../Train_log/[train_date]/models/`. Then the following command evaluate the networks in KITTI:
```
python ./Metric/depth_KITTI/py --json_path ./JsonList/fuse_depth.json --weights_path [pertrained weights dir's path]
```

## The architecture of the Platform
We divide the platform into two main part: which you can design and adjust and which we have finished.<br>
### You can design and adjust
We consider every end-to-end model as three module: Dataset, Network and Costfunctions. When you build your model, you should implement these three modules separately in `_Dataset`, `_Network` and `_Costfunc`. 
### We have finished
We finish the trainer and a powerful logger in `Train`. 
### Visualization and metric
We have finished some of the functions that you may need in these two parts. We suggest that you should check it out before use, because they are not very "simple". They may be further optimized and integrated into other parts in later versions.

## Design and Use your model
We set some rules for designing new models to facilitate its adaptation to this platform. A general rule is that ALL modules transfer data use Python DICTIONARIES. Each module should contain an OPTION class for adjusting the hyperparameters, such as:
```
class Network_Opts(Network_Opts_base):
    def __init__(self):
        super().__init__()
        self.max_depth = 100
        self.min_depth = 0.1
        self.backbone = 18
        self.sample_mode = "nearest"

        self.use_depth = True
```
Just like `Network_Opts_base`, we provide some basic modules and prescribed templates in `base_*.py`. We suggest to use them as the parent classes. For more information, please refer to our example.

### Dataset
1.Dataset should be inherited from the pytorch Map-style dataset.<br>
2.The output of dataset (return in `__getitem__()`) should be a dictionary named `inputs` which contains the input of the model and the data used to calculate the cost.<br>
3.An evaluation method named `evaluation()` should be implement in dataset. Use the following form as input, and add the corresponding metric results in `losses`:
```
def evaluation(self, inputs, outputs, losses=None, visual=None):
    # your metirc
    # most_important_result is used to determine whether the learning rate needs to be adjusted in lr_scheduler.ReduceLROnPlateau.
    most_important_result = losses["metric_result1"]

    return most_important_result, losses, visual
```

### Network
1.Network should be inherited from `Network_base` from `base_module.py`.<br>
2.The part that needs to be trained should be modularized and called through `self.networks["Part_name"] = Part()` in `__init__()` (Part is inherited from `torch.nn.Module`). The following line should be added after all parts are called to let them can be optimized by optimizer:
```
self.all_net = nn.ModuleList(v for k, v in self.networks.items())
```
And then, the following method should be defined so that optimizer can optimize different parts with different learning rates:
```
def get_trainable_params(self):  # examples
    names = ["Encoder", "Decoder"]  # name of parts
    muls = [1, 1]  # set the multiple of the learning rate, the order is the same as the names
    return self.get_module_params(names, muls)
```
3.The forward method should be defined like:
```
def forward(self, inputs, stage):
    # process
    return outputs
```
Where `inputs` is the from the dataset and `stage` holds the training status. `outputs` should alse be a dictionary which contains results and intermediate variables maybe used for cost calculation.

### Costfunction
1.Cost function should be be inherited from `Costfunc_base` from `base_costfunc.py` with a `compute_losses` methods:
```
compute_losses(self, inputs, outputs, stage):
    # compute losses
    return losses
```
Where `losses` is a dictionary and The final cost for back propagation should be placed `losses["loss"]`.

### Visualization(Optional)
If you need to visualize during the training, we provide `logger` for visualization related methods and `stage` is used to let the module know the current state of the training. out Base classes has took `logger` as parameter in the `__init__()` methods and saved it at `self.logger`. When you need to do visualization in Network or Costfunctions, you can use following lines:
```
class Your_Network(Network_base):
    def __init__(self, opts, *args, **kargs):
        # your implement

        # make a folder for visualization, if you only want to do visualization during training
        if self.logger.is_train:           
            self.logger.make_logdir("Disp") 
    
    def forward(self, inputs, stage):
        if stage.is_visual and self.logger.is_train:
            # make your visualzation data
            img = {"img1": img1,
                   "img2": img2,
                   "img3": img3,
                   "img4": img4}
            visual_mode = ["depth", "disp", "error_map", "img"]
            output_size = [["img1", "img2"], ["img3", ["img4", "img1", 0.6]]]
            self.logger.do_visualizion("Disp", img, visual_mode, output_size)
```
This is not a normative function now. Please read the relevant implementation before using it to prevent mistakes.

### Before train
Congratulations on entering the last steps before training. The ease of training will prove that the previous efforts are worthwhile. Firstly, Make sure to define your part in the `__init__.py` of the relevant folder. Then, add your modules and options in the dictionaries at `Utils/import_choice.py`. Finally, use <br>
`Tools/make_otps_json.py` make your JSON file and enjoy the training.

### Tips
1.The same options between different modules are allowed, they will be MERGED in the generated JSON file.<br>
2.We have provided some basic modules in `base_*.py`.<br>
3.We have added some scripts that may be useful in `Tools`, maybe you will need them.


## Acknowledgement
Thanks to [Monodepthv2](https://github.com/nianticlabs/monodepth2) for some inspiration, codes and examples. 

