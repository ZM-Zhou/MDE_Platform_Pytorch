import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import cv2 as cv
from collections import namedtuple

import torch
import torch.nn.functional as F

# Cityscapes segment labels
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

trainid2color = {label.trainId: label.color for label in labels}


def make_output_img(imgs, size):
    '''
    Combine all images into the specified size in order.
    '''
    output_img = []
    for columns in size:
        output_row = []
        for key in columns:
            if isinstance(key, list):
                base_img = imgs[key[0]]
                add_img = imgs[key[1]]
                img = base_img + key[2] * add_img
                max_value = np.max(img)
                img = img / max_value * 255
                output_row.append(img)
            else:
                output_row.append(imgs[key])
        output_row = np.hstack(output_row)
        output_img.append(output_row)
    output_img = np.vstack(output_img).astype(np.uint8)
    output_img = cv.cvtColor(output_img, cv.COLOR_RGB2BGR)

    return output_img
    
class VisualImage():
    def __init__(self):
        self.visual_mode_dict ={"img": self.visual_rgb,
                                "depth": self.visual_depth,
                                "disp": self.visual_disp,
                                "error_heat": self.visual_heatjet,
                                "error_pn": self.visual_pn,
                                "mask": self.visual_mask,
                                "segment": self.visual_segment}
    
    def do_visualize(self, img ,mode):
        output = self.visual_mode_dict[mode](img)
        return output

    def visual_rgb(self, img):
        return img * 255
    
    def visual_depth(self, depth):
        vmax = np.percentile(depth, 95)
        normal_depth = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
        mapper_depth = cm.ScalarMappable(norm=normal_depth, cmap='magma')
        depth_color = (mapper_depth.to_rgba(depth[..., 0])[:, :, :3] * 255)
        return depth_color
    
    def visual_disp(self, disp):
        vmin = np.percentile(disp, 5)
        normal_disp = mpl.colors.Normalize(vmin=vmin, vmax=disp.max())
        mapper_disp = cm.ScalarMappable(norm=normal_disp, cmap='magma')
        disp_color = (mapper_disp.to_rgba(disp[..., 0])[:, :, :3] * 255)
        return disp_color
    
    def visual_heatjet(self, error):
        photo_error = error * 255
        photo_error = cv.applyColorMap(photo_error.astype(np.uint8), cv.COLORMAP_JET)
        photo_error = cv.cvtColor(photo_error, cv.COLOR_BGR2RGB)
        return photo_error
    
    def visual_pn(self, error):
        min_value = np.abs(np.min(error))
        max_value = np.abs(np.max(error))
        if max_value > min_value:
            normal_value = max_value
        else:
            normal_value = min_value
        normal_error = mpl.colors.Normalize(vmin=-normal_value,
                                            vmax=normal_value)
        mapper_error = cm.ScalarMappable(norm=normal_error, cmap='coolwarm')
        error = (mapper_error.to_rgba(error[..., 0])[:, :, :3] * 255)
        return error
    
    def visual_mask(self, mask):
        show_mask = mask * 255
        show_mask = np.tile(show_mask, (1, 1, 3))
        return show_mask
    
    def visual_segment(self, seglabel):
        h, w, _ = seglabel.size()
        seg_show = np.zeros((h, w, 3))
        for k, v in trainid2color.items():
            seg_show = np.where(seglabel == k, v,
                                seglabel)
        return seg_show


def visualize_depth(visual_dict, disp_or_error="disp", dataset="KITTI"):
    """visual_dict:"left_img": raw left image
                   "depth_error" or "disp"
                   "est_left_img": image reprojected from right
                   "depth": output depth
                   "photo_error": photometric error            
            all tensor should be normalized to [0, 1]befor input with
        shape [C, H, W] with .detach()
       disp_or_error: output "disp"arity when used in training or "error" 
       dataset: from "KITTI" "CS"
    """
    for k, v, in visual_dict.items():
        v = v.unsqueeze(0)
        if dataset == "KITTI":
            v = F.interpolate(v, [375, 1242], mode="bilinear",
                            align_corners=False)
        elif dataset == "CS":
            v = F.interpolate(v, [384, 1000], mode="bilinear",
                            align_corners=False)
        v = v.cpu().squeeze(0).permute(1, 2, 0).numpy()
        visual_dict[k] = v
    
    left_img = visual_dict["left_img"] * 255
    est_left_img = visual_dict["est_left_img"] * 255

    if disp_or_error == "error":
        error = visual_dict["depth_error"][..., 0]
        normal_error = mpl.colors.Normalize(vmin=0,
                                            vmax=1)
        mapper_error = cm.ScalarMappable(norm=normal_error, cmap='coolwarm')
        error = (mapper_error.to_rgba(error)[:, :, :3] * 255)
    else:
        error = visual_dict["disp"] * 255
        error = cv.applyColorMap(error.astype(np.uint8),
                                    cv.COLORMAP_OCEAN)
    
    depth = visual_dict["depth"][..., 0]
    disp = 1 / depth
    vmin = np.percentile(disp, 5)
    normal_disp = mpl.colors.Normalize(vmin=vmin, vmax=disp.max())
    mapper_disp = cm.ScalarMappable(norm=normal_disp, cmap='magma')
    depth_color = (mapper_disp.to_rgba(disp)[:, :, :3] * 255)

    photo_error = visual_dict["photo_error"] * 255
    photo_error = cv.applyColorMap(photo_error.astype(np.uint8), cv.COLORMAP_JET)
    photo_error = cv.cvtColor(photo_error, cv.COLOR_RGB2BGR)
    
    
    fused_img = (left_img + est_left_img)/2

    photoerror_img = left_img + 0.5 * photo_error
    photoerror_img = photoerror_img / np.max(photoerror_img)
    photoerror_img *= 255
    depth_img = left_img + 0.8 * depth_color
    depth_img = depth_img / np.max(depth_img)
    depth_img *= 255

    img1 = np.vstack([left_img, est_left_img, depth_color, photo_error])
    img2 = np.vstack([error, fused_img, depth_img, photoerror_img])
    all_img = np.hstack([img1, img2]).astype(np.uint8)
    all_img = cv.cvtColor(all_img, cv.COLOR_RGB2BGR)
    return all_img

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

trainid2color = {label.trainId: label.color for label in labels}

def visualize_segment(visual_dict, dataset="CS"):
    """visual_dict:"left_img": raw left image
                   "est_segment": predicted segment
                   "gt_segment": segment ground truth
                   "error": error map             
            all tensor should be normalized to [0, 1]befor input with
        shape [C, H, W] with .detach(), besides "segment"
       dataset: "CS" only
    """
    for k, v, in visual_dict.items():
        v = v.unsqueeze(0).to(torch.float)
        if "segment" in k:
            v = F.interpolate(v, [384, 1000], mode="nearest")
        else:
            v = F.interpolate(v, [384, 1000], mode="bilinear",
                              align_corners=False)
        v = v.cpu().squeeze(0).permute(1, 2, 0).numpy()
        visual_dict[k] = v

    left_img = visual_dict["left_img"] * 255
    error = visual_dict["error"] * 255
    error = cv.applyColorMap(error.astype(np.uint8), cv.COLORMAP_JET)
    error = cv.cvtColor(error, cv.COLOR_RGB2BGR)
    
    pred_seg_show = np.zeros_like(left_img)
    pred_seg = visual_dict["est_segment"].astype(np.int)
    gt_seg_show = np.zeros_like(left_img)
    gt_seg = visual_dict["gt_segment"].astype(np.int)
    for k, v in trainid2color.items():
        pred_seg_show = np.where(pred_seg == k, v,
                                 pred_seg_show)
        gt_seg_show = np.where(gt_seg == k, v,
                               gt_seg_show)

    pred_img = left_img + pred_seg_show
    pred_img = pred_img / np.max(pred_img)
    pred_img *= 255
    error_img = left_img + error
    error_img = error_img / np.max(error_img)
    error_img *= 255

    img1 = np.vstack([left_img, pred_seg_show, error])
    img2 = np.vstack([gt_seg_show, pred_img, error_img])
    all_img = np.hstack([img1, img2]).astype(np.uint8)
    all_img = cv.cvtColor(all_img, cv.COLOR_RGB2BGR)
    return all_img