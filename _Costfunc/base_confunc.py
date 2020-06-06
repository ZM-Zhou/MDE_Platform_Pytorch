import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


##############################################################################
# reprojection loss
##############################################################################
def compute_reprojection_map(pred, target, ssim=None):
    """Computes reprojection loss between a batch of predicted
       and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff

    if not ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = ssim(pred, target)
        reprojection_loss = l1_loss * 0.15 + ssim_loss* 0.85

    return reprojection_loss


##############################################################################
# smooth loss
##############################################################################
def compute_smooth_map(disp, img):
    """Computes the smoothness loss for a disparity image
       The color image is used for edge-aware smoothness
    """
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    padh = nn.ReplicationPad2d((0, 0, 1, 0))
    padw = nn.ReplicationPad2d((1, 0, 0, 0))
    disph = padh(disp)
    dispw = padw(disp)
    imgh = padh(img)
    imgw = padw(img)


    grad_disp_x = torch.abs(dispw[:, :, :, :-1] - dispw[:, :, :, 1:])
    grad_disp_y = torch.abs(disph[:, :, :-1, :] - disph[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(imgw[:, :, :, :-1] - imgw[:, :, :, 1:]),
                            1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(imgh[:, :, :-1, :] - imgh[:, :, 1:, :]),
                            1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x + grad_disp_y


#############################################################################
# Multiview geometry functions
#############################################################################
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height),
                               indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1,
                                 self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones],
                                       1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K
       and at position T
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        temp = (cam_points[:, 2, :].unsqueeze(1)
                                             + self.eps)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1)
                                             + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2,
                                     self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class Disp_point(nn.Module):
    """Layer to transform a disp image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.max_disp = width * 0.3

        meshgrid = np.meshgrid(range(self.width), range(self.height),
                               indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.from_numpy(self.id_coords)

        pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords_x = pix_coords[:, 0, ...].unsqueeze(1)
        self.pix_coords_y = pix_coords[:, 1, ...].unsqueeze(1)
        self.pix_coords_x = nn.Parameter(self.pix_coords_x, requires_grad=False)
        self.pix_coords_y = nn.Parameter(self.pix_coords_y, requires_grad=False)


    def forward(self, disp, T):
        disp = disp * self.max_disp
        disp = disp.view(self.batch_size, 1, -1)
        side = T[:, 0, 3].unsqueeze(1).unsqueeze(1)
        pix_coords_x_new = disp * side + self.pix_coords_x
        pix_coords_new = torch.cat([pix_coords_x_new, self.pix_coords_y], dim=1)
        pix_coords_new = pix_coords_new.view(self.batch_size, 2,
                                     self.height, self.width)
        pix_coords_new = pix_coords_new.permute(0, 2, 3, 1)
        pix_coords_new[..., 0] /= self.width - 1
        pix_coords_new[..., 1] /= self.height - 1
        pix_coords_new = (pix_coords_new - 0.5) * 2

        return pix_coords_new

##############################################################################
# SSIM module
##############################################################################
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) *\
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
