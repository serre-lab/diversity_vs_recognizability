import torch
import torch.nn.functional as f

def compute_theta_general(theta, invert=False, scale_ratio=None):
    scale = theta[:, :4].view(-1, 2, 2)
    translation = theta[:, 4:].view(-1, 2, 1)
    if scale_ratio is not None:
        scale[:,0, 0] = scale_ratio
        scale[:, -1, -1] = scale_ratio
    if invert:
        scale = torch.inverse(scale)
        translation = torch.matmul(-scale, translation).view(-1, 2, 1)
    theta_mat = torch.cat((scale, translation), dim=2)
    return theta_mat


def read_transformer(theta_mat, image, imagette_size, mode='bilinear'):
    grid = f.affine_grid(theta_mat, imagette_size, align_corners=False)
    imagette = f.grid_sample(image, grid, align_corners=False, mode=mode)
    return imagette


def write_transformer(theta_mat, imagette, image_size, mode='bilinear'):
    grid = f.affine_grid(theta_mat, image_size, align_corners=False)
    image = f.grid_sample(imagette, grid, align_corners=False, mode=mode)
    return image