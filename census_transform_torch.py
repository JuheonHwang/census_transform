import torch
import torchvision
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
import math
import time

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def get_disparity(left_img, right_img, min_disp, max_disp, aggreate_size, census_size, sub_pixel, device, visualize):
    disp_L, disp_R = stereo_matching(left_img, right_img, min_disp, max_disp, aggreate_size, census_size, sub_pixel, device)
    disp_R2L, disp_L2R = disparity_interp(disp_L, disp_R)
    disp_L[torch.abs(disp_R2L - disp_L) > threshold] = 0
    disp_R[torch.abs(disp_L2R - disp_R) > threshold] = 0

    if visualize:
        disp_R_cpu = disp_R.cpu()
        save_image(disp_R_cpu / torch.max(disp_R_cpu), 'disp_R.png')
        disp_L_cpu = disp_L.cpu()
        save_image(disp_L_cpu / torch.max(disp_L_cpu), 'disp_R.png')

    return disp_L, disp_R

def disparity_interp(disp_L, disp_R):
    bz = disp_L.size(0)
    device = disp_L.device
    grid_x, grid_y = torch.meshgrid(torch.arange(disp_L.size(1)), torch.arange(disp_L.size(2)))

    R2L_y_grid = torch.clamp(grid_y - disp_L, min=0, max=disp_L.size(2)-1).type(torch.DoubleTensor).to(device)
    L2R_y_grid = torch.clamp(grid_y + disp_R, min=0, max=disp_L.size(2)-1).type(torch.DoubleTensor).to(device)

    R2L_y_grid1 = torch.floor(R2L_y_grid).long()
    R2L_y_grid2 = torch.clamp(torch.floor(R2L_y_grid+1), min=0, max=disp_L.size(2)-1).long()
    R2L_alpha = R2L_y_grid - R2L_y_grid1

    L2R_y_grid1 = torch.floor(L2R_y_grid).long()
    L2R_y_grid2 = torch.clamp(torch.floor(L2R_y_grid+1), min=0, max=disp_L.size(2)-1).long()
    L2R_alpha = L2R_y_grid - L2R_y_grid1

    disp_R2L = torch.reshape(torch.reshape(R2L_alpha, (bz, -1)) * disp_R[:, torch.reshape(grid_x, (bz, -1)), torch.reshape(R2L_y_grid2.long(), (bz, -1))] + torch.reshape(1-R2L_alpha, (bz, -1)) * disp_R[:, torch.reshape(grid_x, (bz, -1)), torch.reshape(R2L_y_grid1.long(), (bz, -1))], (bz, disp_L.size(1), disp_L.size(2)))
    disp_L2R = torch.reshape(torch.reshape(L2R_alpha, (bz, -1)) * disp_L[:, torch.reshape(grid_x, (bz, -1)), torch.reshape(L2R_y_grid2.long(), (bz, -1))] + torch.reshape(1-L2R_alpha, (bz, -1)) * disp_L[:, torch.reshape(grid_x, (bz, -1)), torch.reshape(L2R_y_grid1.long(), (bz, -1))], (bz, disp_L.size(1), disp_L.size(2)))

    return disp_R2L, disp_L2R

def stereo_matching(left_img, right_img, min_disp, max_disp, aggreate_size, census_size, sub_pixel, device):
    bz = left_img.size(0)

    census_R = census_transform(right_img, census_size, device)
    census_L = census_transform(left_img, census_size, device)

    cost_volume_R = torch.inf * torch.ones((bz, left_img.size(2), left_img.size(3), max_disp - min_disp + 1), dtype=torch.float32, device=device)
    aggreate_filter = torch.ones((1, 1, aggreate_size, aggreate_size), device=device)
    for disp in range(max_disp-min_disp+1):
        census_L_shift = torch.roll(census_L, shifts=-(min_disp+disp), dims=2)
        census_L_shift[:, :, -(min_disp+disp):, :] = 0
        cost_volume_R[:, :, :, disp] = F.conv2d(torch.sum(torch.bitwise_xor(census_R, census_L_shift), dim=3).unsqueeze(1).float(), aggreate_filter, padding='same')
    del census_L_shift
    torch.cuda.empty_cache()
    disp_R = torch.argmin(cost_volume_R, dim=3)

    if sub_pixel:
        disp_R = disp_R.float()
        for i in range(bz):
            i_disp_R = torch.reshape(disp_R[i, :, :], (-1,))
            cost = torch.reshape(cost_volume_R[i, :, :, :], (-1, max_disp - min_disp + 1))
            tar_idx = torch.bitwise_and(i_disp_R > 0, i_disp_R <= (max_disp - min_disp))
            cost = cost[tar_idx, :]

            dD = i_disp_R[tar_idx].int()

            yP = sub2ind(cost.shape, torch.arange(cost.size(0), device=device), dD + 1)
            yM = sub2ind(cost.shape, torch.arange(cost.size(0), device=device), dD - 1)
            yD = sub2ind(cost.shape, torch.arange(cost.size(0), device=device), dD)
            cost = torch.reshape(cost, (-1,))

            disp_R[i, torch.reshape(tar_idx, (left_img.size(2), left_img.size(3)))] = i_disp_R[tar_idx] + ((cost[yP] - cost[yM]) / (2*(2*cost[yD] - cost[yM] - cost[yP])))
        del i_disp_R
    del cost_volume_R
    torch.cuda.empty_cache()
    disp_R = disp_R + min_disp

    cost_volume_L = torch.inf * torch.ones((bz, left_img.size(2), left_img.size(3), max_disp - min_disp + 1), dtype=torch.float32, device=device)
    for disp in range(max_disp-min_disp+1):
        census_R_shift = torch.roll(census_R, shifts=(min_disp+disp), dims=2)
        census_R_shift[:, :, :(min_disp+disp), :] = 0
        cost_volume_L[:, :, :, disp] = F.conv2d(torch.sum(torch.bitwise_xor(census_L, census_R_shift), dim=3).unsqueeze(1).float(), aggreate_filter, padding='same')
    del census_R_shift
    del census_L, census_R
    torch.cuda.empty_cache()
    disp_L = torch.argmin(cost_volume_L, dim=3)

    if sub_pixel:
        disp_L = disp_L.float()
        for i in range(bz):
            i_disp_L = torch.reshape(disp_L[i, :, :], (-1,))
            cost = torch.reshape(cost_volume_L[i, :, :, :], (-1, max_disp - min_disp + 1))
            tar_idx = torch.bitwise_and(i_disp_L > 0, i_disp_L <= (max_disp - min_disp))
            cost = cost[tar_idx, :]

            dD = i_disp_L[tar_idx].int()

            yP = sub2ind(cost.shape, torch.arange(cost.size(0), device=device), dD + 1)
            yM = sub2ind(cost.shape, torch.arange(cost.size(0), device=device), dD - 1)
            yD = sub2ind(cost.shape, torch.arange(cost.size(0), device=device), dD)
            cost = torch.reshape(cost, (-1,))

            disp_L[i, torch.reshape(tar_idx, (left_img.size(2), left_img.size(3)))] = i_disp_L[tar_idx] + ((cost[yP] - cost[yM]) / (2*(2*cost[yD] - cost[yM] - cost[yP])))
        del cost
        del tar_idx
        del yP, yM, yD, dD
        del i_disp_L
    del cost_volume_L
    torch.cuda.empty_cache()
    disp_L = disp_L + min_disp

    return disp_L, disp_R

def census_transform(img, census_size, device):
    bz = img.size(0)
    img_margin = torch.zeros((bz, img.size(1), img.size(2)+census_size-1, img.size(3)+census_size-1), dtype=torch.float32, device=device)
    img_margin[:, :,
        math.floor((census_size-1)/2)+0:math.floor((census_size-1)/2)+img.size(2),
        math.floor((census_size-1)/2)+0:math.floor((census_size-1)/2)+img.size(3)] = img

    img_block = torch.nn.Unfold(kernel_size=(census_size, census_size))(img_margin)
    img_block = img_block > img_block[:, math.ceil(0.5 * (census_size ** 2)), :]

    img_census = torch.permute(torch.reshape(img_block, (bz, census_size ** 2, img.size(2), img.size(3))), (0, 2, 3, 1))
    del img_margin
    del img_block
    torch.cuda.empty_cache()

    return img_census

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    aggreate_size = 11
    census_size = 11
    min_disp = 0
    max_disp = 300
    threshold = 10
    sub_pixel = True
    visualize = True
    device = "cuda:3"

    left_img = read_image('./image/Position000000_CAM00.png')
    right_img = read_image('./image/Position000000_CAM01.png')

    start = time.time()
    disp_L, disp_R = get_disparity(left_img.unsqueeze(0), right_img.unsqueeze(0), min_disp, max_disp, aggreate_size, census_size, sub_pixel, device, visualize)
    print(time.time() - start)
    print()
