import os
import numpy as np
from numpy import cross, dot
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.preprocessing import normalize
# from sklearn.metrics import mean_squared_error
import pickle
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import ndimage

import av
from transformers import AutoImageProcessor, VideoMAEImageProcessor, VideoMAEForVideoClassification


TILEMAP_SHAPE = (9, 16)  # 如果函数中传入的tilemap_shape参数为None, 则使用此默认值;
SALMAP_SHAPE = (64, 128)


def resize_image(img, new_size):
    # 计算缩放比例
    scale = np.array(new_size) / np.array(img.shape)

    # 使用双线性插值法缩小图像
    resized_img = ndimage.interpolation.zoom(img, (scale[0], scale[1]), order=1)

    return resized_img


# load the saliency maps for a "video" normalized between -1 and 1
# RUN_IN_SERVER is a flag used to load the file in a different manner if is stored in the server.
# ToDo check that the saliency and the traces are sampled at the same rate, for now we assume the saliency is sampled manually when running the scripts to create the scaled_images before extracting the saliency in '/home/twipsy/PycharmProjects/UniformHeadMotionDataset/AVTrack360/dataset/videos/cut_wogrey/creation_of_scaled_images'
def load_saliency(saliency_folder, video):
    mmscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    saliency_list = []
    with open(('%s/%s/%s' % (saliency_folder, video, video)), 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        for frame_id in range(1, len(p.keys())+1):
            salmap = p['%03d' % frame_id]
            salmap_norm = mmscaler.fit_transform(salmap.ravel().reshape(-1, 1)).reshape(salmap.shape)
            salmap_ds = resize_image(salmap_norm, SALMAP_SHAPE)
            saliency_list.append(salmap_ds)
    return np.array(saliency_list)

def load_true_saliency(saliency_folder, video):
    saliencies_for_video_file = os.path.join(saliency_folder, video + '.npy')
    saliencies_for_video = np.load(saliencies_for_video_file)
    return saliencies_for_video

# def load_fov(fov_folder, vid, uid):
#     save_path = os.path.join(fov_folder, vid, f'{uid}.npy')
#     fov = np.load(save_path)
#     return fov

def load_fov(fov_folder, vid):
    save_path = os.path.join(fov_folder, f'{vid}.npy')
    fov = np.load(save_path)
    return fov

def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2*np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)



# The (input) corresponds to (x, y, z) of a unit sphere centered at the origin (0, 0, 0)
# Returns the values (theta, phi) with:
# theta in the range 0, to 2*pi, theta can be negative, e.g. cartesian_to_eulerian(0, -1, 0) = (-pi/2, pi/2) (is equal to (3*pi/2, pi/2))
# phi in the range 0 to pi (0 being the north pole, pi being the south pole)
def cartesian_to_eulerian(x, y, z, lib='numpy'):
    if lib == 'numpy':
        r = np.sqrt(x*x+y*y+z*z)
        theta = np.arctan2(y, x)
        phi = np.arccos(z/r)
        # remainder is used to transform it in the positive range (0, 2*pi)
        theta = np.remainder(theta, 2*np.pi)
        return theta, phi
    elif lib == 'torch':
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or not isinstance(z, torch.Tensor):
            x, y, z = torch.tensor(x), torch.tensor(y), torch.tensor(z)
        r = torch.sqrt(x*x+y*y+z*z)
        theta = torch.atan2(y, x)
        phi = torch.acos(z/r)
        # remainder is used to transform it in the positive range (0, 2*pi)
        theta = torch.remainder(theta, 2*np.pi)
        return theta, phi
    else:
        raise NotImplementedError()

# The (input) values of theta and phi are assumed to be as follows:
# theta = Any              phi =   0    : north pole (0, 0, 1)
# theta = Any              phi =  pi    : south pole (0, 0, -1)
# theta = 0, 2*pi          phi = pi/2   : equator facing (1, 0, 0)
# theta = pi/2             phi = pi/2   : equator facing (0, 1, 0)
# theta = pi               phi = pi/2   : equator facing (-1, 0, 0)
# theta = -pi/2, 3*pi/2    phi = pi/2   : equator facing (0, -1, 0)
# In other words
# The longitude ranges from 0, to 2*pi
# The latitude ranges from 0 to pi, origin of equirectangular in the top-left corner
# Returns the values (x, y, z) of a unit sphere with center in (0, 0, 0)
def eulerian_to_cartesian(theta, phi, lib='numpy'):
    if lib == 'numpy':
        x = np.cos(theta)*np.sin(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(phi)
        return np.array([x, y, z])
    elif lib == 'torch':
        x = torch.cos(theta)*torch.sin(phi)
        y = torch.sin(theta)*torch.sin(phi)
        z = torch.cos(phi)
        return torch.stack([x, y, z])


def quaternion_to_cartesian(qx, qy, qz, qw):
    x = 2 * qx * qz + 2 * qy * qw
    y = 2 * qy * qz - 2 * qx * qw
    z = 1 - 2 * qx**2 - 2 * qy**2

    # 将数据集vr-dataset(Wu)所使用的坐标系转换为我们所使用的坐标系:
    x_ = x
    y_ = -z
    z_ = y

    return x_, y_, z_


## Use this function to debug the behavior of the function eulerian_to_cartesian()
def test_eulerian_to_cartesian():
    # trace starting from (1, 0, 0) and moving to point (0, 0, 1)
    yaw_1 = np.linspace(0, np.pi/2, 50, endpoint=True)
    pitch_1 = np.linspace(np.pi/2, 0, 50, endpoint=True)
    positions_1 = np.array([eulerian_to_cartesian(yaw_samp, pitch_samp) for yaw_samp, pitch_samp in zip(yaw_1, pitch_1)])
    yaw_2 = np.linspace(3*np.pi/2, np.pi, 50, endpoint=True)
    pitch_2 = np.linspace(np.pi, np.pi/2, 50, endpoint=True)
    positions_2 = np.array([eulerian_to_cartesian(yaw_samp, pitch_samp) for yaw_samp, pitch_samp in zip(yaw_2, pitch_2)])
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.plot(positions_1[:, 0], positions_1[:, 1], positions_1[:, 2], color='b')
    ax.plot(positions_2[:, 0], positions_2[:, 1], positions_2[:, 2], color='g')
    ax.scatter(positions_1[0, 0], positions_1[0, 1], positions_1[0, 2], color='r')
    ax.scatter(positions_2[0, 0], positions_2[0, 1], positions_2[0, 2], color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax2 = fig.add_subplot(122)
    ax2.plot(yaw_1, pitch_1, color='b')
    ax2.plot(yaw_2, pitch_2, color='g')
    ax2.scatter(yaw_1[0], pitch_1[0], color='r')
    ax2.scatter(yaw_2[0], pitch_2[0], color='r')
    # to turn around the y axis, starting from 0 on the top
    ax2.set_ylim(ax2.get_ylim()[1], ax2.get_ylim()[0])
    ax2.set_xlabel('yaw')
    ax2.set_ylabel('pitch')
    plt.show()

# Transforms the eulerian angles from range (0, 2*pi) and (0, pi) to (-pi, pi) and (-pi/2, pi/2)
def eulerian_in_range(theta, phi):
    theta = theta - np.pi
    phi = (phi - (np.pi / 2.0))
    return theta, phi


# TODO: 下面大括号中注释掉的3个函数好像有问题, 之后如果用不到的话就删掉它们.
{
# # Returns an array of size (numOfTilesHeight, numOfTilesWidth) with values between 0 and 1 specifying the probability that a tile is watched by the user
# # We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
# # pos: (yaw, pitch) 该函数中的yaw和pitch, 是将ERP帧的中点位置作为0点, 而非左上角; 将左上角作为0点的是theta和phi;
# def from_position_to_tile_probability(pos, numTilesWidth, numTilesHeight, mode='soft'):
#     yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, numTilesWidth, endpoint=False), np.linspace(0, 1, numTilesHeight, endpoint=False))
#     yaw_grid += 1.0 / (2.0 * numTilesWidth)
#     pitch_grid += 1.0 / (2.0 * numTilesHeight)
#     yaw_grid = (yaw_grid - 0.5) * 2 * np.pi
#     pitch_grid = -(pitch_grid - 0.5) * np.pi
#     cp_yaw = pos[0]
#     cp_pitch = pos[1]
#     delta_long = np.abs(np.arctan2(np.sin(yaw_grid - cp_yaw), np.cos(yaw_grid - cp_yaw)))
#     numerator = np.sqrt(np.power(np.cos(cp_pitch) * np.sin(delta_long), 2.0) + np.power(np.cos(pitch_grid) * np.sin(cp_pitch) - np.sin(pitch_grid) * np.cos(cp_pitch) * np.cos(delta_long), 2.0))
#     denominator = np.sin(pitch_grid) * np.sin(cp_pitch) + np.cos(pitch_grid) * np.cos(cp_pitch) * np.cos(delta_long)
#     second_ort = np.abs(np.arctan2(numerator, denominator))
#     gaussian_orth = np.exp((-1.0/(2.0*np.square(0.1))) * np.square(second_ort))
#     if mode == 'soft':
#         return gaussian_orth
#     elif mode == 'hard':
#         max_pos = np.where(gaussian_orth==np.max(gaussian_orth), 1, 0)
#         return max_pos
#     else:
#         raise NotImplementedError('mode should be "soft" or "hard".')
# def cartesian_to_heatmap(x, y, z, width, height):
#     theta, phi = cartesian_to_eulerian(x, y, z)
#     return eulerian_to_heatmap(theta, phi, width, height)
# def eulerian_to_heatmap(theta, phi, width, height):
#     theta_grid, phi_grid = np.meshgrid(np.linspace(0, 1, width, endpoint=False), 
#                                        np.linspace(0, 1, height, endpoint=False))
#     theta_grid += 1.0 / (2.0 * width)
#     phi_grid += 1.0 / (2.0 * height)
#     theta_grid = theta_grid * 2 * np.pi
#     phi_grid = phi_grid * np.pi

#     delta_long = np.abs(np.arctan2(np.sin(theta_grid - theta), np.cos(theta_grid - theta)))
#     numerator = np.sqrt(np.power(np.cos(phi) * np.sin(delta_long), 2.0) + np.power(np.cos(phi_grid) * np.sin(phi) - np.sin(phi_grid) * np.cos(phi) * np.cos(delta_long), 2.0))
#     denominator = np.sin(phi_grid) * np.sin(phi) + np.cos(phi_grid) * np.cos(phi) * np.cos(delta_long)
#     second_ort = np.abs(np.arctan2(numerator, denominator))
#     gaussian_orth = np.exp((-1.0/(2.0*np.square(0.1))) * np.square(second_ort))
#     return gaussian_orth
}


def get_xyz_grid(height, width, lib='torch'):
    # 生成网格:
    yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, width, endpoint=False),
                                        np.linspace(0, 1, height, endpoint=False))
    # 移到网格中部:
    yaw_grid += 1.0 / (2.0 * width)
    pitch_grid += 1.0 / (2.0 * height)
    # 转化为真正的角度:
    yaw_grid = yaw_grid * 2 * np.pi
    pitch_grid = pitch_grid * np.pi

    xyz_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)

    if lib == 'torch':
        return torch.tensor(xyz_grid).permute(1, 2, 0)
    elif lib == 'numpy':
        return xyz_grid


# 基于numpy的实现:
def salmap2posalfeat(xyz_grid, salmap):
    '''
    xyz_grid: np.array.shape = [H, W, 3];
    salmap: np.array.shape = [N, H, W];
    return: np.array.shape = [N, H*W, 4];  # 其中的4个通道分别是x, y, z, saliency;
    '''
    N, H, W = salmap.shape
    xyz_grid = np.tile(xyz_grid[np.newaxis, :, :, :], (N, 1, 1, 1))  # xyz_grid.shape = [N, H, W, 3];
    salmap = salmap[:, :, :, np.newaxis]  # salmap.shape = [N, H, W, 1];
    posalfeat = np.concatenate([xyz_grid, salmap], axis=-1)  # posalfeat.shape = [N, H, W, 4];
    posalfeat = posalfeat.reshape(N, H*W, 4)  # posalfeat.shape = [N, H*W, 4];
    return posalfeat


def motmap2motxyz(xyz_grid, motmap):
    '''
    xyz_grid: np.array.shape = [H, W, 3];
    motmap: np.array.shape = [N, H, W, 3];
    return: np.array.shape = [N, H*W, 6];  # 其中的6个通道分别是x, y, z, dx, dy, dz;
    '''
    N, H, W, _ = motmap.shape
    xyz_grid = np.tile(xyz_grid[np.newaxis, :, :, :], (N, 1, 1, 1))  # xyz_grid.shape = [N, H, W, 3];
    motxyz = np.concatenate([xyz_grid, motmap], axis=-1)  # motxyz.shape = [N, H, W, 6];
    motxyz = motxyz.reshape(N, H*W, 6)  # motxyz.shape = [N, H*W, 6];
    return motxyz


# Returns an array of size (NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL) with values between 0 and 1 specifying the probability that a tile is watched by the user
# We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
def from_position_to_tile_probability_cartesian(pos, xyz_grid, way=2):  # 将一个user在某一帧上的视点位置, 高斯模糊为一个heatmap (heigth, width)
    # # 方式1: 只能处理包含一个三维坐标的pos; 且使用numpy库进行处理
    if way == 1:
        assert isinstance(pos, np.ndarray) and isinstance(xyz_grid, np.ndarray)
        x_grid, y_grid, z_grid = xyz_grid
        great_circle_distance = np.arccos(np.maximum(np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0), -1.0))
        gaussian_orth = np.exp((-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance))
        return gaussian_orth
    
    # # 方式2: 能够处理包含多个三维坐标的pos; 尽可能将计算过程并行化;
    elif way == 2:
        assert isinstance(pos, torch.Tensor) and isinstance(xyz_grid, torch.Tensor)
        assert pos.shape[-1] == 3 and xyz_grid.shape[-1] == 3
        xyz_grid = xyz_grid.to(pos.device)
        # pos.shape = (batch_size, seq_len, 3); type=torch.tensor
        # xyz_grid.shape = (height, width, 3); type=torch.tensor
        batch_size = pos.shape[0]
        seq_len = pos.shape[1]
        height = xyz_grid.shape[0]
        width = xyz_grid.shape[1]
        try:
            pos = pos.view(batch_size*seq_len, 1,  -1).expand(-1, height*width, -1)
        except:
            pos = pos.reshape(batch_size*seq_len, 1,  -1).expand(-1, height*width, -1)
        xyz_grid = xyz_grid.view(1, height*width, -1).expand(batch_size*seq_len, -1, -1)
        great_circle_distance = compute_orthodromic_distance(pos, xyz_grid)
        gaussian_orth = torch.exp((-1.0 / (2.0 * torch.square(torch.tensor(0.1)))) * torch.square(great_circle_distance))
        gaussian_orth = gaussian_orth.view(batch_size, seq_len, 1, height, width).float()
        return gaussian_orth

    # # 方式3: 能够处理包含多个三维坐标的pos; 但计算过程采用串行的方式;
    elif way == 3:
        assert isinstance(pos, torch.Tensor) and isinstance(xyz_grid, torch.Tensor)
        assert pos.shape[-1] == 3 and xyz_grid.shape[-1] == 3
        xyz_grid = xyz_grid.to(pos.device)

        # pos.shape = (batch_size, seq_len, 3); type=torch.tensor
        # xyz_grid.shape = (height, width, 3); type=torch.tensor
        batch_size = pos.shape[0]
        seq_len = pos.shape[1]
        height = xyz_grid.shape[0]
        width = xyz_grid.shape[1]

        pos = pos.view(batch_size*seq_len, 3)
        x_grid, y_grid, z_grid = xyz_grid[:,:,0], xyz_grid[:,:,1], xyz_grid[:,:,2]

        gaussian_orth_lst = []
        for i in range(pos.shape[0]):
            great_circle_distance = torch.acos(torch.clamp(x_grid*pos[i][0] + y_grid*pos[i][1] + z_grid*pos[i][2], -1.0, 1.0))
            gaussian_orth = torch.exp((-1.0 / (2.0 * torch.square(torch.tensor(0.1)))) * torch.square(great_circle_distance))
            gaussian_orth_lst.append(gaussian_orth.float())
        ret = torch.stack(gaussian_orth_lst, dim=0).view(batch_size, seq_len, 1, height, width)
        return ret

    else:
        raise ValueError('way should be 1, 2 or 3.')



# TODO: 未完成; 但是其实也可以不用这一函数;
# def from_tile_probability_to_position_cartesian(heatmap):
#     tilemap = heatmap2tilemap(heatmap)
#     heatmap = heatmap[0][0].squeeze()
#     print(tilemap[0][0].squeeze())
#     row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
#     print(row, col)
#     y = (row+0.5) / float(heatmap.shape[0])
#     x = (col+0.5) / float(heatmap.shape[1])
#     pos_cartesian = eulerian_to_cartesian(x*2*np.pi, y*np.pi)
#     return pos_cartesian


def cal_thresh(degree_delta=100):  # 计算tile是否被选中的概率阈值;
    x0, y0, z0 = eulerian_to_cartesian(0, np.pi/2)
    x1, y1, z1 = eulerian_to_cartesian(degrees_to_radian(degree_delta/2), np.pi/2)
    great_circle_distance = np.arccos(np.maximum(np.minimum(x0 * x1 + y0 * y1 + z0 * z1, 1.0), -1.0))
    gaussian_orth = np.exp((-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance))
    return gaussian_orth  # 2.905975422353534e-17


def heatmap2tilemap(heatmap, thresh=2.9e-17):
    if isinstance(heatmap, np.ndarray):
        tilemap = np.empty_like(heatmap)
    elif isinstance(heatmap, torch.Tensor):
        tilemap = torch.empty_like(heatmap)
    tilemap[heatmap < thresh] = False
    tilemap[heatmap > thresh] = True
    return tilemap


def orthogonal(v):
    x = abs(v[0])
    y = abs(v[1])
    z = abs(v[2])
    other = (1, 0, 0) if (x < y and x < z) else (0, 1, 0) if (y < z) else (0, 0, 1)
    return cross(v, other)

def normalized(v):
    return normalize(v[:, np.newaxis], axis=0).ravel()


def rotationBetweenVectors(u, v):
    u = normalized(u)
    v = normalized(v)

    if np.allclose(u, v):
        return Quaternion(angle=0.0, axis=u)
    if np.allclose(u, -v):
        return Quaternion(angle=np.pi, axis=normalized(orthogonal(u)))

    quat = Quaternion(angle=np.arccos(dot(u, v)), axis=normalized(cross(u, v)))
    return quat

def degrees_to_radian(degree):
    return degree*np.pi/180.0

def radian_to_degrees(radian):
    return radian*180.0/np.pi

# time_orig_at_zero is a flag to determine if the time must start counting from zero, if so, the trace is forced to start at 0.0
def interpolate_quaternions(orig_times, quaternions, rate, time_orig_at_zero=True):
    # if the first time-stamps is greater than (half) the frame rate, put the time-stamp 0.0 and copy the first quaternion to the beginning
    if time_orig_at_zero and (orig_times[0] > rate/2.0):
        orig_times = np.concatenate(([0.0], orig_times))
        # ToDo use the quaternion rotation to predict where the position was at t=0
        quaternions = np.concatenate(([quaternions[0]], quaternions))
    key_rots = R.from_quat(quaternions)
    slerp = Slerp(orig_times, key_rots)
    # we add rate/2 to the last time-stamp so we include it in the possible interpolation time-stamps
    times = np.arange(orig_times[0], orig_times[-1]+rate/2.0, rate)
    # to bound it to the maximum original-time in the case of rounding errors
    times[-1] = min(orig_times[-1], times[-1])
    interp_rots = slerp(times)
    return np.concatenate((times[:, np.newaxis], interp_rots.as_quat()), axis=1)

# Compute the orthodromic distance between two points in 3d coordinates (x, y, z) or 2d coordinates (yaw, pitch);
def compute_orthodromic_distance(true_position, pred_position):
    # 如果true_position或者pred_position传入了heatmap形式的结果, 则不进行计算, 直接返回0;
    if len(true_position.shape) > 3:
        return torch.zeros(true_position.shape[0], true_position.shape[1], 1)
    if len(pred_position.shape) > 3:
        return torch.zeros(pred_position.shape[0], pred_position.shape[1], 1)
    
    if true_position.shape[-1] == 3:  # 3d coordinates: x, y, z
        norm_a = torch.sqrt(torch.square(true_position[:, :, 0:1]) + torch.square(true_position[:, :, 1:2]) + torch.square(true_position[:, :, 2:3]))
        norm_b = torch.sqrt(torch.square(pred_position[:, :, 0:1]) + torch.square(pred_position[:, :, 1:2]) + torch.square(pred_position[:, :, 2:3]))
        x_true = true_position[:, :, 0:1] / norm_a
        y_true = true_position[:, :, 1:2] / norm_a
        z_true = true_position[:, :, 2:3] / norm_a
        x_pred = pred_position[:, :, 0:1] / norm_b
        y_pred = pred_position[:, :, 1:2] / norm_b
        z_pred = pred_position[:, :, 2:3] / norm_b
        great_circle_distance = torch.acos(torch.clamp(x_true * x_pred + y_true * y_pred + z_true * z_pred, -1.0, 1.0))
    elif true_position.shape[-1] == 2:  # 2d coordinates: yaw, pitch
        yaw_true = (true_position[:, :, 0:1] - 0.5) * 2*torch.tensor(np.pi)
        pitch_true = (true_position[:, :, 1:2] - 0.5) * torch.tensor(np.pi)
        # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
        yaw_pred = (pred_position[:, :, 0:1] - 0.5) * 2*torch.tensor(np.pi)
        pitch_pred = (pred_position[:, :, 1:2] - 0.5) * torch.tensor(np.pi)
        # Finally compute orthodromic distance
        delta_long = torch.abs(torch.atan2(torch.sin(yaw_true - yaw_pred), torch.cos(yaw_true - yaw_pred)))
        numerator = torch.sqrt(torch.pow(torch.cos(pitch_pred)*torch.sin(delta_long), 2.0) + torch.pow(torch.cos(pitch_true)*torch.sin(pitch_pred)-torch.sin(pitch_true)*torch.cos(pitch_pred)*torch.cos(delta_long), 2.0))
        denominator = torch.sin(pitch_true)*torch.sin(pitch_pred)+torch.cos(pitch_true)*torch.cos(pitch_pred)*torch.cos(delta_long)
        great_circle_distance = torch.abs(torch.atan2(numerator, denominator))
    else:
        raise ValueError('The true and predicted positions must have 2 or 3 dimensions')
    return great_circle_distance


# 支持 3d coordinates (x, y, z) 或者 2d coordinates (yaw, pitch) 或者 heatmap
def compute_mse(true, pred, tilemap_shape):
    if tilemap_shape is None:
        tilemap_shape = TILEMAP_SHAPE
    WEIGHTED = False  # 是否赋予不同时间点不同的权重;
    if true.shape != pred.shape:  # true is 3d but pred is heatmap
        if len(true.shape) == 3 and true.shape[-1] == 3 and len(pred.shape) == 5 and pred.shape[-3] == 1: 
            xyz_grid = get_xyz_grid(*tilemap_shape)
            true = from_position_to_tile_probability_cartesian(true, xyz_grid)
    
    if not WEIGHTED:
        return F.mse_loss(pred, true)
    else:
        loss = F.mse_loss(pred, true, reduction='none')  # (B, T, 3)
        # # 按照e^(-t)计算weights
        # weights = torch.exp(-torch.arange(loss.shape[1]).float().to(loss.device))
        # # 按照0.98^t计算weights
        # weights = torch.pow(0.98, torch.arange(loss.shape[1]).float().to(loss.device))
        # # 按照0.95^t计算weights
        # weights = torch.pow(0.92, torch.arange(loss.shape[1]).float().to(loss.device))
        # 相同权重
        weights = torch.ones(loss.shape[1]).float().to(loss.device)
        # 还可以将前15个时间点的权重设为0
        weights[:15] = 0
        weights = weights / torch.sum(weights)
        weights = weights.view(1, -1, 1)
        loss = torch.mean(loss * weights)
        return loss


# 支持 3d coordinates (x, y, z) 或者 tilemap ( shape=(h, w, 1) )
def compute_acc_prec_reca_f1(true, pred, tilemap_shape):
    if tilemap_shape is None:
        tilemap_shape = TILEMAP_SHAPE
    xyz_grid = get_xyz_grid(*tilemap_shape)
    try:
        assert len(true.shape) == 5 and true.shape[-3] == 1  # (batch_size, seq_len, 1, height, width)
        assert len(pred.shape) == 5 and pred.shape[-3] == 1  # (batch_size, seq_len, 1, height, width)
    except:
        if len(true.shape) == 3 and true.shape[-1] == 3:  # (batch_size, seq_len, 3)
            true = from_position_to_tile_probability_cartesian(true, xyz_grid)
        if len(pred.shape) == 3 and pred.shape[-1] == 3:  # (batch_size, seq_len, 3)
            pred = from_position_to_tile_probability_cartesian(pred, xyz_grid)
    
    true_tilemap = heatmap2tilemap(true)  # (batch_size, seq_len, 1, height, width,)
    pred_tilemap = heatmap2tilemap(pred)  # (batch_size, seq_len, 1, height, width,)
    true_tilemap = torch.flatten(true_tilemap, start_dim=2, end_dim=-1)  # (batch_size, seq_len, height*width)
    pred_tilemap = torch.flatten(pred_tilemap, start_dim=2, end_dim=-1)  # (batch_size, seq_len, height*width)
    # fit = sum(map(sum, (pred_tilemap + true_tilemap) != 1))  # TP + FN
    fit = torch.sum((pred_tilemap + true_tilemap) != 1, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
    mistake = pred_tilemap.shape[-1] - fit
    # fetch = sum(map(sum, (pred_tilemap == 1)))  # P; (TP + FP)
    fetch = torch.sum(pred_tilemap == 1, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
    # need = sum(map(sum, (true_tilemap == 1)))  # T; (TP + TN)
    need = torch.sum(true_tilemap == 1, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
    # right = sum(map(sum, (pred_tilemap + true_tilemap) > 1))  # TP
    right = torch.sum((pred_tilemap + true_tilemap) > 1, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
    wrong = fetch - right

    eps = 1e-3
    accuracy = fit / true_tilemap.shape[-1]
    precision = (right + eps) / (fetch + eps)  # TP / TP + FP
    recall = (right + eps) / (need + eps)  # TP / TP + TN
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


def compute_manhattan_distance(true, pred, tilemap_shape):
    if tilemap_shape is None:
        tilemap_shape = TILEMAP_SHAPE
    true_x, true_y, true_z = true[:, :, 0:1], true[:, :, 1:2], true[:, :, 2:3]
    pred_x, pred_y, pred_z = pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3]
    true_row, true_col = in_which_tile(true_x, true_y, true_z, tilemap_shape)
    pred_row, pred_col = in_which_tile(pred_x, pred_y, pred_z, tilemap_shape)
    manh_dist_row = torch.abs(true_row - pred_row)
    manh_dist_col = torch.abs(true_col - pred_col)
    manh_dist_col = torch.where(manh_dist_col > tilemap_shape[1]/2, tilemap_shape[1] - manh_dist_col, manh_dist_col)
    manh_dist = manh_dist_row + manh_dist_col
    return manh_dist


def compute_mse_ctb(true, pred, tilemap_shape):
    if tilemap_shape is None:
        tilemap_shape = TILEMAP_SHAPE
    mse_loss = F.mse_loss(pred, true, reduction='none')
    manh_dist = compute_manhattan_distance(true, pred, tilemap_shape)
    cross_boundary_mask = (manh_dist>0).float()
    weight_for_cross_boundary = 2.0
    mse_ctb_loss = mse_loss * (1 + cross_boundary_mask * (weight_for_cross_boundary - 1))
    return torch.mean(mse_ctb_loss)


all_metrics = {}
all_metrics['orth'] = compute_orthodromic_distance
all_metrics['mse'] = compute_mse
all_metrics['aprf'] = compute_acc_prec_reca_f1
all_metrics['manh'] = compute_manhattan_distance
all_metrics['mse_ctb'] = compute_mse_ctb
# # 以下两个损失函数, 想实现的话也可以实现, 但总感觉没有必要, 先试用mse试试再说;
# 交叉熵损失函数需要硬标签
# KLDivLoss需要所有值相加为1, 但是我们生成的salmap不具有这一性质


# Returns the position in cartesian coordinates of the point with maximum saliency of the equirectangular saliency map
def get_max_sal_pos(saliency_map, dataset_name):
    row, col = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    y = row/float(saliency_map.shape[0])
    if dataset_name == 'Xu_CVPR_18':
        x = np.remainder(col/float(saliency_map.shape[1])-0.5, 1.0)
    else:
        x = col / float(saliency_map.shape[1])
    pos_cartesian = eulerian_to_cartesian(x*2*np.pi, y*np.pi)
    return pos_cartesian

def store_dict_as_csv(csv_file, csv_columns, dict_data):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def load_dict_from_csv(filename, sep=',', header=0, engine='python'):
    dataframe = pd.read_csv(filename, engine=engine, header=header, sep=sep, dtype=str)
    return dataframe.values


def data_rotation(data, angle):
    """
    对输入的数据进行旋转

    :param data: 待旋转的数据，shape 为 (batch_size, seq_len, 3)
    :param angle: 旋转角度，单位为度数
    :return: 旋转后的数据，shape 与输入数据相同
    """
    # 将角度转为弧度
    angle = torch.tensor(angle * np.pi / 180.0, dtype=data.dtype, device=data.device)

    # 计算旋转矩阵
    rot_matrix = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ], dtype=data.dtype, device=data.device)

    # 对数据进行旋转
    data_rotated = torch.matmul(data, rot_matrix)

    # 返回旋转后的数据
    return data_rotated


def test_data_rotation():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 创建一个 3 维的球面坐标数据
    batch_size = 1
    seq_len = 4
    data = torch.tensor([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [-1.0,0.0, 0.0],]).float().to(device)
    
    # 将数据绕 z 轴旋转 45 度
    data_rotated = data_rotation(data, 45)

    print(data_rotated)


def normalize_coordinates(points):
    # 计算每个坐标点的欧几里得范数
    norms = np.linalg.norm(points, axis=1)

    # 将坐标点除以其欧几里得范数
    normalized_points = points / norms[:, np.newaxis]

    return normalized_points


import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


def plot_trajectories(save_path, x, y, y_pred):
    y_pred = normalize_coordinates(y_pred)

    # 创建一个三维坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制单位球
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)

    # 绘制轨迹
    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='orange', label='x')
    ax.plot(y[:, 0], y[:, 1], y[:, 2], color='green', label='y')
    ax.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], color='blue', label='y_pred')

    # 设置图例和坐标轴标签
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.savefig(save_path)
    plt.clf()
    plt.close()

    # print('-' * 20, '过去 M_WINDOW 个视点的坐标:', '-' * 20)
    # print(x)
    # print('-' * 20, '未来 H_WINDOW 个视点的真实坐标:', '-' * 20)
    # print(y)
    # print('-' * 20, '未来 H_WINDOW 个视点的预测坐标:', '-' * 20)
    # print(y_pred)
    # print()


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    ret = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    print(ret.shape)
    print(ret.dtype)
    print(ret.max())
    print(ret.min())
    exit()
    return ret


def sample_frame_indices(clip_len, frame_rate, end_ms, duration_ms=5000):
    end_idx = int(end_ms / (1000 / frame_rate))
    start_idx = int((end_ms - duration_ms) / (1000 / frame_rate))  # end_ms一定会大于duration_ms
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices



def find_closest_elements(list1, list2):
    list1 = np.array(list1) if not isinstance(list1, np.ndarray) else list1
    list2 = np.array(list2) if not isinstance(list2, np.ndarray) else list2
    idx = np.abs(list2[:, np.newaxis] - list1).argmin(axis=0)
    return list2[idx]


# def find_closest_elements(list1, list2):
#     # 将输入的list转换为Numpy数组
#     arr1 = np.array(list1)
#     arr2 = np.array(list2)
    
#     # 初始化空列表来存储结果
#     result = []
    
#     # 遍历list1中的每个元素
#     for elem in arr1:
#         # 计算list2中每个元素与当前元素的差值的绝对值
#         diff = np.abs(arr2 - elem)
        
#         # 找到差值最小的元素的索引
#         min_index = np.argmin(diff)
        
#         # 将最接近的元素添加到结果列表中
#         result.append(arr2[min_index])
    
#     # 去除重复的结果并返回
#     return list(set(result))


def sample_dataframe(df, col, interval=0.2):
    df.sort_values(by=col, inplace=True)
    result_df = pd.DataFrame(columns=df.columns)
    
    # 方式1: 
    begin_pt = 0
    end_pt = df[col].max()
    for pt in np.arange(begin_pt, end_pt, interval):
        diff = (df[col] - pt).abs()  # 计算PlayTime列中每个元素与当前采样点的差值的绝对值
        min_index = diff.idxmin()  # 找到差值最小的元素的索引
        result_df.loc[len(result_df)] = df.loc[min_index]  # 将最接近的元素添加到结果DataFrame中 (使用append方法会报错, 所以使用loc方法)
    
    # # 方式2: 遍历df; 本以为效率较高, 但结果不如方式1, 这应该是因为方式1使用了向量化的计算;
    # pt = 0
    # i = 0
    # while i < len(df):
    #     diff = np.inf
    #     while i < len(df) and abs(df[col][i] - pt) < diff:
    #         diff = abs(df[col][i] - pt)
    #         i += 1
    #     result_df.loc[len(result_df)] = df.loc[i - 1]
    #     pt += interval

    return result_df


def get_bests_worsts_vuxi(dataset_name, model_name, init_window, m_window, h_window, end_window, num=20, stride=10, met='orthodromic_distance'):
    if isinstance(model_name, str):  # get单个模型的bests和worsts
        # 读取数据:
        df = pd.read_csv(f'./{dataset_name}/{model_name}/Results_init_{init_window}_in_{m_window}_out_{h_window}_end_{end_window}/errors.csv')
        # 按照video,user,x_i分组, 每组求平均error:
        df = df.groupby(['video', 'user', 'x_i']).mean().reset_index()
        # 将df分别按照每个模型列的值从小到大排序, 并且选出最好的10个和最差的10个:
        df = df.sort_values(by=[met])
    elif isinstance(model_name, list) and len(model_name)==2:  # get两个模型的bests和worsts
        mn0, mn1 = model_name[0], model_name[1]
        df0 = pd.read_csv(f'./{dataset_name}/{mn0}/Results_init_{init_window}_in_{m_window}_out_{h_window}_end_{end_window}/errors.csv')
        df0 = df0.groupby(['video', 'user', 'x_i']).mean().reset_index()
        df1 = pd.read_csv(f'./{dataset_name}/{mn1}/Results_init_{init_window}_in_{m_window}_out_{h_window}_end_{end_window}/errors.csv')
        df1 = df1.groupby(['video', 'user', 'x_i']).mean().reset_index()
        df = pd.DataFrame(columns=['video','user','x_i',f'{mn0}-{mn1}'])
        df[['video','user','x_i']] = df0[['video','user','x_i']]
        df[f'{mn0}-{mn1}'] = df0[met] - df1[met]
        df = df.sort_values(by=[f'{mn0}-{mn1}'])
    else:
        raise ValueError('model_name should be a str or a list with length 2.')
    
    # 以num为个数, stride为步长, 选出bests和worsts:
    bests = df[['video','user','x_i']].iloc[::stride].head(num).values.tolist()
    worsts = df[['video','user','x_i']].iloc[::stride].tail(num).values.tolist()
    return bests, worsts


def in_which_tile(x, y, z, tilemap_shape):
    theta, phi = cartesian_to_eulerian(x, y, z, lib='torch')
    row = (phi / np.pi * tilemap_shape[0]).long()
    col = (theta / (2*np.pi) * tilemap_shape[1]).long()
    return row, col
    


if __name__ == '__main__':
    pass


    xyz = quaternion_to_cartesian(0.707,0,0,0.707)#(-0.002, -0.586, 0.195, -0.787)
    print(xyz)

    # # 测试
    # df = pd.DataFrame(columns=["pt"])
    # df["pt"] = [0.11, 0.19, 0.23, 0.27, 0.40, 0.55, 0.57, 0.62, 0.71, 0.77, 0.81, 0.88, 0.92, 1.01]
    # sampled_df = sample_dataframe(df, 'pt', interval=0.2)
    # print(sampled_df)


    # list1 = [2, 5, 6]
    # list2 = [1, 4, 5, 6, 7]
    # result = find_closest_elements(list1, list2)
    # print(type(result))
    # print(result)



    # video_path = '../../datasets/vr-dataset/vid-prep/1-1.mp4'
    # container = av.open(video_path)
    # frame_rate = round(float(container.streams.video[0].average_rate))
    # indices = sample_frame_indices(clip_len=16, frame_rate=frame_rate, end_ms=10000)
    # image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    # # model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    # inputs = image_processor(list(read_video_pyav(container, indices)), return_tensors="pt")
    # model_inputs = inputs["pixel_values"]
    # print(model_inputs.shape)



    # batch_size = 4
    # a = torch.randn(batch_size, 2)
    # x = a[:, 0:1]
    # y = a[:, 1:2]
    # z = torch.tensor([[1.0]]*batch_size, dtype=torch.float32)
    # print(x.shape, y.shape, z.shape)
    # xyz = torch.cat([x, y, z], dim=1)
    # print(xyz)

    # x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # xx = torch.tensor([[10, 11, 12]])
    # x = torch.cat([x, xx], dim=0)
    # print(x)  # tensor([[ 1,  2,  3], [ 4,  5,  6], [ 7,  8,  9], [10, 11, 12]])


    # from PIL import Image
    # import py360convert

    # image_path = '../../datasets/vr-dataset/vid-frames/6/450000.jpg'
    # img = np.array(Image.open(image_path))
    # print(img.shape)
    # img = py360convert.e2p(img, fov_deg=(90, 90), u_deg=-180, v_deg=0, out_hw=(5, 5))
    # print(type(img))
    # print(img.shape)
    # img = Image.fromarray(img)
    # img.save('fov.png')


    # pos = torch.tensor(
    #     [
    #         [
    #             [1, 0, 0],
    #             [0, 1, 0],
    #             [0, 0, 1],
    #             [1, 0, 0],
    #         ],
    #         [
    #             [1, 0, 0],
    #             [0, 1, 0],
    #             [0, 0, 1],
    #             [1, 0, 0],
    #         ],
    #     ]
    # )

    # # # print(pos.shape)  # 2, 4, 3
    # # # exit()

    # xyz_grid = get_xyz_grid(5, 8)
    # heatmap = from_position_to_tile_probability_cartesian(pos=pos, xyz_grid=xyz_grid)
    # tilemap = heatmap2tilemap(heatmap)
    # print(tilemap[0][0][0])

    # xyz = pos.permute(2, 0, 1)
    # t, b, l, r = xyz2bound(xyz, TILEMAP_SHAPE)
    
    # for bi in range(2):
    #     for si in range(4):
    #         print(pos[bi][si], t[bi, si].item(), b[bi, si].item(), l[bi, si].item(), r[bi, si].item())

    # pos2 = torch.tensor(
    #     [
    #         [
    #             [0, 1, 0],
    #             [0, 0, 1],
    #             [1, 0, 0],
    #             [1, 0, 0],
    #         ],
    #         [
    #             [0, 1, 0],
    #             [0, 0, 1],
    #             [1, 0, 0],
    #             [1, 0, 0],
    #         ],
    #     ]
    # )

    # # print(pos.shape)  # 2, 4, 3
    # # exit()

    # xyz_grid = get_xyz_grid(*TILEMAP_SHAPE)
    # heatmap = from_position_to_tile_probability_cartesian(pos=pos, xyz_grid=xyz_grid)
    
    # # print(heatmap.shape)  # torch.Size([2, 4, 9, 16, 1])
    # # # print(type(heatmap))  # <class 'numpy.ndarray'>
    # # # print(heatmap)
    # # # print(cal_thresh())  # 2.905975422353534e-17
    # tilemap = heatmap2tilemap(heatmap=heatmap)
    # tilemap = tilemap.squeeze()
    # # print(tilemap[0][2])
    
    # # acc, prec, reca, f1_score = compute_acc_prec_reca_f1(pos, pos2)
    # # print(acc.squeeze())
    # # print(prec.squeeze())
    # # print(reca.squeeze())
    # # print(f1_score.squeeze())


    # salmap = np.array([
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0.6, 0.7, 0.8, 0.9, 1.0],
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0.6, 0.7, 0.8, 0.9, 1.0],
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    # ])
    # salmap_ds = resize_image(salmap, (3, 3))
    # print(salmap_ds)



    # xyz_grid = get_xyz_grid(*TILEMAP_SHAPE, lib='numpy')
    # heatmap = from_position_to_tile_probability_cartesian(pos=pos, xyz_grid=xyz_grid, way=1)
    # tilemap = heatmap2tilemap(heatmap=heatmap)
    # print(tilemap)

    # t, b, l, r = pos2bound(pos, TILEMAP_SHAPE)
    # print(t, b, l, r)