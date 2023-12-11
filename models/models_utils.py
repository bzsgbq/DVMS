import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class MaxPool(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=(2,2)):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.maxpool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        
    def forward(self, x):
        if len(x.shape) == 4:
            return self.maxpool(x)
        elif len(x.shape) == 5:
            batch_size, seq_len, c, h, w = x.shape
            x = x.view(batch_size*seq_len, c, h, w)
            x = self.maxpool(x)
            c, h, w = x.shape[-3:]
            x = x.view(batch_size, seq_len, c, h, w)
            return x
        else:
            raise ValueError('x.shape must be (batch_size, c, h, w) or (batch_size, seq_len, c, h, w)')


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


def metric_orth_dist(position_a, position_b):
    # Normalize onto the unit sphere
    position_a /= torch.linalg.norm(position_a, dim=-1).unsqueeze(-1).repeat(1, 1, 3)
    position_b /= torch.linalg.norm(position_b, dim=-1).unsqueeze(-1).repeat(1, 1, 3)
    # Finally compute orthodromic distance
    great_circle_distance = 2 * torch.asin(torch.linalg.norm(position_b - position_a, dim=-1) / 2)
    return great_circle_distance


def flat_top_k_orth_dist(k_position_a, position_b, k):
    batch_size, seq_len, _ = position_b.shape
    k_position_b = torch.repeat_interleave(position_b, k, dim=0)
    k_orth_dist = metric_orth_dist(k_position_a, k_position_b).reshape((batch_size, k, seq_len))
    _, best_orth_dist_idx = torch.min(torch.mean(k_orth_dist, dim=-1), dim=-1)
    best_orth_dist = k_orth_dist[range(batch_size), best_orth_dist_idx]
    return best_orth_dist


def to_position_normalized(values):
    orientation = values[0]
    motion = values[1]
    result = orientation + motion
    return result / torch.norm(result, dim=-1).reshape(-1, 1, 1).repeat(1, 1, 3)

# This way we ensure that the network learns to predict the delta angle
def toPosition(values):
    orientation = values[0]
    # The network returns values between 0 and 1, we force it to be between -1/2 and 1/2
    motion = values[1]
    return (orientation + motion)

def selectImageInModel(input_to_selector, curr_idx):
    selected_image = input_to_selector[:, curr_idx:curr_idx+1]
    return selected_image



def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


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


def get_H_half_and_W_half(salmap_shape):
    H_half = int(salmap_shape[0] / 180 * 100 / 2)
    W_half = int(salmap_shape[1] / 360 * 100 / 2)
    return H_half, W_half


def get_cliped_shape(salmap_shape):
    H_half, W_half = get_H_half_and_W_half(salmap_shape)
    H_clip = 2 * H_half + 1
    W_clip = 2 * W_half + 1
    return H_clip, W_clip


def xyz2bound(xyz, salmap_shape):
    x, y, z = xyz
    H_half, W_half = get_H_half_and_W_half(salmap_shape)
    theta, phi = cartesian_to_eulerian(x, y, z, lib='torch')  # theta.shape = (batch_size, seq_len); phi.shape:(batch_size, seq_len)
    center = torch.stack([phi, theta], dim=-1)  # center.shape = (batch_size, seq_len, 2)
    center = center / torch.tensor([np.pi, 2*np.pi], dtype=center.dtype, device=center.device) * torch.tensor(salmap_shape, dtype=center.dtype, device=center.device)
    center = center.round().int().permute(2, 0, 1)  # center.shape = (2, batch_size, seq_len)
    top = center[0] - torch.tensor(H_half, dtype=center.dtype, device=center.device)  # top.shape = (batch_size, seq_len)
    bottom = center[0] + torch.tensor(H_half, dtype=center.dtype, device=center.device)  # bottom.shape = (batch_size, seq_len)
    left = center[1] - torch.tensor(W_half, dtype=center.dtype, device=center.device)  # left.shape = (batch_size, seq_len)
    right = center[1] + torch.tensor(W_half, dtype=center.dtype, device=center.device)  # right.shape = (batch_size, seq_len)

    return top, bottom, left, right


# def clip_sal(salmap, pos):
#     '''
#     :param salmap: (batch_size, seq_len, 1, salmap_shape[0], salmap_shape[1])
#     :param pos: (batch_size, seq_len, 3)
#     :return: salmap_clip: (batch_size, seq_len, 1, H, W)
#     '''
#     xyz = pos.permute(2, 0, 1)
#     salmap_shape = salmap.shape[-2:]
#     batch_size, seq_len = salmap.shape[:2]

#     top, bottom, left, right = xyz2bound(xyz, salmap_shape)
#     H_clip, W_clip = get_cliped_shape(salmap_shape)
#     salmap_clip = torch.zeros((batch_size, seq_len, 1, H_clip, W_clip), dtype=salmap.dtype, device=salmap.device)

#     # 生成网格
#     h_indices = torch.arange(H_clip, dtype=torch.long, device=salmap.device).unsqueeze(-1).expand(-1, W_clip)
#     w_indices = torch.arange(W_clip, dtype=torch.long, device=salmap.device).unsqueeze(0).expand(H_clip, -1)

#     for bi in range(batch_size):
#         for ti in range(seq_len):
#             # 将裁剪后的网格坐标转换为原始salmap中的坐标
#             i = top[bi, ti] + h_indices
#             j = left[bi, ti] + w_indices
#             j = torch.remainder(j, salmap_shape[1])

#             # 计算mask
#             mask = (i < 0) | (i >= salmap_shape[0])
#             i[mask] = 0  # 这里的设为0, 并不能将超出原始salmap边界的像素设为0, 只是给i中超出边界的index一个合法值而已;

#             # 从原始salmap中取出裁剪后的像素
#             salmap_clip[bi, ti, :, :, :] = salmap[bi, ti, :, i, j]
#             salmap_clip[bi, ti, :, i[mask], :] = 0  # 将超出原始salmap边界的像素设为0

#     return salmap_clip


def xyz_to_uvd(xyz, lib='torch'):  # xyz to u_deg and v_deg
    x, y, z = xyz[0], xyz[1], xyz[2]
    theta, phi = cartesian_to_eulerian(x, y, z, lib=lib)
    u_deg = (theta-np.pi) / np.pi * 180
    v_deg = (np.pi/2-phi) / np.pi * 180
    return u_deg, v_deg

def uvd_to_xyz(u_deg, v_deg, lib='torch'):  # u_deg and v_deg to xyz
    theta = (u_deg / 180 * np.pi) + np.pi
    phi = np.pi/2 - (v_deg / 180 * np.pi)
    xyz = eulerian_to_cartesian(theta, phi, lib=lib)
    return xyz


# 以下几个函数是从py360convert.utils.py中改写来的, 将原本对numpy数组的操作改为对torch.tensor的操作;

def rotation_matrix(rad, ax):
    # # test:
    # ax = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.float32, device=ax.device)
    # rad = torch.tensor([1, 2], dtype=torch.float32, device=ax.device)
    '''
    :param rad: tensor.shape = (batch_size,)
    :param ax: tensor.shape = (3)
    :return: 旋转矩阵
    '''
    device = ax.device
    batch_size = rad.shape[0]

    if len(ax.shape) == 1:
        ax = ax.unsqueeze(0).expand(batch_size, -1)
    else:
        assert len(ax.shape) == 2 and ax.shape[0] == batch_size

    ax = ax / torch.norm(ax, dim=1, keepdim=True)  # 对ax进行归一化 (每个batch分别进行归一化)

    # NumPy: R = np.diag([np.cos(rad)] * 3)
    R = torch.zeros((batch_size, 3, 3), device=device)
    cos_rad = torch.cos(rad)
    R[:, 0, 0] = cos_rad
    R[:, 1, 1] = cos_rad
    R[:, 2, 2] = cos_rad

    # NumPy: R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))
    R = R + torch.bmm(ax.unsqueeze(-1), ax.unsqueeze(1)) * (1.0 - torch.cos(rad)).unsqueeze(-1).unsqueeze(-1)

    # Numpy: ax = ax * np.sin(rad)
    ax = ax * torch.sin(rad).unsqueeze(-1)

    # Numpy: R = R + np.array([[0, -ax[2], ax[1]],
                            # [ax[2], 0, -ax[0]],
                            # [-ax[1], ax[0], 0]])
    ax = ax.unsqueeze(-1).expand(batch_size, 3, 3)  # 添加一个维度以匹配 R 的最后一个维度, 并且broadcast ax 到 R 的形状
    rot_mat = torch.zeros(batch_size, 3, 3).to(device)
    rot_mat[:, 0, 1] = -ax[:, 2, 0]
    rot_mat[:, 0, 2] = ax[:, 1, 0]
    rot_mat[:, 1, 0] = ax[:, 2, 0]
    rot_mat[:, 1, 2] = -ax[:, 0, 0]
    rot_mat[:, 2, 0] = -ax[:, 1, 0]
    rot_mat[:, 2, 1] = ax[:, 0, 0]
    R = R + rot_mat  # # 将旋转矩阵加到 R 上

    return R


def get_Rxyi(u, v, in_rot):
    if isinstance(u, torch.Tensor):
        device = u.device
    elif isinstance(v, torch.Tensor):
        device = v.device
    elif isinstance(in_rot, torch.Tensor):
        device = in_rot.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    rad_x = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=device)
    ax_x = torch.tensor([1.0, 0, 0], dtype=torch.float32, device=device)#, requires_grad=True)
    Rx = rotation_matrix(rad_x, ax=ax_x)

    rad_y = u if isinstance(u, torch.Tensor) else torch.tensor(u, dtype=torch.float32, device=device)
    ax_y = torch.tensor([0, 1.0, 0], dtype=torch.float32, device=device)#, requires_grad=True)
    Ry = rotation_matrix(rad_y, ax=ax_y)

    rad_i = in_rot if isinstance(in_rot, torch.Tensor) else torch.tensor(in_rot, dtype=torch.float32, device=device)
    # ax_i = torch.tensor([0, 0, 1.0], dtype=torch.float32).matmul(Rx).matmul(Ry)
    ax_i = torch.tensor([0, 0, 1.0], dtype=torch.float32, device=device)
    ax_expanded = ax_i.unsqueeze(0).unsqueeze(-1)
    ax_i = torch.matmul(torch.matmul(ax_expanded.transpose(1, 2), Rx), Ry).squeeze(1)
    Ri = rotation_matrix(rad_i, ax=ax_i)

    return Rx.detach(), Ry.detach(), Ri.detach()


def xyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    batch_size, device = u.shape[0], u.device
    out = torch.ones((batch_size, *out_hw, 3), dtype=torch.float32, device=device)

    h_fov = torch.tensor(h_fov, dtype=torch.float32, device=device)
    v_fov = torch.tensor(v_fov, dtype=torch.float32, device=device)

    x_max = torch.tan(h_fov / 2)
    y_max = torch.tan(v_fov / 2)
    x_rng = torch.linspace(-x_max, x_max, steps=out_hw[1], dtype=torch.float32, device=device)
    y_rng = torch.linspace(-y_max, y_max, steps=out_hw[0], dtype=torch.float32, device=device)
    grid_x, grid_y = torch.meshgrid(x_rng, -y_rng)  # torch.meshgird 的默认indexing方式是 'ij', 这与Numpy默认的indexing方式 'xy'不同, 而且由于我所使用的1.9版本的torch中所提供的torch.meshgrid函数不支持indexing参数, 所以我只能手动将grid_x和grid_y进行转置以达到与Numpy中meshgrid函数的效果一致
    grid_x = grid_x.permute(1, 0)
    grid_y = grid_y.permute(1, 0)
    out[..., :2] = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    Rx, Ry, Ri = get_Rxyi(u, v, in_rot)
    Rx = Rx.unsqueeze(1).unsqueeze(1).expand(-1, *out_hw, -1, -1)
    Ry = Ry.unsqueeze(1).unsqueeze(1).expand(-1, *out_hw, -1, -1)
    Ri = Ri.unsqueeze(1).unsqueeze(1).expand(-1, *out_hw, -1, -1)
    out = out.unsqueeze(-2)
    ret = out.matmul(Rx).matmul(Ry).matmul(Ri).squeeze(-2)

    # print(Rx.shape)
    # print(out.shape)
    # print(ret.shape)

    return ret


def xyz2uv(xyz):
    '''
    xyz: tensor in shape of [..., 3]
    '''
    x, y, z = torch.split(xyz, 1, dim=-1)
    u = torch.atan2(x, z)
    c = torch.sqrt(x**2 + z**2)
    v = torch.atan2(y, c)

    return torch.cat([u, v], dim=-1)


def uv2unitxyz(uv):
    u, v = torch.split(uv, 1, dim=-1)
    y = torch.sin(v)
    c = torch.cos(v)
    x = c * torch.sin(u)
    z = c * torch.cos(u)

    return torch.cat([x, y, z], dim=-1)


def uv2coor(uv, h, w):
    '''
    uv: tensor in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = torch.split(uv, 1, dim=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return torch.cat([coor_x, coor_y], dim=-1)


# def sample_equirec(e_img, coor_xy, mode='bilinear'):
#     '''
#     e_img: tensor in shape of [B, h_in, w_in]
#     coor_xy: tensor in shape of [B, h_out, w_out, 2]
#     '''
#     batch_size, h_in, w_in = e_img.shape
#     h_out, w_out = coor_xy.shape[1], coor_xy.shape[2]
    
#     coor_x, coor_y = torch.split(coor_xy, 1, dim=-1)
#     pad_u = torch.roll(e_img[:, [0]], shifts=w_in // 2, dims=[2])
#     pad_d = torch.roll(e_img[:, [-1]], shifts=w_in // 2, dims=[2])
#     e_img = torch.cat([e_img, pad_d, pad_u], dim=1)
    
#     coor = torch.cat([coor_y, coor_x], dim=-1)

#     e_img = e_img.unsqueeze(1)  # [B, 1, h_in, w_in]
#     output = F.grid_sample(e_img, coor, mode=mode)
#     ret = output.squeeze(1)

#     print(ret[0])
#     exit()
#     return ret


def sample_equirec(e_img, coor_xy, mode='bilinear'):
    batch_size, h_in, w_in = e_img.shape
    _, h_out, w_out, _ = coor_xy.shape

    coor_x, coor_y = torch.split(coor_xy, 1, dim=-1)
    pad_u = torch.roll(e_img[:, [0]], w_in // 2, 2)
    pad_d = torch.roll(e_img[:, [-1]], w_in // 2, 2)
    e_img = torch.cat([e_img, pad_d, pad_u], 1)

    grid_y = (coor_y / (h_in - 1)) * 2 - 1
    grid_x = (coor_x / (w_in - 1)) * 2 - 1
    grid = torch.stack([grid_x[..., 0], grid_y[..., 0]], dim=-1)
    
    sampled = F.grid_sample(e_img.unsqueeze(1), grid, mode=mode, padding_mode='reflection', align_corners=True)
    ret = sampled.squeeze(1)

    return ret


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear'):
    '''
    e_img:   tensor in shape of [B, H, W, *]
    fov_deg: scalar or (scalar, scalar) field of view in degree
    u_deg:   tensor in shape of [B], every element is horizon viewing angle in range [-180, 180]
    v_deg:   tensor in shape of [B], every element is vertical viewing angle in range [-90, 90]
    in_rot_deg: int, rotation angle of the input equirectangular image in degree
    '''
    batch_size, device = e_img.shape[0], e_img.device
    assert len(e_img.shape) == 4
    h, w = e_img.shape[1], e_img.shape[2]

    h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180

    u = -u_deg * np.pi / 180  # tensor in shape of [B], every element is horizon viewing angle in range [-pi, pi]   
    v = v_deg * np.pi / 180  # tensor in shape of [B], every element is vertical viewing angle in range [-pi/2, pi/2]
    in_rot = torch.tensor([in_rot_deg*np.pi/180]*batch_size, dtype=torch.float32, device=device)

    xyz = xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    pers_img = torch.stack([
        sample_equirec(e_img[..., i], coor_xy, mode=mode)
        for i in range(e_img.shape[-1])
    ], dim=-1)

    return pers_img


def get_fov(pos, img, fov_deg=(90, 90), fov_shape=(224, 224)):
    '''
    :param pos: (B, 3,)
    :param img: (B, H, W, 3,)
    :return: fov: (B, *fov_shape, 3,)
    '''
    xyz = pos.permute(1, 0)
    u_deg, v_deg = xyz_to_uvd(xyz)  # (batch_size, )
    fov = e2p(img, fov_deg=fov_deg, u_deg=u_deg, v_deg=v_deg, out_hw=fov_shape)
    return fov


# # 已废弃
# def clip_sal(salmap, pos):
#     '''
#     :param salmap: (batch_size, seq_len, 1, salmap_shape[0], salmap_shape[1])
#     :param pos: (batch_size, seq_len, 3)
#     :return: salmap_clip: (batch_size, seq_len, 1, H, W)
#     '''
#     salmap_shape = salmap.shape[-2:]
#     batch_size, seq_len = salmap.shape[:2]

#     fov_h, fov_v = 90, 90
#     H_clip, W_clip = salmap_shape[0] * fov_v // 180, salmap_shape[1] * fov_h // 360
#     salmap_clip = torch.zeros((batch_size, seq_len, 1, H_clip, W_clip), dtype=salmap.dtype, device=salmap.device)

#     for bi in range(batch_size):
#         for ti in range(seq_len):
#             img = salmap[bi, ti, :, :, :].cpu().numpy().transpose(1, 2, 0)
#             xyz = pos[bi, ti, :].cpu().numpy()
#             fov = get_fov(xyz, img, fov_h=fov_h, fov_v=fov_v, H_clip=H_clip, W_clip=W_clip)
#             fov = torch.tensor(fov.transpose(2, 0, 1), dtype=salmap.dtype, device=salmap.device)
#             salmap_clip[bi, ti] = fov

#     return salmap_clip


def get_new_xyz(old_xyz, fovxy, fov_deg):  # fovxy中, x轴的方向向右, y轴的方向向上;
    batch_size = fovxy.shape[0]
    device = fovxy.device

    h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
    h_fov = torch.tensor(h_fov, dtype=torch.float32)
    v_fov = torch.tensor(v_fov, dtype=torch.float32)

    u_deg, v_deg = xyz_to_uvd(old_xyz.clone().detach())
    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180  # u, v是当前已知的最新视点的uv坐标
    in_rot = torch.zeros_like(u).to(device)
    Rx, Ry, Ri = get_Rxyi(u, v, in_rot)

    x_max, y_max = torch.tan(h_fov / 2), torch.tan(v_fov / 2)  # 以下几行的x, y, z不是指三维空间坐标中的x, y, z, 具体含义与py360convert.utils.py中的xyzpers中的含义相同;
    x_pred, y_pred = fovxy[:, 0:1], fovxy[:, 1:2]  # 模型预测出的是下一时刻视点相对于当前时刻视点的FoV上的偏移量'
    x_pred, y_pred = x_pred * x_max, y_pred * y_max
    z = torch.tensor([[1.0]]*batch_size, dtype=torch.float32).to(device)
    xyz_pred = torch.cat([x_pred, y_pred, z], dim=1)

    viewpoint_xyz = xyz_pred.unsqueeze(-2).matmul(Rx).matmul(Ry).matmul(Ri).squeeze(-2)
    viewpoint_uv = xyz2uv(viewpoint_xyz)
    u_deg, v_deg = viewpoint_uv[:, 0] * 180 / np.pi, viewpoint_uv[:, 1] * 180 / np.pi
    new_xyz = uvd_to_xyz(u_deg, v_deg)

    return new_xyz


def xyz2fovxy(xyz1, xyz2, fov_deg):  # 相当于get_new_xyz函数的逆过程; xyz1是当前视点, xyz2是下一时刻视点;
    batch_size = xyz1.shape[0]
    device = xyz1.device

    h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
    h_fov = torch.tensor(h_fov, dtype=torch.float32)
    v_fov = torch.tensor(v_fov, dtype=torch.float32)

    # 通过xyz1计算旋转矩阵:
    u_deg, v_deg = xyz_to_uvd(xyz1.clone().detach())
    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180  # u, v是当前已知的最新视点的uv坐标
    in_rot = torch.zeros_like(u).to(device)
    Rx, Ry, Ri = get_Rxyi(u, v, in_rot)

    # 计算unitxyz:

    # # 方式1: get_new_xyz函数后半部分的逆过程:
    # u_deg, v_deg = xyz_to_uvd(xyz2)
    # u, v = u_deg * np.pi / 180, v_deg * np.pi / 180
    # unitxyz = uv2unitxyz(torch.stack([u, v], dim=1))
    
    # 方式2: 直接根据xyz2写出unitxyz: (xyz2和unitxyz只是坐标系不同)
    unitxyz = torch.stack([-xyz2[1, :], xyz2[2, :], -xyz2[0, :]], dim=1)

    # 计算真正的xyz: (xyz需要在unitxyz的基础上进行缩放)
    dot_product = torch.sum(xyz1 * xyz2, dim=0)  # 计算xyz1和xyz2之间的夹角: (xyz1和xyz2的shape都是[3, batch_size])
    xyz_len = 1 / dot_product  # 这就是实际的xyz的长度:
    xyz = unitxyz * xyz_len.unsqueeze(-1).expand(-1, 3)  # 这里xyz的shape是[batch_size, 3]

    Rx_inv, Ry_inv, Ri_inv = torch.inverse(Rx), torch.inverse(Ry), torch.inverse(Ri)
    xyz_pred = xyz.unsqueeze(-2).matmul(Ri_inv).matmul(Ry_inv).matmul(Rx_inv).squeeze(-2)
    x_max, y_max = torch.tan(h_fov / 2), torch.tan(v_fov / 2)
    x_pred, y_pred = xyz_pred[:, 0:1]/x_max, xyz_pred[:, 1:2]/y_max

    fovxy = torch.cat([x_pred, y_pred], dim=1)
    return fovxy


# 从utils.py中复制过来的函数:
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


# 基于torch的实现: 
def salmap2posalfeat(xyz_grid, salmap):  # 将salmap转换为嵌入了pos信息的saliency feature;
    '''
    xyz_grid: tensor.shape = [H, W, 3];
    salmap: tensor.shape = [B, T, 1, H, W];
    return: tensor.shape = [B, T, H*W, 4];  # 其中的4个通道分别是x, y, z, saliency;
    '''
    B, T, _, H, W = salmap.shape
    xyz_grid = xyz_grid.unsqueeze(0).unsqueeze(0).expand(B, T, H, W, 3)  # xyz_grid.shape = [B, T, H, W, 3];
    salmap = salmap.view(B, T, H, W, 1)  # salmap.shape = [B, T, H, W, 1];
    posalfeat = torch.cat([xyz_grid, salmap], dim=-1)  # posalfeat.shape = [B, T, H, W, 4];
    posalfeat = posalfeat.view(B, T, H*W, 4)  # posalfeat.shape = [B, T, H*W, 4];
    return posalfeat


if __name__ == '__main__':
    


    B, T, H, W = 2, 3, 64, 128
    xyz_grid = get_xyz_grid(H, W, lib='torch').float()  # xyz_grid.shape = [H, W, 3];
    salmap = torch.rand(B, T, 1, H, W)
    posalfeat = salmap2posalfeat(xyz_grid, salmap)
    print(posalfeat.shape)
    h, w = 32, 32
    print(salmap[0, 0, 0, h, w])
    print(posalfeat[0, 0, W*h+w, :3])
    print(posalfeat[0, 0, W*h+w, -1])
    exit()


    xyz_grid = get_xyz_grid(4, 8, lib='torch')
    print(xyz_grid.shape)
    print(xyz_grid[:, :, 2])
    exit()


    import math
    old_xyz = torch.tensor([[1, 0, 0]], dtype=torch.float32).permute(1, 0)

    fovxy = torch.tensor([[6.6667e-01, 2.9802e-08]], dtype=torch.float32)
    fov_deg = (90, 90)
    new_xyz = get_new_xyz(old_xyz, fovxy, fov_deg)
    print(new_xyz)

    fovxy2 = xyz2fovxy(old_xyz, new_xyz, fov_deg)
    print(fovxy)
    print(fovxy2)
    


    # B = 128
    # T = 25
    # C = 1
    # H_in = 216
    # W_in = 384
    # H_out = 64
    # W_out = 128

    # maxpool = MaxPool(kernel_size=(2,2), stride=(2,2))
    # x = torch.randn(B, T, C, H_in, W_in)
    # y = maxpool(x)
    # print(y.shape)

    # # test1
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

    # sal = torch.randn((2, 4, 1, 5, 8))
    # sal_clip = clip_sal(sal, pos)

    # print(sal[0][0][0])
    # print(sal_clip.shape)
    # print(sal_clip[0][0][0])


    # test2
    xyz = torch.tensor([[0, 0, 1], [0, -1, 0]], dtype=torch.float32)
    print(xyz.shape)

    image_path1 = './450000.jpg'
    image_path2 = './450000.jpg'
    img1 = np.array(Image.open(image_path1))
    img2 = np.array(Image.open(image_path2))
    img = np.stack([img1, img2], axis=0)
    img = torch.tensor(img, dtype=torch.float32)
    print(img.shape)

    fovs = get_fov(xyz, img, fov_deg=(90, 90), fov_shape=(224, 224))
    fovs = fovs.cpu().detach().numpy()

    for i in range(fovs.shape[0]):
        fov = fovs[i]
        fov = fov.astype(np.uint8)
        print(fov.max())
        print(fov.min())
        # print(type(fov))
        # print(fov.shape)
        fov = Image.fromarray(fov)
        fov.save(f'fov{i}.png')