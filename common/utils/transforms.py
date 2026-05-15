import torch
import numpy as np
import math

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z), 1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z), 1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def get_view_matrix(R, t):
    Rt = torch.cat((R, t.view(3,1)),1)
    view_matrix = torch.cat((Rt, torch.FloatTensor([0,0,0,1]).cuda().view(1,4)))
    return view_matrix

def get_proj_matrix(focal, princpt, img_shape, z_near, z_far, z_sign):
    fov = get_fov(focal, princpt, img_shape)
    fovY = fov[1]
    fovX = fov[0]
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right
    z_sign = 1.0

    proj_matrix = torch.zeros(4, 4).float().cuda()
    proj_matrix[0, 0] = 2.0 * z_near / (right - left)
    proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
    proj_matrix[0, 2] = (right + left) / (right - left)
    proj_matrix[1, 2] = (top + bottom) / (top - bottom)
    proj_matrix[3, 2] = z_sign
    proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near)
    proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return proj_matrix

def get_fov(focal, princpt, img_shape):
    fov_x = 2 * torch.atan(img_shape[1] / (2 * focal[0]))
    fov_y = 2 * torch.atan(img_shape[0] / (2 * focal[1]))
    fov = torch.FloatTensor([fov_x, fov_y]).cuda()
    return fov

def get_covariance_matrix(scale, rotation):
    point_num = scale.shape[0]
    S = torch.zeros((point_num, 3, 3)).float().cuda()
    S[:,0,0] = scale[:,0]
    S[:,1,1] = scale[:,1]
    S[:,2,2] = scale[:,2]
    RS = torch.bmm(rotation, S)
    covariance = torch.bmm(RS, RS.permute(0,2,1))
    return covariance

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def eval_sh_bases(deg, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    assert deg <= 4 and deg >= 0
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy;
            result[..., 5] = C2[1] * yz;
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = C2[3] * xz;
            result[..., 8] = C2[4] * (xx - yy);

            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy);
                result[..., 10] = C3[1] * xy * z;
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = C3[5] * z * (xx - yy);
                result[..., 15] = C3[6] * x * (xx - 3 * yy);

                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy);
                    result[..., 17] = C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result
    

import torch

def rodrigues_rotation_batch(r, theta):
    """
    批量化的 Rodrigues 旋转公式
    :param r: 旋转轴，形状为 [batch_size, 3]
    :param theta: 旋转角度，形状为 [batch_size]
    :return: 旋转矩阵，形状为 [batch_size, 3, 3]
    """
    # 确保输入是 torch.Tensor
    r = torch.as_tensor(r, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)

    # 将 theta 扩展为 [batch_size, 1, 1] 以便广播
    theta = theta.view(-1, 1, 1)

    # 构造反对称矩阵 M [batch_size, 3, 3]
    rx, ry, rz = r.unbind(dim=1)
    zeros = torch.zeros_like(rx)
    M = torch.stack([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=1).view(-1, 3, 3)

    # 计算旋转矩阵 R [batch_size, 3, 3]
    identity = torch.eye(3, device=r.device).unsqueeze(0)  # [1, 3, 3]
    r = r.unsqueeze(2)  # [batch_size, 3, 1]
    r_T = r.transpose(1, 2)  # [batch_size, 1, 3]

    R = torch.cos(theta) * identity + \
        (1 - torch.cos(theta)) * (r @ r_T) + \
        torch.sin(theta) * M

    return R

def calc_theta(a, b, device):
    a = a.to(device)
    b = b.to(device)
    dot_product = torch.sum(a * b, dim=1)
 
    # 计算模长
    norm_a = torch.norm(a, dim=1)
    norm_b = torch.norm(b, dim=1)
    
    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_a * norm_b)
    
    # 计算夹角（以弧度为单位）
    theta = torch.acos(cos_theta)
    
    return theta

def calc_rot_axis(a, b, device): 
    a = a.to(device)
    b = b.to(device)
    cross_product = torch.cross(a, b)  # 简化后的写法
    
    # 单位化叉乘结果
    norm_cross_product = torch.norm(cross_product, dim=1, keepdim=True)
    epsilon = 1e-8
    unit_cross_product = cross_product / torch.clamp(norm_cross_product, min=epsilon)
    
    # 打印结果
    return unit_cross_product

def calc_rot_matrix(normal):
    device = normal.device
    z_axis = torch.tensor([0.0, 0.0, 1.0]).to(device).unsqueeze(0).repeat(normal.shape[0], 1)
    theta = calc_theta(normal, z_axis, device)
    rot_axis = calc_rot_axis(normal, z_axis, device)
    rot_mat = rodrigues_rotation_batch(rot_axis, theta)
    return rot_mat

def batch_rodrigues(rot_vecs, epsilon = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
