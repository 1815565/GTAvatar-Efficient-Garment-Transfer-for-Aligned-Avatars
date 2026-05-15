import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.feature2d import grid_sample


class PoseVocab(nn.Module):
    def __init__(self,
                 joint_num,
                 point_num,
                 line_size,
                 feat_dim,
                 factor=4.0,
                 init = 'random'):
        """
        :param point_num: P, discrete pose point number
        :param line_size: (Lx, Ly, Lz), spacial resolutions
        :param feat_dim: C, feature channel number
        :param spacial_bounds: [min_xyz, max_xyz] spacial bounds along x, y, z axes
        """
        super(PoseVocab, self).__init__()


        self.J = joint_num
        self.P = point_num
        self.Lx, self.Ly, self.Lz = line_size
        self.C = feat_dim
        self.out_channels = self.J * self.C * 3

        self.feat_lines_x = nn.ParameterDict()
        self.feat_lines_y = nn.ParameterDict()
        self.feat_lines_z = nn.ParameterDict()
        for j in range(self.J):
            # Lx_j = int(self.Lx * (1.0 + (factor - 1)*(j/(self.J - 1))))  # 计算当前的Lx_j
            # print("Lx_j", Lx_j)
            Lx_j = self.Lx
            param = nn.Parameter(torch.zeros((self.P, Lx_j, Lx_j, self.C), dtype=torch.float32))
            if init == 'random':
                nn.init.uniform_(param.data, -1e-2, 1e-2)
            elif init == 'zeros':
                nn.init.constant_(param.data, 0.)
            else:
                raise ValueError('Invalid init method')
            self.feat_lines_x[str(j)] = param

            # Ly_j = int(self.Ly * (1.0 + (factor - 1)*(j/(self.J - 1))))  # 计算当前的Lx_j
            Ly_j = self.Ly
            param = nn.Parameter(torch.zeros((self.P, Ly_j, Ly_j, self.C), dtype=torch.float32))
            if init == 'random':
                nn.init.uniform_(param.data, -1e-2, 1e-2)
            elif init == 'zeros':
                nn.init.constant_(param.data, 0.)
            else:
                raise ValueError('Invalid init method')
            self.feat_lines_y[str(j)] = param

            # Lz_j = int(self.Lz * (1.0 + (factor - 1)*(j/(self.J - 1))))  # 计算当前的Lx_j
            Lz_j = self.Lz
            param = nn.Parameter(torch.zeros((self.P, Lz_j, Lz_j, self.C), dtype=torch.float32))
            if init == 'random':
                nn.init.uniform_(param.data, -1e-2, 1e-2)
            elif init == 'zeros':
                nn.init.constant_(param.data, 0.)
            else:
                raise ValueError('Invalid init method')
            self.feat_lines_z[str(j)] = param


        # pose_points = torch.from_numpy(np.load(pose_points_path)).to(torch.float32)
        # if used_point_num is not None:
        #     pose_points = pose_points[:, :used_point_num]
        # else:
        #     pose_points = pose_points[:, :point_num]
        # self.register_buffer('pose_points', pose_points)  # (J, P, 3) or (J, P, 4)

    def sample(self, feat_lines_x, x, id):
        """
        :param feat_lines_x: (J, P, Lx, C)
        :param x: (B, N, 1)

        :return: (B, N, J, C)
        """
        feat_list = []
        for _, param in feat_lines_x.items():
            # 獲取參數的數據，形狀為 (P, Lx_j, Lx_j, C)
            feat = param.data[[id]]
            feat = feat.permute(0, 3, 1, 2) 
            feat = grid_sample(feat, x)
            feat_list.append(feat)
        feat_lines = torch.cat(feat_list, dim=2)

        return feat_lines

    def forward(self, id, query_points, scale, skinning_weight=None):
        """
        :param query_points: (B, N, 3)
        :param query_poses: (B, J, 4)
        :return: feat: (B, N, J, 3*C)
        """

        # min_points = query_points.min(dim=1)[0]  # 形状为 [B, 3]
        # max_points = query_points.max(dim=1)[0]  # 形状为 [B, 3]
        # epsilon = 1e-8
        # range_points = max_points - min_points + epsilon
        # query_points = 2. * (query_points - min_points.unsqueeze(1)) / range_points.unsqueeze(1) - 1.

        query_points = query_points - torch.mean(query_points,0)[None,:]
        x = query_points[:,0] / (scale[0]/2)
        y = query_points[:,1] / (scale[1]/2)
        z = query_points[:,2] / (scale[2]/2)
        
        # extract features from the triplane
        yx, xz, zy = torch.stack((y,x),1), torch.stack((x,z),1), torch.stack((z,y),1)
        

        feat_x = self.sample(self.feat_lines_x,  yx[None,:,None,:], id)  # (B, N, J, C)
        feat_y = self.sample(self.feat_lines_y,  zy[None,:,None,:], id)
        feat_z = self.sample(self.feat_lines_z,  xz[None,:,None,:], id)

        if skinning_weight is None:
            feat = torch.cat([feat_x, feat_y, feat_z], dim = 3)
            # weight = F.softmax(self.coef, dim = 0)
            # print(weight)
            # feat = feat * weight[None, None, :, None]
            B, N, J, C = feat.shape
            feat = feat.reshape(B, N, J * C)
        else:
            feat = torch.einsum("bnjc, nj -> bnc", torch.cat([feat_x, feat_y, feat_z], dim = 3), skinning_weight)
        
        
        return feat, None
        

    @staticmethod
    def smooth_loss(feat_lines):
        """
        :param feat_lines: (B, J, K, Lx, C)
        :return:
        """
        tv = torch.square(feat_lines[..., 1:, 1:, :] - feat_lines[..., :-1, :-1, :]).mean()
        return tv
