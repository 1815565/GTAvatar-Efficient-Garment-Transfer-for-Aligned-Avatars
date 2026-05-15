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

        feat_lines_x = torch.zeros((self.J, self.P, self.Lx, self.Lx, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_x', nn.Parameter(feat_lines_x))
        feat_lines_y = torch.zeros((self.J, self.P, self.Ly, self.Ly, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_y', nn.Parameter(feat_lines_y))
        feat_lines_z = torch.zeros((self.J, self.P, self.Lz, self.Lz, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_z', nn.Parameter(feat_lines_z))
        # coef = torch.ones(self.J, dtype = torch.float32)
        # self.register_parameter('coef', nn.Parameter(coef))

        if init == 'random':
            nn.init.uniform_(self.feat_lines_x.data, -1e-2, 1e-2)
            nn.init.uniform_(self.feat_lines_y.data, -1e-2, 1e-2)
            nn.init.uniform_(self.feat_lines_z.data, -1e-2, 1e-2)
        elif init == 'zeros':
            nn.init.constant_(self.feat_lines_x.data, 0.)
            nn.init.constant_(self.feat_lines_y.data, 0.)
            nn.init.constant_(self.feat_lines_z.data, 0.)
        else:
            raise ValueError('Invalid init method')

        # pose_points = torch.from_numpy(np.load(pose_points_path)).to(torch.float32)
        # if used_point_num is not None:
        #     pose_points = pose_points[:, :used_point_num]
        # else:
        #     pose_points = pose_points[:, :point_num]
        # self.register_buffer('pose_points', pose_points)  # (J, P, 3) or (J, P, 4)

        self.pose_point_graph = None

    def sample(self, feat_lines_x, x, id):
        """
        :param feat_lines_x: (J, P, Lx, C)
        :param x: (B, N, 1)

        :return: (B, N, J, C)
        """
        J, P, Lx, W, C = feat_lines_x.shape
        B, N = x.shape[:2]


        feat_lines = feat_lines_x[:, [id], : , :, :].permute(1, 0, 2, 3, 4) # (1, J, Lx, 6, C)
        
        # smooth_loss = self.smooth_loss(feat_lines)
        
        feat_lines = feat_lines.permute(0, 1, 4, 2, 3).reshape(B, J * C, Lx, W)
        feat = grid_sample(feat_lines, x)
        feat = feat.view(B, N, J, C)
        return feat

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
            # print(feat.shape)
            B, N, J, C = feat.shape
            feat = feat.reshape(B, N, J * C)
        else:
            feat = torch.einsum("bnjc, nj -> bnc", torch.cat([feat_x, feat_y, feat_z], dim = 3), skinning_weight)
        
        # self.show()
        # exit()
        return feat
        

    @staticmethod
    def smooth_loss(feat_lines):
        """
        :param feat_lines: (B, J, K, Lx, C)
        :return:
        """
        tv = torch.square(feat_lines[..., 1:, 1:, :] - feat_lines[..., :-1, :-1, :]).mean()
        return tv

    def show(self):
        from PIL import Image
        tensor = self.feat_lines_x[:, 0, :, :, :].squeeze(0).reshape(128, 128, 11, 3).permute(2, 0, 1, 3) * 100
        tensor_normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 归一化到 0-1
        tensor_scaled = tensor_normalized * 255  # 转换到 0-255 范围，但仍然是浮点数
        tensor_scaled = tensor_scaled.clamp(0, 255)  # 将值限制在 0-255 范围内
        
        # 转换为 uint8 类型（注意：这可能会导致数据丢失，因为浮点数会被四舍五入到最接近的整数）
        tensor_np = tensor_scaled.cpu().numpy().astype(np.uint8)

        height, width = tensor_np.shape[1:3]
        total_height = height   # 拼接后的总高度
        total_width = width * 11        # 拼接后的总宽度（假设每张图像宽度相同）
        
        # 创建一个空白图像，背景为白色或其他颜色
        background_color = (255, 255, 255)  # 白色背景
        stitched_image = Image.new('RGB', (total_width, total_height), background_color)
        
        # 将每张图像粘贴到拼接图像中的正确位置
        for i in range(11):
            image = Image.fromarray(tensor_np[i])
            image = image.rotate(90, expand=True)
            stitched_image.paste(image, (i * width, 0))  # (x, y) 表示粘贴的起始位置
        
        # 指定保存路径并保存拼接后的图像
        save_path = '/home/ljr/Downloads/SelfAvatar_ori/show.jpg'
        stitched_image.save(save_path)





class PoseVocab_semantic(nn.Module):
    def __init__(self,
                 joint_num,
                 point_num,
                 line_size,
                 feat_dim,
                 init = 'random'):
        """
        :param point_num: P, discrete pose point number
        :param line_size: (Lx, Ly, Lz), spacial resolutions
        :param feat_dim: C, feature channel number
        :param spacial_bounds: [min_xyz, max_xyz] spacial bounds along x, y, z axes
        """
        super(PoseVocab_semantic, self).__init__()


        self.J = joint_num
        self.P = point_num
        self.Lx, self.Ly, self.Lz = line_size
        self.C = feat_dim

        feat_lines_x = torch.zeros((self.J, self.P, self.Lx, self.Lx, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_x', nn.Parameter(feat_lines_x))
        feat_lines_y = torch.zeros((self.J, self.P, self.Ly, self.Ly, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_y', nn.Parameter(feat_lines_y))
        feat_lines_z = torch.zeros((self.J, self.P, self.Lz, self.Lz, self.C-1), dtype = torch.float32)
        self.register_parameter('feat_lines_z', nn.Parameter(feat_lines_z))
        # coef = torch.ones(self.J, dtype = torch.float32)
        # self.register_parameter('coef', nn.Parameter(coef))

        if init == 'random':
            nn.init.uniform_(self.feat_lines_x.data, -1e-2, 1e-2)
            nn.init.uniform_(self.feat_lines_y.data, -1e-2, 1e-2)
            nn.init.uniform_(self.feat_lines_z.data, -1e-2, 1e-2)
        elif init == 'zeros':
            nn.init.constant_(self.feat_lines_x.data, 0.)
            nn.init.constant_(self.feat_lines_y.data, 0.)
            nn.init.constant_(self.feat_lines_z.data, 0.)
        else:
            raise ValueError('Invalid init method')

        # pose_points = torch.from_numpy(np.load(pose_points_path)).to(torch.float32)
        # if used_point_num is not None:
        #     pose_points = pose_points[:, :used_point_num]
        # else:
        #     pose_points = pose_points[:, :point_num]
        # self.register_buffer('pose_points', pose_points)  # (J, P, 3) or (J, P, 4)

        self.pose_point_graph = None

    def sample(self, feat_lines_x, x, id):
        """
        :param feat_lines_x: (J, P, Lx, C)
        :param x: (B, N, 1)

        :return: (B, N, J, C)
        """
        J, P, Lx, W, C = feat_lines_x.shape
        B, N = x.shape[:2]


        feat_lines = feat_lines_x[:, [id], : , :, :].permute(1, 0, 2, 3, 4) # (1, J, Lx, 6, C)
        
        # smooth_loss = self.smooth_loss(feat_lines)
        
        feat_lines = feat_lines.permute(0, 1, 4, 2, 3).reshape(B, J * C, Lx, W)
        feat = grid_sample(feat_lines, x)
        feat = feat.view(B, N, J, C)
        return feat

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
            # print(feat.shape)
            B, N, J, C = feat.shape
            feat = feat.reshape(B, N, J * C)
        else:
            feat = torch.einsum("bnjc, nj -> bnc", torch.cat([feat_x, feat_y, feat_z], dim = 3), skinning_weight)
        
        # self.show()
        # exit()
        return feat
        

    @staticmethod
    def smooth_loss(feat_lines):
        """
        :param feat_lines: (B, J, K, Lx, C)
        :return:
        """
        tv = torch.square(feat_lines[..., 1:, 1:, :] - feat_lines[..., :-1, :-1, :]).mean()
        return tv