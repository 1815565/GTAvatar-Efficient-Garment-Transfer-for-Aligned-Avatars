import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import eval_sh, RGB2SH, get_fov, get_view_matrix, get_proj_matrix, eval_sh_bases, calc_rot_matrix, batch_rodrigues
from utils.smpl_x import smpl_x
from utils.flame import flame
from smplx.lbs import batch_rigid_transform
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from nets.layer import make_linear_layers
from pytorch3d.structures import Meshes
from config import cfg
import copy
from simple_knn._C import distCUDA2

from nets.posevocab import PoseVocab, PoseVocab_semantic
from nets.net_util import PositionalEncoding
from nets.CrossAttn import MultiHeadCrossAttention
import numpy as np
import os
from nets.gaussian_model import GaussianModel
from nets.loss import RGBLoss, SSIM, LPIPS, LaplacianReg, LayeredConstraintLoss, HandMeanReg, HandRGBReg, ArmRGBReg, SemanticLoss, TVLoss, ColorConsistencyLoss
import trimesh
from obj_io import save_gaussians_as_ply, SH2RGB


class HumanGaussian(nn.Module):
    def __init__(self, num_subjects):
        super(HumanGaussian, self).__init__()
        # self.triplane = nn.Parameter(torch.zeros(
        #     (3, *cfg.triplane_shape)).float().cuda())
        # self.triplane_face = nn.Parameter(torch.zeros(
        #     (3, *cfg.triplane_shape)).float().cuda())
        # self.xyfilter = FuseHGFilter(cfg, 2, cfg.triplane_shape[0], 2*cfg.triplane_shape[0])
        # self.xzfilter = FuseHGFilter(cfg, 2, cfg.triplane_shape[0], 2*cfg.triplane_shape[0])
        # self.yzfilter = FuseHGFilter(cfg, 2, cfg.triplane_shape[0], 2*cfg.triplane_shape[0])

        self.vocab = PoseVocab(11, num_subjects, cfg.line_size, 3)
        self.vocab_face = PoseVocab(11, num_subjects, cfg.line_size, 3)
        # self.semantic_code = PoseVocab_semantic(1, num_subjects, cfg.line_size, 2)
        self.cross_attention_layer = MultiHeadCrossAttention(3, 3, 16, 1)
        self.cross_attention_layer_self = MultiHeadCrossAttention(3, 3, 3, 1)

        # self.id_embed = nn.Embedding(num_subjects, 32)

        # self.embedder = PositionalEncoding(num_freqs=5)
        # self.embedder.create_embedding_fn()
        # in_dim = self.embedder.out_dim + 96
        

        self.geo_encoder = make_linear_layers(
            [99, 128, 128, 128], use_gn=True)  # cfg.triplane_shape[0]*3,
        self.surface_net = make_linear_layers([128, 6], relu_final=False)
        # self.radius_net = make_linear_layers([128, 3], relu_final=False)
        self.geo_pose_encoder = make_linear_layers(
            [99+127, 128, 128, 128], use_gn=True)  # (len(smpl_x.joint_part['body'])-1)
        self.surface_offset_net = make_linear_layers(
            [128, 6], relu_final=False)
        # self.radius_offset_net = make_linear_layers([128, 3], relu_final=False)
        self.appearance_net = make_linear_layers(
            [99, 128, 128, 128, 6], relu_final=False, use_gn=True)

        # d_in = (self.sh_degree + 1) ** 2 - 1
        self.sh_embed = lambda dir: eval_sh_bases(3, dir)[..., 1:]
        self.appearance_offset_net = make_linear_layers(
            [99+127+15, 128, 128, 128, 6], relu_final=False, use_gn=True)
        
        # self.semantic_net = make_linear_layers([99, 128, 5], relu_final=False)

        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender]).cuda()
        shape_param = torch.stack(smpl_x.shape_param, dim=0)
        joint_offset = torch.stack(smpl_x.joint_offset, dim=0)
        self.shape_param = nn.Parameter(shape_param.float().cuda())
        self.joint_offset = nn.Parameter(joint_offset.float().cuda())
        self.semantic_code = nn.Parameter(RGB2SH(torch.rand(num_subjects, smpl_x.vertex_num_upsampled, 5)).float().cuda())

        self.num_subjects = num_subjects


    def get_optimizable_params(self):
        optimizable_params = [
            # {'params': [self.triplane],
            #     'name': 'triplane_human', 'lr': cfg.lr},
            # {'params': [self.triplane_face],
            #     'name': 'triplane_face_human', 'lr': cfg.lr},
            {'params': list(self.vocab.parameters()),
             'name': 'vocab_human', 'lr': cfg.lr},
            {'params': list(self.vocab_face.parameters()),
             'name': 'vocab_human', 'lr': cfg.lr},

            {'params': list(self.cross_attention_layer.parameters()),
             'name': 'xyfilter_human', 'lr': cfg.lr},
            {'params': list(self.cross_attention_layer_self.parameters()),
             'name': 'xzfilter_human', 'lr': cfg.lr},

            {'params': list(self.geo_encoder.parameters()),
             'name': 'geo_net_human', 'lr': cfg.lr},
            {'params': list(self.surface_net.parameters()),
             'name': 'mean_offset_net_human', 'lr': cfg.lr},

            {'params': list(self.geo_pose_encoder.parameters()),
             'name': 'geo_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.surface_offset_net.parameters()),
             'name': 'mean_offset_offset_net_human', 'lr': cfg.lr},

            {'params': list(self.appearance_net.parameters()),
             'name': 'rgb_net_human', 'lr': cfg.lr},
            {'params': list(self.appearance_offset_net.parameters()),
             'name': 'rgb_offset_net_human', 'lr': cfg.lr},
            {'params': [self.shape_param],
                'name': 'shape_param_human', 'lr': cfg.lr},
            {'params': [self.joint_offset],
                'name': 'joint_offset_human', 'lr': cfg.lr},
            {'params': [self.semantic_code],
                'name': 'semantic_human', 'lr': cfg.lr},    
            # {'params': list(self.semantic_code.parameters()),
            #  'name': 'semantic_net_human', 'lr': cfg.lr},
        ]
        return optimizable_params
    
    def init_(self):
        mean_3d_list, radius_list = [], []
        rgb_list, rot_list = [], []
        joint_zero_pose_list, nn_vertex_idxs_list, transform_mat_neutral_pose_list = [], [], []

        for id in range(self.num_subjects):
            mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(
                jaw_zero_pose=True, use_id_info=True, id=id)
            joint_zero_pose = self.get_zero_pose_human(id)

            mesh = trimesh.Trimesh(vertices=mesh_neutral_pose_wo_upsample.detach().cpu().numpy(), faces=smpl_x.face)
            mesh.export(f"/home/ljr/Downloads/Cloth_avatar/assets/smplx{id}.ply", file_type='ply')

            tri_feat = self.extract_tri_feat(id)

            geo_feat = self.geo_encoder(tri_feat)
            mean_offset, radius = torch.split(self.surface_net(
                geo_feat), [3, 3], dim=1)  # mean offset of Gaussians
            appearance = self.appearance_net(tri_feat)  # rgb of Gaussians

            radius = torch.sigmoid(radius)
            rgb, rot = torch.split(
            torch.sigmoid(appearance), [3, 3], dim=-1)

            mean_3d = mesh_neutral_pose + mean_offset  # 大 pose
            radius = (radius - 0.5) * 2
            rot = (rot -0.5) * np.pi

            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.matmul(R_delta, self.init_rot)

            dist2 = torch.clamp_min(distCUDA2(mean_3d), 0.0000001)
            scales = torch.sqrt(dist2)[..., None].repeat(1, 3).detach()
            scale = (radius+1)*scales
            
            vmin = mesh_neutral_pose_wo_upsample.min(dim=0).values
            vmax = mesh_neutral_pose_wo_upsample.max(dim=0).values
            ori_center = (vmax + vmin) / 2.
            print(ori_center)
        
            gaissian_vals = {}
            gaissian_vals['mean_3d'] = mean_3d - ori_center
            gaissian_vals['opacity'] = torch.ones((smpl_x.vertex_num_upsampled, 1)).float().cuda()
            gaissian_vals['scale'] = scale
            gaissian_vals['rotation'] = matrix_to_quaternion(R)
            gaissian_vals['rgb'] = rgb
            
            save_gaussians_as_ply(f'/home/ljr/Downloads/Cloth_avatar/assets/gauss{str(id)}.ply', gaissian_vals)
            np.save(f'/home/ljr/Downloads/Cloth_avatar/assets/semantic{str(id)}.npy', torch.argmax(torch.softmax(self.semantic_code[id], dim=1), dim=1).cpu().numpy())
            
            nn_vertex_idxs = knn_points(mean_3d[None, :, :], mesh_neutral_pose_wo_upsample[None, :, :],
                                    K=1, return_nn=True).idx[0, :, 0]  # dimension: smpl_x.vertex_num_upsampled
            nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
            mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
            nn_vertex_idxs[mask] = torch.arange(
            smpl_x.vertex_num_upsampled).cuda()[mask]

            mean_3d_list.append(mean_3d)
            radius_list.append(radius)
            rgb_list.append(rgb)
            rot_list.append(R)
            joint_zero_pose_list.append(joint_zero_pose)
            nn_vertex_idxs_list.append(nn_vertex_idxs)
            transform_mat_neutral_pose_list.append(transform_mat_neutral_pose)
        exit()
        self.gauss_dict = {'mean_3d' : torch.stack(mean_3d_list, dim=0), 'radius': torch.stack(radius_list, dim=0), \
            'rgb': torch.stack(rgb_list, dim=0), 'rotation': torch.stack(rot_list, dim=0), 'opacity': torch.ones((smpl_x.vertex_num_upsampled, 1)).float().cuda(),\
                'nn_vertex_idxs': torch.stack(nn_vertex_idxs_list, dim=0), 'joint_zero_pose': torch.stack(joint_zero_pose_list, dim=0),\
                    'transform_mat_neutral_pose': torch.stack(transform_mat_neutral_pose_list, dim=0)}
    
    def quick_deform(self, id, smplx_param, cam_param, is_world_coord=False):
        mean_3d = self.gauss_dict['mean_3d'][id]
        radius = self.gauss_dict['radius'][id]
        rgb = self.gauss_dict['rgb'][id]
        R = self.gauss_dict['rotation'][id]
        opacity = self.gauss_dict['opacity'][id]
        nn_vertex_idxs = self.gauss_dict['nn_vertex_idxs'][id]
        joint_zero_pose = self.gauss_dict['joint_zero_pose'][id]
        transform_mat_neutral_pose = self.gauss_dict['transform_mat_neutral_pose'][id]
        transform_mat_joint = self.get_transform_mat_joint(
            transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, nn_vertex_idxs)
        mean_3d = self.lbs(mean_3d, transform_mat_vertex,
                           smplx_param['trans'])
        if not is_world_coord:
            mean_3d = torch.matmul(torch.inverse(
                cam_param['R']), (mean_3d - cam_param['t'].view(1, 3)).permute(1, 0)).permute(1, 0)

        R_def = torch.matmul(transform_mat_vertex[:, :3, :3], R)
        rotation = matrix_to_quaternion(R_def)

        dist2 = torch.clamp_min(distCUDA2(mean_3d), 0.0000001)
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3).detach()
        scale = (radius+1)*scales

        assets = {
            'mean_3d': mean_3d,
            'opacity': opacity,
            'radius': scale,
            'rotation': rotation,
            'rgb': rgb,
            'semantic_code': self.semantic_code[id],
        }
        return assets

    def init(self):
        # upsample mesh and other assets
        xyz, _, _, _ = self.get_neutral_pose_human(
            jaw_zero_pose=False, use_id_info=False)  # torch.Size([167333, 3])
        # normal = self.get_normal(xyz)
        # color = vertex_normal_2_vertex_color(normal)
        # customized_export_ply("normal_temp.ply", v = xyz, f = smpl_x.face_upsampled, v_c = color)
        # exit()
        skinning_weight = self.smplx_layer.lbs_weights.float()
        pose_dirs = self.smplx_layer.posedirs.permute(1, 0).reshape(
            smpl_x.vertex_num, 3*(smpl_x.joint_num-1)*9)
        expr_dirs = self.smplx_layer.expr_dirs.view(
            smpl_x.vertex_num, 3*smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face, is_face_expr, is_scalp, is_neck = torch.zeros((smpl_x.vertex_num, 1)).float().cuda(), torch.zeros((smpl_x.vertex_num, 1)).float(
        ).cuda(), torch.zeros((smpl_x.vertex_num, 1)).float().cuda(), torch.zeros((smpl_x.vertex_num, 1)).float().cuda(), torch.zeros((smpl_x.vertex_num, 1)).float().cuda(), torch.zeros((smpl_x.vertex_num, 1)).float().cuda()
        is_rhand[smpl_x.rhand_vertex_idx], is_lhand[smpl_x.lhand_vertex_idx], is_face[
            smpl_x.face_vertex_idx], is_face_expr[smpl_x.expr_vertex_idx] = 1.0, 1.0, 1.0, 1.0
        is_scalp[smpl_x.scalp_vertex_idx] = 1.0
        is_neck[smpl_x.neck_vertex_idx] = 1.0
        # is_cavity = torch.FloatTensor(smpl_x.is_cavity).cuda()[:,None]
        # uv = torch.FloatTensor(smpl_x.uv).cuda()

        _, (skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_scalp, is_neck) = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num, 3)
                                                                                                                                ).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_scalp, is_neck])  # upsample with dummy vertex

        pose_dirs = pose_dirs.reshape(
            smpl_x.vertex_num_upsampled*3, (smpl_x.joint_num-1)*9).permute(1, 0)
        expr_dirs = expr_dirs.view(
            smpl_x.vertex_num_upsampled, 3, smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face, is_face_expr = is_rhand[:, 0] > 0, is_lhand[:, 0] > 0, is_face[:, 0] > 0, is_face_expr[:, 0] > 0
        is_scalp = is_scalp[:, 0] > 0
        is_neck = is_neck[:, 0] > 0
        # print(torch.sum(is_scalp).item())
        # is_cavity = is_cavity[:,0] > 0

        is_body, is_larm, is_rarm, is_lleg, is_rleg = smpl_x.divide_parts(
            skinning_weight)
        # blend_weight = smpl_x.blend_weight(skinning_weight)
        # blend_weight = F.softmax(skinning_weight[:, :16], dim=1)
        # self.register_buffer('blend_weight', blend_weight)

        # init_rot = self.compute_rot_init(xyz.cpu(), smpl_x.face_upsampled, smpl_x.uv_upsampled, smpl_x.uv_face_upsampled) #(([334528, 3, 3]))
        # print(init_rot.shape)
        # self.register_buffer('init_rot', init_rot)

        init_normal = self.get_normal(xyz)
        init_rot = calc_rot_matrix(init_normal)  # torch.Size([167333, 3, 3])
        self.register_buffer('init_rot', init_rot)

        self.register_buffer('pos_enc_mesh', xyz)
        # print(xyz.shape, smpl_x.face_upsampled.shape)
        # exit()
        # self.register_buffer('pos_enc_mesh', xyz.unsqueeze(0))
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('pose_dirs', pose_dirs)
        self.register_buffer('expr_dirs', expr_dirs)
        self.register_buffer('is_rhand', is_rhand)
        self.register_buffer('is_lhand', is_lhand)
        self.register_buffer('is_face', is_face)
        self.register_buffer('is_face_expr', is_face_expr)
        self.register_buffer('is_scalp', is_scalp)
        self.register_buffer('is_neck', is_neck)
        # self.register_buffer('is_cavity', is_cavity)

        self.register_buffer('is_body', is_body)
        self.register_buffer('is_larm', is_larm)
        self.register_buffer('is_rarm', is_rarm)
        self.register_buffer('is_lleg', is_lleg)
        self.register_buffer('is_rleg', is_rleg)

    
    def compute_rot_init(self, v, vi, vt, vti):
        vt = torch.as_tensor(vt)

        v = torch.as_tensor(v).unsqueeze(0)
        vi = torch.tensor(vi).long()
        vti = torch.tensor(vti).long()

        v0 = v[:, vi[:, 0], :].unsqueeze(2)
        v1 = v[:, vi[:, 1], :].unsqueeze(2)
        v2 = v[:, vi[:, 2], :].unsqueeze(2)

        vt0 = vt[vti[:, 0], :].unsqueeze(1)
        vt1 = vt[vti[:, 1], :].unsqueeze(1)
        vt2 = vt[vti[:, 2], :].unsqueeze(1)

        primrotmesh = smpl_x.compute_tbn(
            v0, v1, v2, vt0, vt1, vt2).view(v0.size(0), -1, 3, 3)
        init_rot = primrotmesh[0]
        return init_rot

    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info, id=None):
        zero_pose = torch.zeros((1, 3)).float().cuda()
        neutral_body_pose = smpl_x.neutral_body_pose.view(
            1, -1).cuda()  # 大 pose
        zero_hand_pose = torch.zeros(
            (1, len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1, smpl_x.expr_param_dim)).float().cuda()
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1, 3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1, 3).cuda()  # open mouth
        if use_id_info:
            shape_param = self.shape_param[id][None, :]
            face_offset = smpl_x.face_offset[id][None, :, :].float().cuda()
            joint_offset = smpl_x.get_joint_offset(
                self.joint_offset[id][None, :, :])
        else:
            shape_param = torch.zeros(
                (1, smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None
        output = self.smplx_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose,
                                  leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)

        mesh_neutral_pose = output.vertices[0]  # 大 pose human
        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh(
            mesh_neutral_pose)  # 大 pose human
        # 大 pose human
        joint_neutral_pose = output.joints[0][:smpl_x.joint_num, :]

        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(
            len(smpl_x.joint_part['body'])-1, 3)
        zero_hand_pose = zero_hand_pose.view(
            len(smpl_x.joint_part['lhand']), 3)
        neutral_body_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(neutral_body_pose)))
        jaw_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(jaw_pose)))
        pose = torch.cat((zero_pose, neutral_body_pose_inv, jaw_pose_inv,
                         zero_pose, zero_pose, zero_hand_pose, zero_hand_pose))
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[None, :, :, :], joint_neutral_pose[None, :, :], self.smplx_layer.parents)
        transform_mat_neutral_pose = transform_mat_neutral_pose[0]
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose

    def get_zero_pose_human(self, id, return_mesh=False):
        zero_pose = torch.zeros((1, 3)).float().cuda()
        zero_body_pose = torch.zeros(
            (1, (len(smpl_x.joint_part['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros(
            (1, len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1, smpl_x.expr_param_dim)).float().cuda()
        shape_param = self.shape_param[id][None, :]
        face_offset = smpl_x.face_offset[id][None, :, :].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[id][None, :, :])
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose,
                                  leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)

        # zero pose human
        joint_zero_pose = output.joints[0][:smpl_x.joint_num, :]
        if not return_mesh:
            return joint_zero_pose
        else:
            mesh_zero_pose = output.vertices[0]  # zero pose human
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(
                mesh_zero_pose)  # zero pose human
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

    def get_transform_mat_joint(self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = smplx_param['root_pose'].view(1, 3)
        body_pose = smplx_param['body_pose'].view(
            len(smpl_x.joint_part['body'])-1, 3)
        jaw_pose = smplx_param['jaw_pose'].view(1, 3)
        leye_pose = smplx_param['leye_pose'].view(1, 3)
        reye_pose = smplx_param['reye_pose'].view(1, 3)
        lhand_pose = smplx_param['lhand_pose'].view(
            len(smpl_x.joint_part['lhand']), 3)
        rhand_pose = smplx_param['rhand_pose'].view(
            len(smpl_x.joint_part['rhand']), 3)
        trans = smplx_param['trans'].view(1, 3)

        # forward kinematics
        pose = torch.cat((root_pose, body_pose, jaw_pose,
                         leye_pose, reye_pose, lhand_pose, rhand_pose))
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(
            pose[None, :, :, :], joint_zero_pose[None, :, :], self.smplx_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]

        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        transform_mat_joint = torch.bmm(
            transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint

    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        skinning_weight = self.skinning_weight[nn_vertex_idxs, :]
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(
            smpl_x.joint_num, 16)).view(smpl_x.vertex_num_upsampled, 4, 4)
        return transform_mat_vertex

    def lbs(self, xyz, transform_mat_vertex, trans):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1)  # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:, :, None]).view(
            smpl_x.vertex_num_upsampled, 4)[:, :3]
        xyz = xyz + trans
        return xyz

    def extract_tri_feat(self, id):
        # 1. triplane features of all vertices
        # normalize coordinates to [-1,1]

        tri_feat = self.vocab(
            id, self.pos_enc_mesh, cfg.triplane_shape_3d, None)
        tri_feat = tri_feat[0]

        tri_feat_face = self.vocab_face(
            id, self.pos_enc_mesh[self.is_face, :], cfg.triplane_face_shape_3d, None)  # self.blend_weight[self.is_face, :]
        tri_feat_face = tri_feat_face[0]

        tri_feat[self.is_face] = tri_feat_face


        return tri_feat


    def extract_tri_feature(self, id):
        # 1. triplane features of all vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh
        xyz = xyz - torch.mean(xyz, 0)[None, :]
        x = xyz[:, 0] / (cfg.triplane_shape_3d[0]/2)
        y = xyz[:, 1] / (cfg.triplane_shape_3d[1]/2)
        z = xyz[:, 2] / (cfg.triplane_shape_3d[2]/2)

        # extract features from the triplane
        yx, xz, zy = torch.stack((y, x), 1), torch.stack(
            (x, z), 1), torch.stack((z, y), 1)
        feat_xy = F.grid_sample(self.triplane[id, 0, None, :, :, :], yx[None, :, None, :])[
            0, :, :, 0]  # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane[id, 1, None, :, :, :], xz[None, :, None, :])[
            0, :, :, 0]  # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane[id, 2, None, :, :, :], zy[None, :, None, :])[
            0, :, :, 0]  # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_yz, feat_xz)).permute(
            1, 0)  # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3

        # 2. triplane features of face vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh[self.is_face, :]
        xyz = xyz - torch.mean(xyz, 0)[None, :]
        x = xyz[:, 0] / (cfg.triplane_face_shape_3d[0]/2)
        y = xyz[:, 1] / (cfg.triplane_face_shape_3d[1]/2)
        z = xyz[:, 2] / (cfg.triplane_face_shape_3d[2]/2)

        # extract features from the triplane
        yx, xz, zy = torch.stack((y, x), 1), torch.stack(
            (x, z), 1), torch.stack((z, y), 1)
        feat_xy = F.grid_sample(self.triplane_face[id, 0, None, :, :, :], yx[None, :, None, :])[
            0, :, :, 0]  # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane_face[id, 1, None, :, :, :], xz[None, :, None, :])[
            0, :, :, 0]  # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane_face[id, 2, None, :, :, :], zy[None, :, None, :])[
            0, :, :, 0]  # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat_face = torch.cat((feat_xy, feat_yz, feat_xz)).permute(
            1, 0)  # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3

        # combine 1 and 2
        tri_feat[self.is_face] = tri_feat_face
        return tri_feat

    def cross_pose_feat(self, tri_feat, smplx_param):
        # pose from smplx parameters (only use body pose as face/hand poses are not diverse in the training set)
        tri_feat = torch.cat((tri_feat, torch.zeros(
            tri_feat.size(0), 64).to(tri_feat)), dim=-1)

        body_pose = smplx_param['body_pose'].view(
            len(smpl_x.joint_part['body'])-1, 3).unsqueeze(0).detach()

        upper_pose = body_pose[:, [2, 5, 8, 11], :]
        lhand_pose = body_pose[:, [12, 15, 17, 19], :]
        rhand_pose = body_pose[:, [13, 16, 18, 20], :]
        lleg_pose = body_pose[:, [0, 3, 6, 9], :]
        rleg_pose = body_pose[:, [1, 4, 7, 10], :]

        query_pose = torch.cat(
            (upper_pose, lhand_pose, rhand_pose, lleg_pose, rleg_pose))
        key_pose = value_pose = body_pose
        out = self.cross_attention_layer(
            query_pose, key_pose, value_pose, None).reshape(5, -1)

        out_self = self.cross_attention_layer_self(
            body_pose, body_pose, body_pose, None).reshape(1, -1)

        tri_feat[self.is_body][:, -64:] = out[0]
        tri_feat[self.is_larm][:, -64:] = out[1]
        tri_feat[self.is_rarm][:, -64:] = out[2]
        tri_feat[self.is_lleg][:, -64:] = out[3]
        tri_feat[self.is_rleg][:, -64:] = out[4]
        tri_feat = torch.cat(
            (tri_feat, out_self.repeat(smpl_x.vertex_num_upsampled, 1)), 1)
        return tri_feat

    def forward_geo_network(self, tri_feat, smplx_param):
        # pose from smplx parameters (only use body pose as face/hand poses are not diverse in the training set)
        # body_pose = smplx_param['body_pose'].view(
        #     len(smpl_x.joint_part['body'])-1, 3)

        # combine pose with triplane feature
        # pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).view(1, (len(
        #     smpl_x.joint_part['body'])-1)*6).repeat(smpl_x.vertex_num_upsampled, 1)  # without root pose
        
        # feat = torch.cat((tri_feat, pose.detach()),1)
        # feat = torch.cat((feat, pose.detach()), 1)
        feat = self.cross_pose_feat(tri_feat, smplx_param)

        # forward to geometry networks
        geo_offset_feat = self.geo_pose_encoder(feat)
        mean_offset_offset, radius_offset = torch.split(self.surface_offset_net(
            geo_offset_feat), [3, 3], dim=1)  # pose-dependent mean offset of Gaussians
        # radius_offset = self.radius_offset_net(
        #     geo_offset_feat)  # pose-dependent scale of Gaussians
        return mean_offset_offset, radius_offset

    def get_mean_offset_offset(self, smplx_param, mean_offset_offset):
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(
            len(smpl_x.joint_part['body'])-1, 3)
        jaw_pose = smplx_param['jaw_pose'].view(1, 3)
        leye_pose = smplx_param['leye_pose'].view(1, 3)
        reye_pose = smplx_param['reye_pose'].view(1, 3)
        lhand_pose = smplx_param['lhand_pose'].view(
            len(smpl_x.joint_part['lhand']), 3)
        rhand_pose = smplx_param['rhand_pose'].view(
            len(smpl_x.joint_part['rhand']), 3)
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose,
                         lhand_pose, rhand_pose))  # without root pose

        # smplx pose-dependent vertex offset
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)
                [None, :, :].float().cuda()).view(1, (smpl_x.joint_num-1)*9)
        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(
            smpl_x.vertex_num_upsampled, 3)

        # combine it with regressed mean_offset_offset
        # for face and hands, use smplx offset
        mask = ((self.is_rhand + self.is_lhand +
                self.is_face_expr) > 0)[:, None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        smplx_pose_offset = smplx_pose_offset * mask
        output = mean_offset_offset + smplx_pose_offset
        return output, mean_offset_offset
    
    def get_smplx_outputs(self, id, smplx_param, cam_param):
        root_pose = smplx_param['root_pose'].view(1, 3)
        body_pose = smplx_param['body_pose'].view(
            len(smpl_x.joint_part['body'])-1, 3)
        jaw_pose = smplx_param['jaw_pose'].view(1, 3)
        leye_pose = smplx_param['leye_pose'].view(1, 3)
        reye_pose = smplx_param['reye_pose'].view(1, 3)
        lhand_pose = smplx_param['lhand_pose'].view(
            len(smpl_x.joint_part['lhand']), 3)
        rhand_pose = smplx_param['rhand_pose'].view(
            len(smpl_x.joint_part['rhand']), 3)
        trans = smplx_param['trans'].view(1, 3)

        shape_param = self.shape_param[id][None, :]
        face_offset = smpl_x.face_offset[id][None, :, :].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[id][None, :, :])
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, jaw_pose=jaw_pose,
                                  leye_pose=leye_pose, reye_pose=reye_pose, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset, trans=trans)
        posed_verts = output.vertices[0]

        

    def forward_rgb_network(self, tri_feat, smplx_param, cam_param, normal_code):
        # pose from smplx parameters (only use body pose as face/hand poses are not diverse in the training set)
        # body_pose = smplx_param['body_pose'].view(
        #     len(smpl_x.joint_part['body'])-1, 3)
        # pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).view(1, (len(
        #     smpl_x.joint_part['body'])-1)*6).repeat(smpl_x.vertex_num_upsampled, 1)  # without root pose
        feat = self.cross_pose_feat(tri_feat, smplx_param)

        # feat = torch.cat((feat, normal_code), 1)

        # per-vertex normal in world coordinate system
        # with torch.no_grad():
        #     normal = Meshes(verts=xyz[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled,3)
        #     is_cavity = self.is_cavity[:,None].float()
        #     normal = normal * (1 - is_cavity) + (-normal) * is_cavity # cavity has opposite normal direction in the template mesh

        # forward to rgb network
        # feat = torch.cat((tri_feat, pose.detach(), normal_code), 1)
        feat = torch.cat((feat, normal_code), 1)
        # pose-and-view-dependent rgb offset of Gaussians
        appearance_offset = self.appearance_offset_net(feat)
        return appearance_offset

    def get_normal(self, xyz):
        # per-vertex normal in world coordinate system
        with torch.no_grad():
            normal = Meshes(verts=xyz[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[
                            None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled, 3)
            # is_cavity = self.is_cavity[:,None].float()
            # normal = normal * (1 - is_cavity) + (-normal) * is_cavity # cavity has opposite normal direction in the template mesh
            # 处理范数为零的情况
            normal = F.normalize(normal, p=2, dim=1, eps=1e-8)
            # norms = torch.norm(normal, p=2, dim=1, keepdim=True)
            # norms = torch.where(norms == 0, torch.ones_like(norms), norms)  # 将范数为零的替换为 1
            # normal = normal / norms
        return normal.detach()

    def get_normal_grad(self, xyz):
        # per-vertex normal in world coordinate system
        
        normal = Meshes(verts=xyz[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[
                        None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled, 3)
        # is_cavity = self.is_cavity[:,None].float()
        # normal = normal * (1 - is_cavity) + (-normal) * is_cavity # cavity has opposite normal direction in the template mesh
        norms = torch.norm(normal, p=2, dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)  # 将范数为零的替换为 1
        normal = normal / norms
        normal = (normal + 1.)/2.
        return normal
    
    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx

    def forward(self, id, smplx_param, cam_param, is_world_coord=False):
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(
            jaw_zero_pose=True, use_id_info=True, id=id)
        joint_zero_pose = self.get_zero_pose_human(id)

        # mesh = trimesh.Trimesh(vertices=mesh_neutral_pose_wo_upsample.cpu(), faces=smpl_x.face)
        # mesh.export("./cit_smplx.ply", file_type='ply')
        # exit()


        tri_feat = self.extract_tri_feat(id)

        geo_feat = self.geo_encoder(tri_feat)
        mean_offset, radius = torch.split(self.surface_net(
            geo_feat), [3, 3], dim=1)  # mean offset of Gaussians
        # radius = self.radius_net(geo_feat)  # scale of Gaussians
        appearance = self.appearance_net(tri_feat)  # rgb of Gaussians
        mean_3d = mesh_neutral_pose + mean_offset  # 大 pose

        mean_offset_offset, radius_offset = self.forward_geo_network(
            tri_feat, smplx_param)

        radius, radius_refined = torch.sigmoid(
            radius), torch.sigmoid(radius+radius_offset)
        
        
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(
            smplx_param, mean_offset_offset)
        mean_3d_refined = mean_3d + mean_combined_offset  # 大 pose

        # smplx facial expression offset
        smplx_expr_offset = (
            smplx_param['expr'][None, None, :] * self.expr_dirs).sum(2)
        mean_3d = mean_3d + smplx_expr_offset  # 大 pose
        mean_3d_refined = mean_3d_refined + smplx_expr_offset  # 大 pose

        # get nearest vertex
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(mean_3d[None, :, :], mesh_neutral_pose_wo_upsample[None, :, :],
                                    K=1, return_nn=True).idx[0, :, 0]  # dimension: smpl_x.vertex_num_upsampled
        nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
        mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
        nn_vertex_idxs[mask] = torch.arange(
            smpl_x.vertex_num_upsampled).cuda()[mask]

        # get transformation matrix of the nearest vertex and perform lbs
        transform_mat_joint = self.get_transform_mat_joint(
            transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, nn_vertex_idxs)
        mean_3d = self.lbs(mean_3d, transform_mat_vertex,
                           smplx_param['trans'])  # posed with smplx_param
        mean_3d_refined = self.lbs(
            mean_3d_refined, transform_mat_vertex, smplx_param['trans'])  # posed with smplx_param
        # neutral_mesh = self.lbs(self.pos_enc_mesh[0]+mean_offset+mean_combined_offset, transform_mat_vertex, smplx_param['trans'])
        # camera coordinate system -> world coordinate system
        if not is_world_coord:
            mean_3d = torch.matmul(torch.inverse(
                cam_param['R']), (mean_3d - cam_param['t'].view(1, 3)).permute(1, 0)).permute(1, 0)
            mean_3d_refined = torch.matmul(torch.inverse(
                cam_param['R']), (mean_3d_refined - cam_param['t'].view(1, 3)).permute(1, 0)).permute(1, 0)
            # neutral_mesh = torch.matmul(torch.inverse(cam_param['R']), (neutral_mesh - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)



        # normal = self.get_normal_grad(mean_3d)
        normal_refined = self.get_normal(mean_3d_refined)
        normal_enc = self.sh_embed(normal_refined)

        # forward to rgb network
        appearance_offset = self.forward_rgb_network(
            tri_feat, smplx_param, cam_param, normal_enc)
        rgb, rot = torch.split(
            torch.sigmoid(appearance), [3, 3], dim=-1)
        
        # # rgb[self.is_rhand + self.is_lhand] = torch.mean(rgb[self.is_rhand + self.is_lhand], dim=0)
        # lower_green = torch.tensor([0.0, 0.50, 0.0])  # 下限
        # upper_green = torch.tensor([0.40, 1.0, 0.40])  # 上限
        # green_mask = (rgb[:, 1] >= lower_green[1]) & (rgb[:, 1] <= upper_green[1]) & \
        #      (rgb[:, 0] <= upper_green[0]) & (rgb[:, 2] <= upper_green[2])
        # print(green_mask.sum())
        # # green_mask = (rgb[:, 1] > 0.4) & (rgb[:, 0] < 0.3) & (rgb[:, 2] < 0.3)
        # hand_mask = self.is_rhand | self.is_lhand
        # hand_green_mask = green_mask & hand_mask
        # normal_hand_pixels = rgb[hand_mask & ~hand_green_mask]
        # normal_color = torch.mean(normal_hand_pixels, dim=0, keepdim=True)
        # rgb[hand_green_mask] = normal_color

        rgb_refined, rot_refined = torch.split(
            torch.sigmoid(appearance + appearance_offset), [3, 3], dim=-1)
        radius, radius_refined = (radius - 0.5) * 2, (radius_refined - 0.5) * 2
        rot, rot_refined = (rot - 0.5) * np.pi, (rot_refined - 0.5) * np.pi


        dist2 = torch.clamp_min(distCUDA2(mean_3d), 0.0000001)
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3).detach()
        scale = (radius+1)*scales

        dist2 = torch.clamp_min(distCUDA2(mean_3d_refined), 0.0000001)
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3).detach()
        scale_refined = (radius_refined+1)*scales

        # Gaussians and offsets
        # rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant rotation
        opacity = torch.ones((smpl_x.vertex_num_upsampled, 1)
                             ).float().cuda()  # constant opacity

        R_delta, R_delta_refined = batch_rodrigues(
            rot.reshape(-1, 3)), batch_rodrigues(rot_refined.reshape(-1, 3))
        R, R_refined = torch.matmul(R_delta, self.init_rot), torch.matmul(R_delta_refined, self.init_rot)
        # R, R_refined = torch.cat((R, torch.ones_like(R[:, :, :1])), 2), torch.cat((R_refined, torch.ones_like(R_refined[:, :, :1])), 2)
        R_def, R_def_refined = torch.matmul(transform_mat_vertex[:, :3, :3], R), torch.matmul(transform_mat_vertex[:, :3, :3], R_refined)
        if not is_world_coord:
            R_def = torch.matmul(torch.inverse(cam_param['R']), R_def)
            R_def_refined = torch.matmul(torch.inverse(cam_param['R']), R_def_refined)
        rotation, rotation_refined = matrix_to_quaternion(R_def), matrix_to_quaternion(R_def_refined)

        # R_delta, R_delta_refined = batch_rodrigues(
        #     rot.reshape(-1, 3)), batch_rodrigues(rot_refined.reshape(-1, 3))
        # R, R_refined = torch.bmm(self.init_rot, R_delta), torch.bmm(
        #     self.init_rot, R_delta_refined)
        # R_def, R_def_refined = torch.bmm(
        #     transform_mat_vertex[:, :3, :3], R), torch.bmm(self.init_rot, R_refined)
        # rotation, rotation_refined = matrix_to_quaternion(
        #     R_def), matrix_to_quaternion(R_def_refined)

        # semantic_code = self.semantic_code[id]
        

        # semantic_code = self.semantic_code(
        #     id, self.pos_enc_mesh, cfg.triplane_shape_3d, None)[0]
        semantic_code = self.semantic_code[id]
        # print(semantic_code.shape)

        assets = {
            'mean_3d': mean_3d,
            'opacity': opacity,
            'radius': scale,
            'rotation': rotation,
            'rgb': rgb,
            'semantic_code': semantic_code,
        }
        assets_refined = {
            'mean_3d': mean_3d_refined,
            'opacity': opacity,
            'radius': scale_refined,
            'rotation': rotation_refined,
            'rgb': rgb_refined,
            'semantic_code': semantic_code,
        }
        offsets = {
            'mean_offset': mean_offset,
            'mean_offset_offset': mean_offset_offset,
            'radius_offset': radius_offset,
            'appearance_offset': appearance_offset,
            # 'tv_loss': tv_loss,
        }
        return assets, assets_refined, offsets, mesh_neutral_pose
    
    def transfer(self, des_id, src_id1, src_id2, smplx_param, cam_param, label_list, is_world_coord=False):
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(
            jaw_zero_pose=True, use_id_info=True, id=des_id)
        joint_zero_pose = self.get_zero_pose_human(des_id)

        # extract triplane feature
        des_tri_feat = self.extract_tri_feat(des_id)
        src1_tri_feat = self.extract_tri_feat(src_id1)
        src2_tri_feat = self.extract_tri_feat(src_id2)
        
        # semantic_code = self.semantic_code(
        #     id, self.pos_enc_mesh, cfg.triplane_shape_3d, None)[0]
        seg_class = torch.argmax(torch.softmax(self.semantic_code, dim=2), dim=2)

        # for i, label in enumerate(label_list):
        #     if label == 'upper_body':
        #         body_mask = smpl_x.get_upper_body(self.skinning_weight) | self.is_rhand | self.is_lhand 
        #     elif label == 'hands':
        #         body_mask = self.is_rhand | self.is_lhand
        #     elif label == 'scalp':
        #         is_scalp = torch.zeros((smpl_x.vertex_num, 1)).float().cuda()
        #         is_scalp[smpl_x.scalp_vertex_idx] = 1.0
        #         _, is_scalp = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num, 3)).float().cuda(), [is_scalp])
        #         is_scalp = is_scalp[0][:, 0] > 0 
        #         body_mask = (is_scalp ) & ~self.is_neck  # | smpl_x.get_foot(self.skinning_weight)
        #     elif label == 'head':
        #         body_mask = smpl_x.get_head(self.skinning_weight)
        #     elif label == 'eyes':
        #         body_mask = smpl_x.get_eyes(self.skinning_weight)
        #     elif label == 'foot':
        #         body_mask = smpl_x.get_foot(self.skinning_weight)
        #     elif label == 'lower_body':
        #         body_mask = smpl_x.get_lower_body(self.skinning_weight) | smpl_x.get_foot(self.skinning_weight)
        #     elif label == 'face':
        #         body_mask = self.is_face_expr
        #     elif label == "head_hand":
        #         body_mask = self.is_rhand | self.is_lhand | smpl_x.get_head(self.skinning_weight)
        #     elif label == "whole_body":
        #         body_mask = torch.logical_or(smpl_x.get_upper_body(self.skinning_weight),
        #                                     smpl_x.get_lower_body(self.skinning_weight)) | self.is_rhand | self.is_lhand 
        #     else:
        #         raise NotImplementedError
        #     if i == 0:
        #         des_tri_feat[body_mask] = src1_tri_feat[body_mask]
        #         tri_feat = des_tri_feat
        #     else:
        #         tri_feat[body_mask] = src2_tri_feat[body_mask]
        
        body_mask_total = (seg_class[des_id] == 0) | (seg_class[des_id] == 1) | (seg_class[des_id] == 4)
        for i, label in enumerate(label_list):
            if i == 0:
                id = src_id1
            else:
                id = src_id2

            if label == 'hair':
                body_mask = (seg_class[id] == 1) 
            elif label == 'upper_body':
                body_mask = (seg_class[id] == 2) | (seg_class[id] == 4)  #| self.is_rhand | self.is_lhand | smpl_x.get_upper_body(self.skinning_weight) #| self.is_neck #& 
            elif label == 'lower_body':
                body_mask = ((seg_class[id] == 3)) #| (seg_class[id] == 4)  #| smpl_x.get_foot(self.skinning_weight) | smpl_x.get_lower_body(self.skinning_weight) # | smpl_x.get_lower_body(self.skinning_weight) #((seg_class[id] == 3) | (seg_class[id] == 4)) #
            elif label == 'foot':
                body_mask = (seg_class[id] == 4) 
            else:
                raise NotImplementedError
            if i == 0:
                des_tri_feat[body_mask] = src1_tri_feat[body_mask]
                tri_feat = des_tri_feat
                body_mask_total |= body_mask
            else:
                tri_feat[body_mask] = src2_tri_feat[body_mask]
                body_mask_total |= body_mask

        # get Gaussian assets
        geo_feat = self.geo_encoder(tri_feat)
        mean_offset, radius = torch.split(self.surface_net(
            geo_feat), [3, 3], dim=1)  # mean offset of Gaussians
        # radius = self.radius_net(geo_feat)  # scale of Gaussians
        appearance = self.appearance_net(tri_feat)  # rgb of Gaussians
        mean_3d = mesh_neutral_pose + mean_offset  # 大 pose

        radius = torch.sigmoid(radius)
    
        # smplx facial expression offset
        # smplx_expr_offset = (
        #     smplx_param['expr'][None, None, :] * self.expr_dirs).sum(2)
        # mean_3d = mean_3d + smplx_expr_offset  # 大 pose
        

        # minimal_clothed_gs = GaussianModel(sh_degree = 0)
        # minimal_clothed_gs.load_ply("/home/ljr/Downloads/GaussianIP-main/assets/3_white.ply")  # /home/ljr/Downloads/GaussianIP-main/assets/compose4000.ply /home/ljr/Downloads/Cloth_avatar/assets/gauss3.ply
        
        # mean_3d[~body_mask_total] = minimal_clothed_gs.get_xyz[~body_mask_total] #+ torch.tensor([[ 0.0042, -0.2578, -0.0007]], dtype=torch.float32).cuda()
        
        # gaussian_vals = {}
        # gaussian_vals['mean_3d'] = mean_3d

        nn_vertex_idxs = knn_points(mean_3d[None, :, :], mesh_neutral_pose_wo_upsample[None, :, :],
                                    K=1, return_nn=True).idx[0, :, 0]  # dimension: smpl_x.vertex_num_upsampled
        nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
        mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
        nn_vertex_idxs[mask] = torch.arange(
            smpl_x.vertex_num_upsampled).cuda()[mask]

        # get transformation matrix of the nearest vertex and perform lbs
        transform_mat_joint = self.get_transform_mat_joint(
            transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, nn_vertex_idxs)
        mean_3d = self.lbs(mean_3d, transform_mat_vertex,
                           smplx_param['trans'])  # posed with smplx_param

        # camera coordinate system -> world coordinate system
        if not is_world_coord:
            mean_3d = torch.matmul(torch.inverse(
                cam_param['R']), (mean_3d - cam_param['t'].view(1, 3)).permute(1, 0)).permute(1, 0)

        rgb, rot = torch.split(
            torch.sigmoid(appearance), [3, 3], dim=-1)
        
        # rgb[~body_mask_total] = SH2RGB(minimal_clothed_gs.get_features[~body_mask_total].transpose(1,2).reshape(-1,3))


        radius = (radius - 0.5) * 2
        rot = (rot - 0.5) * np.pi

        dist2 = torch.clamp_min(distCUDA2(mean_3d), 0.0000001)
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3).detach()
        scale = (radius+1)*scales
        # scale[~body_mask_total] = minimal_clothed_gs.get_scaling[~body_mask_total]

        opacity = torch.ones((smpl_x.vertex_num_upsampled, 1)
                             ).float().cuda()

        # gaussian_vals['opacity'] = opacity
        # gaussian_vals['scale'] = scale
        # gaussian_vals['rgb'] = rgb
        
        R_delta = batch_rodrigues(rot.reshape(-1, 3))
        R = torch.matmul(R_delta, self.init_rot)

        # R[~body_mask_total] = torch.eye(3, device=R.device).expand(R.shape[0], 3, 3)[~body_mask_total]
        # gaussian_vals['rotation'] = matrix_to_quaternion(R)
    
        # save_gaussians_as_ply('/home/ljr/Downloads/GaussianIP-main/assets/canonical.ply', gaussian_vals)
        # exit()

        R_def = torch.matmul(transform_mat_vertex[:, :3, :3], R)
        if not is_world_coord:
            R_def = torch.matmul(torch.inverse(cam_param['R']), R_def)
        rotation = matrix_to_quaternion(R_def)
        
        assets = {
            'mean_3d': mean_3d,
            'opacity': opacity,
            'radius': scale,
            'rotation': rotation,
            'rgb': rgb,
            'semantic_code': self.semantic_code[des_id],
        }
        assets_refined = None

        
        return assets, assets_refined



class GaussianRenderer(nn.Module):
    def __init__(self):
        super(GaussianRenderer, self).__init__()

    def forward(self, gaussian_assets, img_shape, cam_param, bg=torch.ones((3)).float().cuda(), return_extra=False, state='cotrain'): #torch.ones((3))  (0.8706, 0.9216, 0.9686)   (0.9607, 0.9607, 0.9607)
        # assets for the rendering                                                                            # torch.tensor(((0.8706, 0.9216, 0.9686)))
        mean_3d = gaussian_assets['mean_3d']
        opacity = gaussian_assets['opacity']
        semantic = gaussian_assets['semantic_code']

        seg_class = torch.argmax(torch.softmax(semantic, dim=1), dim=1)

        scale = gaussian_assets['radius']


        rotation = gaussian_assets['rotation']
        rgb = gaussian_assets['rgb']

        # mean_3d_copy = mean_3d.clone() + torch.tensor([0.5, 0.0, 0.6]).float().cuda()
        # opacity_copy = opacity.clone()
        # scale_copy = scale.clone()
        # rotation_copy = rotation.clone()
        # rgb_copy = rgb.clone()
        # rgb_copy[seg_class == 0] = torch.tensor([0.5, 0.0, 0.0]).float().cuda()
        # rgb_copy[seg_class == 1] = torch.tensor([0.0, 0.5, 0.0]).float().cuda()
        # rgb_copy[seg_class == 2] = torch.tensor([0.5, 0.5, 0.0]).float().cuda()
        # rgb_copy[seg_class == 3] = torch.tensor([0.0, 0.0, 0.5]).float().cuda()
        # rgb_copy[seg_class == 4] = torch.tensor([0.5, 0.0, 0.5]).float().cuda()

        # mean_3d_total = torch.cat([mean_3d, mean_3d_copy], dim=0)
        # opacity_total = torch.cat([opacity, opacity_copy], dim=0)
        # scale_total = torch.cat([scale, scale_copy], dim=0)
        # rotation_total = torch.cat([rotation, rotation_copy], dim=0)
        # rgb_total = torch.cat([rgb, rgb_copy], dim=0)

        # normal = gaussian_assets['normal']
        # normal = torch.gather(normal, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        # norms = torch.norm(normal, p=2, dim=1, keepdim=True)
        # norms = torch.where(norms == 0, torch.ones_like(norms), norms)  # 将范数为零的替换为 1
        # normal = normal / norms

        # create rasterizer
        # permute view_matrix and proj_matrix following GaussianRasterizer's configuration following below links
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L55

        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)
        view_matrix = get_view_matrix(
            cam_param['R'], cam_param['t']).permute(1, 0)
        proj_matrix = get_proj_matrix(
            cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1, 0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3, :3]


        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view_matrix,
            projmatrix=full_proj_matrix,
            sh_degree=0,  # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # prepare Gaussian position in the image space for the gradient tracking
        # point_num = mean_3d_total.shape[0]
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num, 3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()

        # rgb[seg_class == 0] = torch.tensor([0.5, 0.0, 0.0]).float().cuda()
        # rgb[seg_class == 1] = torch.tensor([0.0, 0.5, 0.0]).float().cuda()
        # rgb[seg_class == 2] = torch.tensor([0.5, 0.5, 0.0]).float().cuda()
        # rgb[seg_class == 3] = torch.tensor([0.0, 0.0, 0.5]).float().cuda()
        # rgb[seg_class == 4] = torch.tensor([0.5, 0.0, 0.5]).float().cuda()
        if return_extra:
            if state == 'cotrain':
                results = {}
                for i in range(5):
                    render_img_part, depth_part, normal_part, alpha_part, radius_part, extra_part = rasterizer(
                        means3D=mean_3d[seg_class == i],
                        means2D=mean_2d[seg_class == i],
                        shs=None,
                        colors_precomp=rgb[seg_class == i],
                        opacities=opacity[seg_class == i],
                        scales=scale[seg_class == i],
                        rotations=rotation[seg_class == i],
                        cov3Ds_precomp=None,
                        extra_attrs = semantic[seg_class == i])
                    results[f'iteration_{i}'] = (render_img_part, depth_part, normal_part, alpha_part, radius_part, extra_part)
                
                render_img_collect = torch.stack([result[0] for result in results.values()], dim=0)
                depth_collect = torch.stack([result[1] for result in results.values()], dim=0)
                normal_collect = torch.stack([result[2] for result in results.values()], dim=0)
                alpha_collect = torch.stack([result[3] for result in results.values()], dim=0)
                extra_collect = torch.stack([result[5] for result in results.values()], dim=0)
                
                render_img, depth, normal, alpha, radius, extra = rasterizer(
                    means3D=mean_3d,
                    means2D=mean_2d,
                    shs=None,
                    colors_precomp=rgb,
                    opacities=opacity,
                    scales=scale,
                    rotations=rotation,
                    cov3Ds_precomp=None,
                    extra_attrs = semantic)

                return {'img': render_img,
                        'semantic_img': extra,
                        'img_collect': render_img_collect,
                        'semantic_img_collect': extra_collect,
                        'alpha': alpha,
                        'alpha_collect': alpha_collect,
                        'depth': depth,
                        'depth_collect': depth_collect,
                        'normal': normal,
                        'normal_collect': normal_collect,
                        'mean_2d': mean_2d,
                        'is_vis': radius > 0,
                        'radius': radius}
            else:          
                render_img, depth, normal, alpha, radius, _ = rasterizer(
                    means3D=mean_3d,
                    means2D=mean_2d,
                    shs=None,
                    colors_precomp=rgb,
                    opacities=opacity,
                    scales=scale,
                    rotations=rotation,
                    cov3Ds_precomp=None)
                
                _, _, _, _, _, extra = rasterizer(
                    means3D=mean_3d.detach(),
                    means2D=mean_2d,
                    shs=None,
                    colors_precomp=rgb.detach(),
                    opacities=opacity.detach(),
                    scales=scale.detach(),
                    rotations=rotation.detach(),
                    cov3Ds_precomp=None,
                    extra_attrs = semantic)
            

                return {'img': render_img,
                        'semantic_img': extra,
                        'alpha': alpha,
                        'depth': depth,
                        'normal': normal,
                        'mean_2d': mean_2d,
                        'is_vis': radius > 0,
                        'radius': radius}
        
        else:
            render_img, depth, normal, alpha, radius, extra = rasterizer(
                means3D=mean_3d,  #mean_3d_total
                means2D=mean_2d,
                shs=None,
                colors_precomp=rgb,  #rgb_total
                opacities=opacity,   #opacity_total
                scales=scale,       #scale_total
                rotations=rotation,   #rotation_total
                cov3Ds_precomp=None)

            return {'img': render_img,
                    'mean_2d': mean_2d,
                    'is_vis': radius > 0,
                    'radius': radius}


class GaussianRendererTogether(nn.Module):
    def __init__(self):
        super(GaussianRendererTogether, self).__init__()

    def forward(self, gaussian_external, gaussian_internal, img_shape, cam_param, bg=torch.ones((3)).float().cuda(), return_body=False):
        # assets for the rendering
        mean_3d_ex = gaussian_external['mean_3d']
        opacity_ex = gaussian_external['opacity']
        rotation_ex = gaussian_external['rotation']
        rgb_ex = gaussian_external['rgb']

        scale_ex = gaussian_external['radius']
       

        semantic = gaussian_external['semantic_code'].detach()
        seg_class = torch.argmax(torch.softmax(semantic, dim=1), dim=1)
        mask_cloth = (seg_class != 4).bool()


        mean_3d_in = gaussian_internal['mean_3d']
        opacity_in = gaussian_internal['opacity']
        rotation_in = gaussian_external['rotation'] #gaussian_internal['rotation']
        rgb_in = gaussian_internal['rgb']
        # rgb_in = torch.mean(gaussian_external['rgb'][~mask_cloth], dim=0)[None].expand(mean_3d_in.shape[0], -1)
        scale_in = gaussian_internal['radius'] 


        mean_3d = torch.cat([mean_3d_ex, mean_3d_in[mask_cloth]], dim=0)
        opacity = torch.cat([opacity_ex, opacity_in[mask_cloth]], dim=0)
        rotation = torch.cat([rotation_ex, rotation_in[mask_cloth]], dim=0)
        rgb = torch.cat([rgb_ex, rgb_in[mask_cloth]], dim=0)
        scale = torch.cat([scale_ex, scale_in[mask_cloth]], dim=0)

        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)
        view_matrix = get_view_matrix(
            cam_param['R'], cam_param['t']).permute(1, 0)
        proj_matrix = get_proj_matrix(
            cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1, 0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3, :3]
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view_matrix,
            projmatrix=full_proj_matrix,
            sh_degree=0,  # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num, 3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()


        render_img, depth, normal, alpha, radius, extra = rasterizer(
                means3D=mean_3d,
                means2D=mean_2d,
                shs=None,
                colors_precomp=rgb,
                opacities=opacity,
                scales=scale,
                rotations=rotation,
                cov3Ds_precomp=None)
        
        body_xyz = torch.cat([mean_3d_in[mask_cloth], mean_3d_ex[~mask_cloth]], dim=0)
        body_opacity = torch.cat([opacity_in[mask_cloth], opacity_ex[~mask_cloth]], dim=0)
        body_rotation = torch.cat([rotation_in[mask_cloth], rotation_ex[~mask_cloth]], dim=0)
        body_rgb = torch.cat([rgb_in[mask_cloth], rgb_ex[~mask_cloth]], dim=0)
        body_scale = torch.cat([scale_in[mask_cloth], scale_ex[~mask_cloth]], dim=0)
        # print("body xyz", body_xyz.shape)
        # print("body opacity", body_opacity.shape)
        # print('body_color_mean', torch.mean(body_rgb[mask_cloth], dim=0))
        # print('skin_color_mean', torch.mean(body_rgb[~mask_cloth], dim=0))
        # print(body_rgb[mask_cloth].max(), body_rgb[mask_cloth].min())
        # print(body_scale[mask_cloth][:5, :])
        # print(body_scale[~mask_cloth][:5, :])
        # print(body_opacity[mask_cloth][:5, :])
        gauss_body = {'xyz': body_xyz,
                      'opacity': body_opacity,
                      'rotation': body_rotation,
                      'rgb': body_rgb,
                      'scale': body_scale}
        
        render_img_body = None
        
        if return_body:
            point_num = gauss_body['xyz'].shape[0]
            mean_2d_body = torch.zeros((point_num, 3)).float().cuda()


            render_img_body, depth_body, normal_body, alpha_body, radius_body, extra_body = rasterizer(
                    means3D=gauss_body['xyz'],
                    means2D=mean_2d_body,
                    shs=None,
                    colors_precomp=gauss_body['rgb'],
                    opacities=gauss_body['opacity'],
                    scales=gauss_body['scale'],
                    rotations=gauss_body['rotation'],
                    cov3Ds_precomp=None)

        return {'img': render_img,
                    'mean_2d': mean_2d,
                    'is_vis': radius > 0,
                    'radius': radius,
                    'gauss_body': gauss_body,
                    'mask_cloth': mask_cloth,
                    'render_img_body': render_img_body}


class SMPLXParamDict(nn.Module):
    def __init__(self):
        super(SMPLXParamDict, self).__init__()
        self.smplx_params = nn.ParameterDict({})

    # initialize SMPL-X parameters of all frames
    def init(self, smplx_params_list):
        for id, smplx_params in enumerate(smplx_params_list):           
            _smplx_params = {}
            for frame_idx in smplx_params.keys():
                _smplx_params[frame_idx] = nn.ParameterDict({})
                for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                    if 'pose' in param_name:
                        _smplx_params[frame_idx][param_name] = nn.Parameter(matrix_to_rotation_6d(
                            axis_angle_to_matrix(smplx_params[frame_idx][param_name].cuda())))
                    else:
                        _smplx_params[frame_idx][param_name] = nn.Parameter(
                            smplx_params[frame_idx][param_name].cuda())

            self.smplx_params[str(id)] = nn.ParameterDict(_smplx_params)

    def get_optimizable_params(self):
        optimizable_params = []
        # for id, smplx_params in enumerate(self.smplx_params):
        for id, smplx_params in self.smplx_params.items():
            for frame_idx in smplx_params.keys():
                for param_name in smplx_params[frame_idx].keys():
                    optimizable_params.append({'params': [smplx_params[frame_idx][param_name]],
                                            'name': 'id_' + id + '_' + 'smplx_' + param_name + '_' + frame_idx, 'lr': cfg.smplx_param_lr})
        return optimizable_params

    def forward(self, id, frame_idxs):
        out = []
        for frame_idx in frame_idxs:
            frame_idx = str(frame_idx.item())
            # frame_idx = frame_idx.item()
            smplx_param = {}
            for param_name in self.smplx_params[id][frame_idx].keys():
                if 'pose' in param_name:
                    smplx_param[param_name] = matrix_to_axis_angle(
                        rotation_6d_to_matrix(self.smplx_params[id][frame_idx][param_name]))
                else:
                    smplx_param[param_name] = self.smplx_params[id][frame_idx][param_name]
            out.append(smplx_param)
        return out



def customized_export_ply(outfile_name, v, f = None, v_n = None, v_c = None, f_c = None, e = None):
    '''
    Author: Jinlong Yang, jyang@tue.mpg.de

    Exports a point cloud / mesh to a .ply file
    supports vertex normal and color export
    such that the saved file will be correctly displayed in MeshLab

    # v: Vertex position, N_v x 3 float numpy array
    # f: Face, N_f x 3 int numpy array
    # v_n: Vertex normal, N_v x 3 float numpy array
    # v_c: Vertex color, N_v x (3 or 4) uchar numpy array
    # f_n: Face normal, N_f x 3 float numpy array
    # f_c: Face color, N_f x (3 or 4) uchar numpy array
    # e: Edge, N_e x 2 int numpy array
    # mode: ascii or binary ply file. Value is {'ascii', 'binary'}
    '''

    v_n_flag=False
    v_c_flag=False
    f_c_flag=False
    
    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        if type(v_n) == 'torch.Tensor':
            v_n = v_n.detach().cpu().numpy()
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            # warnings.warn("Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype = np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                # warnings.warn("Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype = np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n'%(N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n'%(N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n'%(N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2], \
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2],\
                    f_c[i,0], f_c[i,1], f_c[i,2], f_c[i,3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n'%(e[i,0], e[i,1]))


def vertex_normal_2_vertex_color(vertex_normal):
    # Normalize vertex normal
    import torch
    if torch.is_tensor(vertex_normal):
        vertex_normal = vertex_normal.detach().cpu().numpy()
    normal_length = ((vertex_normal**2).sum(1))**0.5
    normal_length = normal_length.reshape(-1, 1)
    vertex_normal /= normal_length
    # Convert normal to color:
    color = vertex_normal * 255/2.0 + 128
    return color.astype(np.ubyte)


class GaussConverter(nn.Module):
    def __init__(self, model, stage='stage_one'):
        super(GaussConverter, self).__init__()
        self.model = model
        if stage == "stage_one":
            for param in self.model.module.parameters():
                param.requires_grad = False
            self.model.eval()
        self.gaussians = GaussianModel()
        self.init_()
        self.gauss_renderer = GaussianRendererTogether()
        self.rgb_loss = RGBLoss()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.lap_reg = LaplacianReg(smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.consistency_loss = ColorConsistencyLoss()
        self.layer_loss = LayeredConstraintLoss()

    def init_(self):
        with torch.no_grad():
            pcd_list = []
            for i in range(self.model.module.human_gaussian.num_subjects):
                pcd, _, _, _ = self.model.module.human_gaussian.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True, id=i)
                pcd_list.append(pcd)
            pcd_total = torch.stack(pcd_list, dim=0)
        self.gaussians.create_from_pcd(pcd_total, spatial_lr_scale=1.0)
        self.gaussians.training_setup(cfg)
    
    def forward(self, data, mode):
        
        batch_size = data['cam_param']['R'].shape[0]
        img_height, img_width = data['img'].shape[2:]

        if mode == 'train':
            bg = torch.rand(3).float().cuda()
        else:
            bg = torch.ones((3)).float().cuda()

        for i in range(batch_size):
            subject_id = data['subject_id'][i].item()
            

            smplx_param = self.model.module.smplx_param_dict(str(subject_id), [data['frame_idx'][i]])[0]

            human_asset, human_asset_refined, human_offset, mesh_neutral_pose \
                = self.model.module.human_gaussian(subject_id, smplx_param, {k: v[i] for k,v in data['cam_param'].items()})

            mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose\
                  = self.model.module.human_gaussian.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True, id=subject_id)
            
            label =  torch.argmax(torch.softmax(self.model.module.human_gaussian.semantic_code[subject_id], dim=1), dim=1)
            mask_cloth = (label!= 4).bool()

            gaussians_body = self.gaussians.get_subject(subject_id)
            # gaussians_body = self.scale_assets(human_asset, mesh_neutral_pose, human_offset, mask_cloth)
            gaussians_body_offset = (gaussians_body['mean_3d'][mask_cloth] - mesh_neutral_pose[mask_cloth])
            joint_zero_pose = self.model.module.human_gaussian.get_zero_pose_human(subject_id)

            nn_vertex_idxs = knn_points(gaussians_body['mean_3d'][None, :, :], mesh_neutral_pose_wo_upsample[None, :, :],
                                    K=1, return_nn=True).idx[0, :, 0]  # dimension: smpl_x.vertex_num_upsampled
            nn_vertex_idxs = self.model.module.human_gaussian.lr_idx_to_hr_idx(nn_vertex_idxs)
            mask = (self.model.module.human_gaussian.is_rhand + self.model.module.human_gaussian.is_lhand + self.model.module.human_gaussian.is_face) > 0
            nn_vertex_idxs[mask] = torch.arange(
                smpl_x.vertex_num_upsampled).cuda()[mask]

            # get transformation matrix of the nearest vertex and perform lbs
            transform_mat_joint = self.model.module.human_gaussian.get_transform_mat_joint(
                transform_mat_neutral_pose, joint_zero_pose, smplx_param)
            transform_mat_vertex = self.model.module.human_gaussian.get_transform_mat_vertex(
                transform_mat_joint, nn_vertex_idxs)

            gaussians_body['mean_3d'] = self.model.module.human_gaussian.lbs(gaussians_body['mean_3d'], transform_mat_vertex,
                            smplx_param['trans'])  # posed with smplx_param
            gaussians_body['mean_3d'] = torch.matmul(torch.inverse(
                data['cam_param']['R'][i]), (gaussians_body['mean_3d'] - data['cam_param']['t'][i].view(1, 3)).permute(1, 0)).permute(1, 0)
            
            # human_render_refined = self.gauss_renderer(human_asset_refined, gaussians_body, (img_height, img_width), {
            #                                           k: v[i] for k, v in data['cam_param'].items()}, bg)
           
        if mode == 'train':
            human_render = self.gauss_renderer(human_asset, gaussians_body, (img_height, img_width), {
                                                  k: v[i] for k, v in data['cam_param'].items()}, bg)
            # loss functions
            loss = {}
            # loss['ssim_human'] = (1 - self.ssim(human_render['img'][None], data['img'],
            #                       bbox=data['bbox'], mask=data['mask'])) * cfg.ssim_loss_weight
            # loss['lpips_human'] = self.lpips(
            #     human_render['img'][None], data['img'], bbox=data['bbox']) * cfg.lpips_weight
            
            # loss['rgb_human_rand_bg'] = self.rgb_loss(
            #     human_render['img'][None], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])
            
            # loss['ssim_human_refined'] = (1 - self.ssim(human_render_refined['img'],
            #                               data['img'], bbox=data['bbox'], mask=data['mask'])) * cfg.ssim_loss_weight
            # loss['lpips_human_refined'] = self.lpips(
            #     human_render_refined['img'], data['img'], bbox=data['bbox']) * cfg.lpips_weight
            # loss['rgb_human_refined_rand_bg'] = self.rgb_loss(
            #     human_render_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
            
            weight = torch.ones(
                (1, smpl_x.vertex_num_upsampled, 1)).float().cuda()
            # weight[:, self.human_gaussian.is_rhand, :] = 100
            # weight[:, self.human_gaussian.is_lhand, :] = 100
            loss['lap_xyz'] = (self.lap_reg(human_render['gauss_body']['xyz'][None], None))
            loss['lap_rgb'] = (self.lap_reg(human_render['gauss_body']['rgb'][None], None))
            loss['consistency_loss'] = self.consistency_loss(
                                human_render['gauss_body']['rgb'][mask_cloth], human_render['gauss_body']['rgb'][~mask_cloth]) * 10
            # loss['rgb_reg'] = gaussians_body['rgb'][mask_cloth] ** 2 *10
            # loss['xyz_reg'] = gaussians_body_offset ** 2 * 100000
            loss['scale_reg'] = human_render['gauss_body']['scale'][mask_cloth] ** 2 *1000
            # loss['layer'] = self.layer_loss(human_render['gauss_body']['xyz'], human_asset['mean_3d'])
            
            return loss

        else:
            human_render = self.gauss_renderer(human_asset, gaussians_body, (img_height, img_width), {
                                                  k: v[i] for k, v in data['cam_param'].items()}, bg, return_body=True)
            return human_render
        
    def scale_assets(self, assets, mesh_neutral_pose, offsets, mask_cloth):
        a = copy.deepcopy(assets)
        normal = self.compute_normal(mesh_neutral_pose)
        dot_product = torch.sum(offsets['mean_offset'] * normal, dim=1, keepdim=True)
        
        # 创建全1的PyTorch张量，与mesh_neutral_pose类型一致
        scale = torch.ones_like(mesh_neutral_pose) * 1/3
        
        # 创建与scale形状匹配的掩码
        positive_mask = dot_product >= 0
        negative_mask = dot_product < 0
        
        # 扩展掩码以匹配scale的形状
        positive_mask = positive_mask.expand_as(scale)
        negative_mask = negative_mask.expand_as(scale)
        
        # 应用缩放因子
        # scale[positive_mask] = 1/3
        # scale[negative_mask] = 9/6        
        # 计算处理后的3D坐标
        a['mean_3d'] = mesh_neutral_pose + offsets['mean_offset'] * scale
        a['radius'] = assets['radius'] * 1/2
        a['rgb'] = torch.mean(assets['rgb'][~mask_cloth], dim=0)[None].expand(mesh_neutral_pose.shape[0], -1)
        
        return a
    

    def compute_normal(self, mesh_neutral_pose):
        """计算内层顶点法向量（简化版：邻域面法向量平均）"""
        with torch.no_grad():
            normal = Meshes(verts=mesh_neutral_pose[None,:,:], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None,:,:]).verts_normals_packed().reshape(1,smpl_x.vertex_num_upsampled,3).detach()
        
        return normal[0]

