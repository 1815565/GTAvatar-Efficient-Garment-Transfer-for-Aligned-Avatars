from dataclasses import dataclass

import os
import torch
import torch.nn.functional as F

import numpy as np
import math
import random
from argparse import ArgumentParser
import yaml

import lpips
import matplotlib.pyplot as plt

import threestudio
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
)
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, LaplacianReg, save_gaussians_as_ply
from threestudio.utils.normalrender import NormalRenderer
from threestudio.utils.typing import *
from threestudio.utils.smpl_x import smpl_x

from gaussiansplatting.gaussian_renderer import render, render_with_smaller_scale
from gaussiansplatting.arguments import PipelineParams, OptimizationParams
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.utils.sh_utils import RGB2SH, SH2RGB
import cv2
from pytorch3d.renderer import look_at_view_transform


@threestudio.register("gaussianip-system")
class GaussianIP(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # basic settings
        log_path: str = "GaussianIP-main"
        cur_time: str = ""
        config_path: str = "configs/exp.yaml"
        stage: str = "stage1"
        apose: bool = True
        bg_white: bool = False
        radius: float = 4
        ipa_ori: bool = True
        use_pose_controlnet: bool = False
        smplx_path: str = "/path/to/smplx/model"
        pts_num: int = 100000
        sh_degree: int = 0
        height: int = 512
        width: int = 512
        ori_height: int = 1024
        ori_width: int = 1024
        head_offset: float = 0.65

        # 3dgs optimization related
        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        
        # densify & prune settings
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        densify_prune_min_opacity: float = 0.15
        densify_prune_screen_size_threshold: int = 20
        densify_prune_world_size_threshold: float = 0.008
        densify_prune_screen_size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        gender: str = 'neutral'
        
        # prune_only settings
        prune_only_start_step: int = 1700
        prune_only_end_step: int = 1900
        prune_only_interval: int = 300
        prune_opacity_threshold: float = 0.05
        prune_screen_size_threshold: int = 20
        prune_world_size_threshold: float = 0.008

        # refine related
        refine_start_step: int = 2400
        refine_n_views: int = 64
        refine_train_bs: int = 16
        refine_elevation: float = 17.
        refine_fovy_deg: float = 70.
        refine_camera_distance: float = 1.5
        refine_patch_size: int = 200
        refine_num_bboxes: int = 3
        lambda_l1: float = 1.0
        lambda_lpips: float = 0.5

    cfg: Config

    def configure(self) -> None:
        self.log_path = self.cfg.log_path
        self.cur_time = self.cfg.cur_time
        self.config_path = self.cfg.config_path
        self.stage = self.cfg.stage
        self.radius = self.cfg.radius
        self.gaussian = GaussianModel(sh_degree = self.cfg.sh_degree)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.ipa_ori = self.cfg.ipa_ori
        self.use_pose_controlnet = self.cfg.use_pose_controlnet
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.head_offset = self.cfg.head_offset

        self.refine_start_step = self.cfg.refine_start_step
        self.refine_n_views = self.cfg.refine_n_views
        self.refine_train_bs = self.cfg.refine_train_bs
        self.refine_elevation = self.cfg.refine_elevation
        self.refine_fovy_deg = self.cfg.refine_fovy_deg
        self.refine_camera_distance = self.cfg.refine_camera_distance
        self.refine_patch_size = self.cfg.refine_patch_size
        self.refine_num_bboxes = self.cfg.refine_num_bboxes
        self.refine_batch = self.create_refine_batch()
        self.l1_loss_fn = F.l1_loss
        self.lpips_loss_fn = lpips.LPIPS(net='vgg')
        self.refine_loss = {'training_step': [], 'l1_loss': [], 'lpips_loss': []}
        self.refine_logger = []

        self.skel = NormalRenderer()
        self.lap_reg = LaplacianReg(
            smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.point_cloud = self.pcd()
        
        self.cameras_extent = 1.0


    def pcd(self):
        points = self.skel.sample_smplx_points()

        return points
    

    def forward(self, batch: Dict[str, Any], renderbackground=None, phase='train') -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
            
        images = []
        depths = []
        pose_images = []
        self.viewspace_point_list = []

        for id in range(batch['R'].shape[0]):
            viewpoint_cam  = Camera(R = batch['R'][id], T = batch['T'][id], height = batch['height'], width = batch['width'])
            if phase == 'val' or phase == 'test':
                render_pkg = render_with_smaller_scale(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            else:
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)

            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)
                
            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            
            if phase == 'train':
                self.height = self.cfg.height
                self.width = self.cfg.width
            else:
                self.height = self.cfg.ori_height
                self.width = self.cfg.ori_width


            pose_image = self.skel.render_mesh(viewpoint_cam)
            pose_images.append(pose_image)


        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        pose_images = torch.cat(pose_images, 0)


        self.visibility_filter = self.radii > 0.0

        # pass
        # if self.cfg.disable_hand_densification:
        #     points = self.gaussian.get_xyz # [N, 3]
        #     hand_centers = torch.from_numpy(self.skel.hand_centers).to(points.dtype).to('cuda') # [2, 3]
        #     distance = torch.norm(points[:, None, :] - hand_centers[None, :, :], dim=-1) # [N, 2]
        #     hand_mask = distance.min(dim=-1).values < self.cfg.hand_radius # [N]
        #     self.visibility_filter = self.visibility_filter & (~hand_mask)

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images
        render_pkg["opacity"] = self.gaussian.get_opacity
        render_pkg["scale"] = self.gaussian.get_scaling
        render_pkg['xyz'] = self.gaussian.get_xyz
        render_pkg['sh'] = self.gaussian.get_features
        render_pkg['rot'] = self.gaussian.get_rotation

        return {
            **render_pkg,
        }

    def create_refine_batch(self):

        R_all, T_all = [], []
        root_joint_world = torch.zeros(3)  # 物体中心在世界坐标系原点

        for i in range(self.refine_n_views):
            # 1. 计算方位角（azim）：从 0 到 2π，实现环绕一周
            azim = 2 * math.pi * i / self.refine_n_views  # 修正：覆盖 0~2π 范围
            
            if i == 0:
                # 相机位置：初始在 (0, 0, dist)，指向物体在原点
                dist = 4.0  # 相机到物体的距离（可根据需求调整）
                cam_pos = torch.tensor([0.0, 0.0, dist])  # 世界坐标系下的相机位置
                at_point = root_joint_world  # 相机看向的目标点（物体中心）
                
                # 2. 正确计算仰角（elev）：基于相机与目标点的相对位置
                # 相对位置 = 相机位置 - 目标点位置
                relative_pos = cam_pos - at_point  # [0, 0, dist] - [0,0,0] = [0,0,dist]
                # 仰角 = arctan2(相对y坐标, 相对z坐标)，避免除以零
                elev = torch.arctan2(relative_pos[1], relative_pos[2])  # 此处为 0（水平视角）
            
            # 3. 生成相机姿态（R: 相机到世界的旋转，T: 相机到世界的平移）
            # degrees=False 表示输入为弧度
            R, T = look_at_view_transform(
                dist=dist,
                elev=elev,
                azim=azim,
                degrees=False,
                at=at_point[None, :],  # 目标点形状为 (1, 3)
                up=((0, -1, 0),)  # 上方向为y轴
            )
            
            # 4. 若需要世界到相机的旋转，取逆（旋转矩阵的逆 = 转置，更稳定）
            R = R.transpose(1, 2)  # 等价于 R.inverse()，但旋转矩阵转置更稳定
            R_all.append(R)
            T_all.append(T)

        R_all = torch.cat(R_all, dim=0).cuda()
        T_all = torch.cat(T_all, dim=0).cuda()

        return {
            "R": R_all,
            "T": T_all,
            "height": self.height,                  # 512
            "width": self.width,                    # 512
        }


    def render_refine_rgb(self, phase='init', renderbackground=None):
        if renderbackground is None:
            renderbackground = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")  #self.background_tensor

        
        images = []
        depths = []
        pose_images = []
        self.viewspace_point_list = []

        assert phase in ['init', 'random', 'debug']

        if phase == 'init':
            id_list = [i for i in range(self.refine_n_views)]
        elif phase == 'random':
            id_list = random.sample(range(self.refine_n_views), self.refine_train_bs)
        else:
            id_list = [0, 8, 16, 24, 31]
        
        refine_height = self.refine_batch['height']
        refine_width = self.refine_batch['width']

        for idx, id in enumerate(id_list):
            viewpoint_cam  = Camera(R = self.refine_batch['R'][id], T = self.refine_batch['T'][id], height = refine_height, width = refine_width)
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            # manually accumulate max radii across self.refine_batch
            # if idx == 0:
            #     self.refine_radii = radii
            # else:
            #     self.refine_radii = torch.max(radii, self.refine_radii)
                
            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)

            images.append(image) # [1024, 1024, 3]
            depths.append(depth) # [1024, 1024, 3]

            pose_image = self.skel.render_mesh(viewpoint_cam)
            pose_images.append(pose_image)


        images = torch.stack(images, 0)             # [refine_n_views or refine_train_bs, 1024, 1024, 3]
        depths = torch.stack(depths, 0)             # [refine_n_views or refine_train_bs, 1024, 1024, 3]
        pose_images = torch.cat(pose_images, 0)   # [refine_n_views or refine_train_bs, 1024, 1024, 3]
        print(pose_images.shape)

        # self.refine_visibility_filter = self.refine_radii > 0.0
# render_pkg = {}
        render_pkg["comp_rgb"] = images   
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images

        return {**render_pkg}, id_list


    def on_fit_start(self) -> None:
    #     super().on_fit_start()
    #     # stage 1: AHDS training
        if self.stage == "stage1":
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.prepare_for_sds(self.prompt_processor.prompt, self.prompt_processor.negative_prompt, self.prompt_processor.null_prompt)
            
            
            self.gaussian2 = GaussianModel(sh_degree = 0)
            self.gaussian2.load_ply("/home/ljr/Downloads/Cloth_avatar/assets/gauss0.ply")
            semantic_data = np.load("/home/ljr/Downloads/Cloth_avatar/assets/semantic0.npy")

            self.mask_body = (
                (torch.from_numpy(semantic_data) == 0)      
            ).to(self.device)


            # self.mask_body = (
            #     (torch.from_numpy(semantic_data) == 0) | (torch.from_numpy(semantic_data) == 1) 
                
            # ).to(self.device)

            # self.gaussian3 = GaussianModel(sh_degree = 0)
            # self.gaussian3.load_ply("logs/20250910-165121/it4000.ply")     #logs/20250828-144323/it4000.ply   logs/20250827-003638/it3200.ply      logs/20250819-140907/it4000.ply  logs/20250826-230827/it2400.ply
            
            # gauss_vals = {}
            # gauss_vals['mean_3d'] = torch.empty_like(self.gaussian3.get_xyz)
            # gauss_vals['mean_3d'][self.mask_body] = self.gaussian2.get_xyz[self.mask_body]
            # gauss_vals['mean_3d'][~self.mask_body] = self.gaussian3.get_xyz[~self.mask_body]
            # gauss_vals['mean_3d'] = gauss_vals['mean_3d'] + torch.tensor(self.skel.ori_center, dtype=torch.float32).cuda()

            # gauss_vals['sh'] = torch.empty_like(self.gaussian3.get_features)
            # gauss_vals['sh'][self.mask_body] = self.gaussian2.get_features[self.mask_body]
            # gauss_vals['sh'][~self.mask_body] = self.gaussian3.get_features[~self.mask_body]

            # gauss_vals['rotation'] = torch.empty_like(self.gaussian3.get_rotation)
            # gauss_vals['rotation'][self.mask_body] = self.gaussian2.get_rotation[self.mask_body]
            # gauss_vals['rotation'][~self.mask_body] = self.gaussian3.get_rotation[~self.mask_body]

            # gauss_vals['opacity'] = torch.empty_like(self.gaussian3.get_opacity)
            # gauss_vals['opacity'][self.mask_body] = self.gaussian2.get_opacity[self.mask_body]
            # gauss_vals['opacity'][~self.mask_body] = self.gaussian3.get_opacity[~self.mask_body]

            # gauss_vals['scale'] = torch.empty_like(self.gaussian3.get_scaling)
            # gauss_vals['scale'][self.mask_body] = self.gaussian2.get_scaling[self.mask_body]
            # gauss_vals['scale'][~self.mask_body] = self.gaussian3.get_scaling[~self.mask_body]

            # save_gaussians_as_ply("/home/ljr/Downloads/GaussianIP-main/assets/Liu4000.ply", gauss_vals)
    

            # self.gaussian2 = GaussianModel(sh_degree = 0)
            # self.gaussian2.load_ply("/home/ljr/Downloads/GaussianIP-main/assets/Liu4000.ply")
        
        else:
            self.refined_rgbs_small = torch.load(os.path.join(self.log_path, self.cur_time, 'after_refine.pth'))['refined_rgbs_small'].to(self.device)


    def training_step(self, batch, batch_idx):
        if self.true_global_step < self.refine_start_step-1 and self.stage == "stage1" and self.true_global_step!= 0: # and self.true_global_step!= 0
            self.gaussian.update_learning_rate(self.true_global_step)

            out = self(batch, phase='train')

            prompt_utils = self.prompt_processor()
            images = out["comp_rgb"]
            control_images = out['pose']
            all_vis_all = None #out['all_vis_all']

            guidance_out = self.guidance(self.true_global_step, images, control_images, prompt_utils, self.use_pose_controlnet, all_vis_all, **batch)
            
            # init loss
            loss = 0.0

            # loss_sds
            loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])
            
            # # loss_sparsity
            # loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            # self.log("train/loss_sparsity", loss_sparsity)
            # loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            
            # # loss_opaque
            # opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            # loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            # self.log("train/loss_opaque", loss_opaque)
            # loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)


            mean_3d = torch.cat([out['xyz'][~self.mask_body], self.gaussian2.get_xyz[self.mask_body].detach()], dim=0)
            # rotation = torch.cat([out['rot'][~self.mask_body], self.gaussian2.get_rotation[self.mask_body].detach()], dim=0)
            sh = torch.cat([out['sh'][~self.mask_body], self.gaussian2.get_features[self.mask_body].detach()], dim=0).transpose(1,2).reshape(-1,3)
            scale = torch.cat([out['scale'][~self.mask_body], self.gaussian2.get_scaling[self.mask_body].detach()], dim=0)

            loss_lap_xyz = self.lap_reg(mean_3d[None], self.point_cloud[None]).mean()
            loss_lap_sh = self.lap_reg(sh[None], None).mean()
            loss_lap_scale = self.lap_reg(scale[None], None).mean()
            # loss_lap_rot = self.lap_reg(rotation[None], None).mean()
            loss_scale = (out['scale'] ** 2).mean()
            loss_sh = F.mse_loss(self.gaussian2.get_features[self.mask_body].detach().mean(dim=0)[None], out['sh']).cuda()  # self.gaussian2.get_features[self.mask_body].detach().mean(dim=0)[None]
                                                                    #   SH2RGB(out['sh'])  ,  1 ：0.647, 0.588, 0.588   3 ： 0.725 0.588 0.588
            print("loss sds: ", guidance_out['loss_sds'])
            print("loss lap_xyz: ", loss_lap_xyz)
            print("loss lap_sh: ", loss_lap_sh)
            print("loss lap_scale: ", loss_lap_scale)
            # print("loss lap_rot: ", loss_lap_rot)
            print("loss scale: ", loss_scale)
            print("loss sh: ", loss_sh)
            # print("loss lap_opacity: ", loss_lap_opacity)
            # # print("loss_offset: ", loss_offset)
            
            if self.true_global_step < 1300:
                s1 = 5
                s2 = 10000
            else:
                s1 = 5
                s2 = 10000

            loss = 0.5 * loss + loss_lap_xyz * 100 +\
                 + loss_lap_sh * s1  + loss_sh * 20 + (loss_lap_scale + loss_scale) * s2# (loss_lap_scale + loss_scale) * s2 + loss_lap_rot * 1

            # loss = loss + loss_lap_xyz * 100 + loss_scale * 1000 
            
            #+  loss_lap_opacity * 0.01  #+ loss_offset * 10     + loss_lap_rot   
            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))

            return {"loss": loss}

        elif self.true_global_step == self.refine_start_step-1  :  #self.refine_start_step-1   0
            gs_out, _ = self.render_refine_rgb(phase='init', renderbackground=None)
            images = gs_out["comp_rgb"].detach()      # [refine_n_views, H, W, 3]
            control_images = gs_out['pose'].detach()  # [refine_n_views, H, W, 3]
            # save image data before refine
            images = images.to('cpu').numpy()
            control_images = control_images.to('cpu').numpy()
            torch.save({'images': images, 'control_images': control_images}, os.path.join(self.log_path, self.cur_time, 'before_refine.pth'))

            # self.refined_rgbs, self.view_idx_all = self.guidance.refine_rgb(images, control_images, self.prompt_processor.prompt)  # [refine_n_views, H, W, 3]
            
            self.view_idx_all = [24, 8, 16, 0, 20, 28, 4, 12, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
            # save images before refine
            for i, _ in enumerate(self.view_idx_all):
                cur_raw_rgb = images[i]
                cur_control_image = control_images[i]
                # cur_refined_rgb = self.refined_rgbs[i]
                self.save_image(f"raw_rgb_{i}.png", cur_raw_rgb)
                self.save_image(f"control_image_{i}.png", cur_control_image)
                # self.save_image(f"refined_rgb_{view_idx}.png", cur_refined_rgb)

                # cv2.imwrite(f"control_image_{i}.png", cv2.cvtColor(cur_control_image*255, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(f"raw_rgb_{i}.png", cur_raw_rgb*255)
                
            # self.idx_mapper = [3, 20, 21, 22, 6, 23, 24, 25, 1, 26, 27, 28, 7, 29, 30, 31, 2, 8, 9, 10, 4, 11, 12, 13, 0, 14, 15, 16, 5, 17, 18, 19]
            # self.refined_rgbs = self.refined_rgbs[self.idx_mapper]
            # self.refined_rgbs = self.refined_rgbs.permute(0, 3, 1, 2)[:, :, 60:890, 220:800]
            # self.refined_rgbs_small = F.interpolate(self.refined_rgbs, scale_factor=0.5, mode="bilinear", align_corners=False)
            return None


    def on_before_optimizer_step(self, optimizer):
        if self.true_global_step % 100 == 0:
            threestudio.info('Gaussian points num: {}'.format(self.gaussian.get_features.shape[0]))
        # if self.true_global_step < self.refine_start_step and self.stage == "stage1":
        #     with torch.no_grad():
        #         if self.true_global_step < self.cfg.densify_prune_end_step:
        #             viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
        #             for idx in range(len(self.viewspace_point_list)):
        #                 viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
        #             # Keep track of max radii in image-space for pruning
        #             self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
        #             self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

        #             # densify_and_prune
        #             self.min_opacity = self.cfg.densify_prune_min_opacity if self.true_global_step > 1900 else 0.05
        #             if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0:
        #                 densify_prune_screen_size_threshold = self.cfg.densify_prune_screen_size_threshold if self.true_global_step > self.cfg.densify_prune_screen_size_threshold_fix_step else None
        #                 self.gaussian.densify_and_prune(self.cfg.max_grad, self.min_opacity, self.cameras_extent, densify_prune_screen_size_threshold, self.cfg.densify_prune_world_size_threshold) 

        #         # "prune-only" phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
        #         if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
        #             viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
        #             for idx in range(len(self.viewspace_point_list)):
        #                 viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
        #             # Keep track of max radii in image-space for pruning
        #             self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
        #             self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

        #             if self.true_global_step % self.cfg.prune_only_interval == 0:
        #                 self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)
        
        # if self.stage == "stage3":
        #     with torch.no_grad():
        #         if self.true_global_step + self.refine_start_step < 10000:
        #             viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
        #             for idx in range(len(self.viewspace_point_list)):
        #                 viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
        #             # Keep track of max radii in image-space for pruning
        #             if self.true_global_step == 0:
        #                 # When stage 3 starts, the loaded gaussians don't have max_radii2D
        #                 self.gaussian.max_radii2D = self.refine_radii  # self.get_xyz.shape[0]
        #                 self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
        #                 self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
        #             else:
        #                 self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
        #                 self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
        #             # densify_and_prune
        #             if self.true_global_step + self.refine_start_step == 2500:
        #                 densify_prune_screen_size_threshold = self.cfg.densify_prune_screen_size_threshold if self.true_global_step > self.cfg.densify_prune_screen_size_threshold_fix_step else None
        #                 self.gaussian.densify_and_prune(self.cfg.max_grad, 0.05, self.cameras_extent, densify_prune_screen_size_threshold, self.cfg.densify_prune_world_size_threshold) 

        #         # "prune-only" phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
        #         if self.true_global_step + self.refine_start_step > 2500 and self.true_global_step + self.refine_start_step < 3000:
        #             viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
        #             for idx in range(len(self.viewspace_point_list)):
        #                 viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
        #             # Keep track of max radii in image-space for pruning
        #             self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
        #             self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
        #             if self.true_global_step + self.refine_start_step % self.cfg.prune_only_interval == 0:
        #                 self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)


    def validation_step(self, batch, batch_idx):
        out = self(batch, phase='val')
        if self.stage == "stage1":
            self.save_image(f"it{self.true_global_step}-{batch['index'][0]}_rgb.png", out["comp_rgb"][0])
        else:
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-{batch['index'][0]}_rgb.png", out["comp_rgb"][0])


    def on_validation_epoch_end(self):
        if self.true_global_step % 200 == 0: #self.true_global_step > 2300 and
            self.gaussian.save_ply(os.path.join(self.log_path, self.cur_time, f"it{self.true_global_step}.ply"))


    # test the gaussians
    def test_step(self, batch, batch_idx):
        if self.stage == "stage1":
            pass
        else:
            pass
            bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]
            background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 新创建的tensor需指定device
            out = self(batch, renderbackground=background_tensor, phase='test')
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/rgb/{batch['index'][0]}.png", out["comp_rgb"][0])
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/pose/{batch['index'][0]}.png", out["pose"][0])
        

    # save something
    def on_test_epoch_end(self):
        if self.stage == "stage1":
            self.gaussian.save_ply(os.path.join(self.log_path, self.cur_time, f"it{self.true_global_step}.ply"))
        else:
            self.save_img_sequence(
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step + self.refine_start_step,
            )
            save_path = self.get_save_path(f"it{self.true_global_step + self.refine_start_step}-test/last.ply")
            self.gaussian.save_ply(save_path)

            # change the max_steps in config.yaml from total_step to refine_start_step
            config_file_path = self.config_path

            # read config.yaml
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)

            # change args
            config['system']['stage'] = 'stage1'
            config['trainer']['max_steps'] = self.refine_start_step

            # write it back to config.yaml
            with open(config_file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            print(f"Updated max_steps to {self.refine_start_step} in {config_file_path}.")
            

    def configure_optimizers(self):
        if self.stage == "stage1":
            opt = OptimizationParams(self.parser)
            # point_cloud = self.pcd()
            self.gaussian.create_from_pcd(self.point_cloud, self.cameras_extent)
            self.gaussian.training_setup(opt)
            ret = {"optimizer": self.gaussian.optimizer}
        else:
            # load 3dgs from stage 1
            opt = OptimizationParams(self.parser)
            self.gaussian.load_ply(os.path.join(self.log_path, self.cur_time, f"it{self.refine_start_step}.ply"))
            self.gaussian.training_setup(opt)
            ret = {"optimizer": self.gaussian.optimizer}
        return ret
