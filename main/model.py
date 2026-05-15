import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.module import HumanGaussian, SMPLXParamDict, GaussianRenderer
from nets.layer import MeshRenderer
from nets.loss import RGBLoss, SSIM, LPIPS, LaplacianReg, JointOffsetSymmetricReg, HandMeanReg, HandRGBReg, ArmRGBReg, SemanticLoss, TVLoss, GloabalDirReg
from utils.smpl_x import smpl_x
from utils.flame import flame
from render import Renderer
import copy



class Model(nn.Module):
    def __init__(self, human_gaussian, smplx_param_dict):
        super(Model, self).__init__()
        self.human_gaussian = human_gaussian
        self.smplx_param_dict = smplx_param_dict
        self.gaussian_renderer = GaussianRenderer()
        # self.smplx_renderer = Renderer()
        self.face_mesh_renderer = MeshRenderer(flame.vertex_uv, flame.face_uv)
        if cfg.fit_pose_to_test:
            self.optimizable_params = self.smplx_param_dict.get_optimizable_params()
        else:
            self.optimizable_params = self.human_gaussian.get_optimizable_params()
            if smplx_param_dict is not None:
                self.optimizable_params += self.smplx_param_dict.get_optimizable_params()
        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender])
        self.rgb_loss = RGBLoss()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.lap_reg = LaplacianReg(
            smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()
        self.hand_mean_reg = HandMeanReg()
        self.hand_rgb_reg = HandRGBReg()
        self.arm_rgb_reg = ArmRGBReg()
        self.eval_modules = [self.lpips]
        self.semantic = SemanticLoss()
        self.global_dir_reg = GloabalDirReg()
        

        self.tv_loss = TVLoss()

    def get_smplx_outputs(self, id, smplx_param, cam_param):
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint_part['body'])-1)*3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint_part['lhand'])*3)
        rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint_part['rhand'])*3)
        expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
        trans = smplx_param['trans'].view(1,3)

        shape = self.human_gaussian.shape_param[id][None]
        face_offset = smpl_x.face_offset[id].cuda()[None]
        joint_offset = self.human_gaussian.joint_offset[id][None]

        # camera coordinate system
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
        mesh = output.vertices[0]

        # camera coordinate system -> world coordinate system
        mesh = torch.matmul(torch.inverse(cam_param['R']), (mesh - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        return mesh

    def forward(self, data, mode, epoch):
        batch_size = data['cam_param']['R'].shape[0]
        img_height, img_width = data['img'].shape[2:]

        # in the training, increase SH degree following the schedule
        # in the testing stage, load and use it

        # background color for the human-only rendering
        if mode == 'train':
            bg = torch.rand(3).float().cuda()
        else:
            bg = torch.ones((3)).float().cuda()

        # bg=torch.tensor(((0.9607, 0.9607, 0.9607))).float().cuda()
        # get assets for the rendering and render
        human_assets, human_assets_refined, human_offsets, smplx_outputs = {}, {}, {}, []
        scene_renders, human_renders, scene_human_renders = {}, {}, {}
        human_renders_refined, scene_human_renders_refined = {}, {}
        face_renders, face_renders_refined = [], []
        for i in range(batch_size):

            subject_id = data['subject_id'][i].item()

            smplx_param = self.smplx_param_dict(str(subject_id), [data['frame_idx'][i]])[0]
            
            human_asset, human_asset_refined, human_offset, mesh_neutral_pose = self.human_gaussian(subject_id, smplx_param, {k: v[i] for k,v in data['cam_param'].items()})

            key_list = ['mean_3d', 'opacity', 'rotation', 'rgb', 'radius', 'semantic_code']

            # gather assets
            for key in key_list:
                if key not in human_assets:
                    human_assets[key] = [human_asset[key]]
                    human_assets_refined[key] = [human_asset_refined[key]]
                else:
                    human_assets[key].append(human_asset[key])
                    human_assets_refined[key].append(human_asset_refined[key])

            # gather offsets
            for key in ['mean_offset', 'mean_offset_offset', 'radius_offset', 'appearance_offset']:
            # for key in ['mean_offset', 'mean_offset_offset', 'radius_offset', 'appearance_offset', 'tv_loss']:
                if key not in human_offsets:
                    human_offsets[key] = [human_offset[key]]
                else:
                    human_offsets[key].append(human_offset[key])

            # smplx outputs
            smplx_output = self.get_smplx_outputs(subject_id,
                smplx_param, {k: v[i] for k, v in data['cam_param'].items()})
            smplx_outputs.append(smplx_output)
            
            # normal_map = self.smplx_renderer.render_mesh(smplx_output[None], smpl_x.face, {k: v[i, None] for k, v in data['cam_param'].items()}, (img_height, img_width), mode='n')
            
            
            if epoch < cfg.appearance_itr:
                return_extra = False
                key_list = ['img']
                state = 'appearance'
            elif epoch < cfg.independ_itr:
                return_extra = True
                key_list = ['img', 'semantic_img', 'alpha', 'depth', 'normal']
                state = 'independent'
            else:
                return_extra = True
                key_list = ['img', 'semantic_img', 'semantic_img_collect', 'img_collect', 'alpha', 'alpha_collect', 'depth', 'normal']
                state = 'cotrain'

            # human render
            human_render = self.gaussian_renderer(human_asset, (img_height, img_width), {
                                                  k: v[i] for k, v in data['cam_param'].items()}, bg, return_extra=return_extra, state=state)
            for key in key_list: #'semantic_img_collect',
                if key not in human_renders:
                    human_renders[key] = [human_render[key]]
                else:
                    human_renders[key].append(human_render[key])

            # human render (refined)
            human_render_refined = self.gaussian_renderer(human_asset_refined, (img_height, img_width), {
                                                          k: v[i] for k, v in data['cam_param'].items()}, bg, return_extra=return_extra)
            for key in key_list: #'semantic_img_collect', 
                if key not in human_renders_refined:
                    human_renders_refined[key] = [human_render_refined[key]]
                else:
                    human_renders_refined[key].append(
                        human_render_refined[key])
                    
            # sd_mask_render = self.gaussian_renderer(human_asset_refined, (img_height, img_width), {
            #                                               k: v[i] for k, v in data['cam_param'].items()}, torch.zeros((3)).float().cuda(), return_mask=True)
            # sd_mask = sd_mask_render['alpha']

            # face render
            face_texture, face_texture_mask = flame.texture[subject_id][None], flame.texture_mask[subject_id][None, 0:1]
            face_texture = torch.cat((face_texture, face_texture_mask), 1)
            face_render = self.face_mesh_renderer(face_texture, human_asset['mean_3d'][None, smpl_x.face_vertex_idx, :], flame.face, {
                                                  k: v[i, None] for k, v in data['cam_param'].items()}, (img_height, img_width))
            face_render_refined = self.face_mesh_renderer(face_texture, human_asset_refined['mean_3d'][None, smpl_x.face_vertex_idx, :], flame.face, {
                                                          k: v[i, None] for k, v in data['cam_param'].items()}, (img_height, img_width))
            face_renders.append(face_render[0])
            face_renders_refined.append(face_render_refined[0])

        # aggregate assets and renders
        # do not perform any differentiable operations on mean_2d to get its gradients (we should make it the left node)
        human_assets = {k: torch.stack(v) for k, v in human_assets.items()}
        human_assets_refined = {k: torch.stack(
            v) for k, v in human_assets_refined.items()}
        human_offsets = {k: torch.stack(v) for k, v in human_offsets.items()}
        smplx_outputs = torch.stack(smplx_outputs)
        human_renders = {k: torch.stack(v) for k, v in human_renders.items()}
        human_renders_refined = {k: torch.stack(
            v) for k, v in human_renders_refined.items()}
        face_renders = torch.stack(face_renders)
        face_renders_refined = torch.stack(face_renders_refined)

        if mode == 'train':
            # loss functions
            loss = {}
            
            loss['ssim_human'] = (1 - self.ssim(human_renders['img'], data['img'],
                                bbox=data['bbox'], mask=data['mask'])) * cfg.ssim_loss_weight
            loss['lpips_human'] = self.lpips(
                human_renders['img'], data['img'], bbox=data['bbox']) * cfg.lpips_weight
            is_face = ((face_renders[:, :3] != -1) *
                    (face_renders[:, 3:] == 1)).float()
            loss['rgb_face'] = self.rgb_loss(human_renders['img'] * (
                1 - is_face) + face_renders[:, :3] * is_face, data['img'], bbox=data['bbox']) * cfg.rgb_loss_weight
            
            loss['rgb_human_rand_bg'] = self.rgb_loss(
                human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])
            
            loss['ssim_human_refined'] = (1 - self.ssim(human_renders_refined['img'],
                                        data['img'], bbox=data['bbox'], mask=data['mask'])) * cfg.ssim_loss_weight
            loss['lpips_human_refined'] = self.lpips(
                human_renders_refined['img'], data['img'], bbox=data['bbox']) * cfg.lpips_weight
            is_face = ((face_renders_refined[:, :3] != -1)
                    * (face_renders_refined[:, 3:] == 1)).float()
            loss['rgb_face_refined'] = self.rgb_loss(human_renders_refined['img'] * (
                1 - is_face) + face_renders_refined[:, :3] * is_face, data['img'], bbox=data['bbox']) * cfg.rgb_loss_weight
            
            loss['rgb_human_refined_rand_bg'] = self.rgb_loss(
                human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight

            
            if epoch >= cfg.appearance_itr:
                loss['lap_semantic'] = self.lap_reg(torch.softmax(human_assets['semantic_code'], dim=2), None) 
                loss['semantic'] = self.semantic(human_renders['semantic_img'], data['semantic_gt'],  bbox=data['bbox'], mask=data['mask']) * 0.1
                loss['semantic_refined'] = self.semantic(human_renders_refined['semantic_img'], data['semantic_gt'],  bbox=data['bbox'], mask=data['mask']) * 0.1                
                loss['alpha'] = F.l1_loss(human_renders['alpha'], data['mask']) * 0.01
                loss['alpha_refined'] = F.l1_loss(human_renders_refined['alpha'], data['mask']) * 0.01
                loss['tv'] = self.tv_loss(torch.softmax(human_renders['semantic_img'], dim=1)) * 1   # 0.1 -> 1     decrease 0.5
                loss['tv_refined'] = self.tv_loss(torch.softmax(human_renders_refined['semantic_img'], dim=1)) * 1   # 0.1 -> 1     decrease 0.5
            if epoch >= cfg.independ_itr:

                loss['semantic_partition'] = self.semantic(human_renders['semantic_img_collect'][0], data['semantic_gt'],  bbox=data['bbox'], mask=data['part_mask'].permute(1,0,2,3)) * 1    # 0.1 -> 1     increase 0.3
                loss['semantic_refined_partition'] = self.semantic(human_renders_refined['semantic_img_collect'][0], data['semantic_gt'],  bbox=data['bbox'], mask=data['part_mask'].permute(1,0,2,3)) * 1  # 0.1 -> 1     increase 0.3
                loss['alpha_partition'] = F.l1_loss(human_renders['alpha_collect'][0], data['part_mask'].permute(1,0,2,3)) * 0.01
                loss['alpha_partition_refined'] = F.l1_loss(human_renders_refined['alpha_collect'][0], data['part_mask'].permute(1,0,2,3)) * 0.01
                loss['tv_partition'] = self.tv_loss(torch.softmax(human_renders['semantic_img_collect'][0], dim=1)) * 1      # 0.1 -> 1     decrease 0.3
                loss['tv_partition_refined'] = self.tv_loss(torch.softmax(human_renders_refined['semantic_img_collect'][0], dim=1)) * 1    # 0.1 -> 1     decrease 0.3

                loss['rgb_human_rand_bg_partition'] = self.rgb_loss(
                human_renders['img_collect'][0], data['img'], bbox=data['bbox'], mask=data['part_mask'].permute(1,0,2,3), bg=bg[None]) * 1.2
                loss['rgb_human_refined_rand_bg_partition'] = self.rgb_loss(
                human_renders_refined['img_collect'][0], data['img'], bbox=data['bbox'], mask=data['part_mask'].permute(1,0,2,3), bg=bg[None]) * 1.2

 
            if cfg.fit_pose_to_test:
                return loss

            weight = torch.ones(
                (1, smpl_x.vertex_num_upsampled, 1)).float().cuda() * 10
            weight[:, self.human_gaussian.is_rhand, :] = 1000
            weight[:, self.human_gaussian.is_lhand, :] = 1000
            # weight[:, self.human_gaussian.is_face, :] = 1
            weight[:, self.human_gaussian.is_face_expr, :] = 10
            loss['gaussian_mean_reg'] = (
                human_offsets['mean_offset'] ** 2 + human_offsets['mean_offset_offset'] ** 2) * weight
            

            # loss['gaussian_mean_hand_reg'] = self.hand_mean_reg(mesh_neutral_pose, human_offsets['mean_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand) + self.hand_mean_reg(
            #     mesh_neutral_pose, human_offsets['mean_offset_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand)

            # loss['dir_reg'] = self.global_dir_reg(mesh_neutral_pose, human_offsets['mean_offset'])

            weight = torch.ones(
                (1, smpl_x.vertex_num_upsampled, 1)).float().cuda()
            # weight[:, self.human_gaussian.is_face, :] = 10
            weight[:, self.human_gaussian.is_face_expr, :] = 50
            loss['lap_mean'] = (self.lap_reg(mesh_neutral_pose[None, :, :].detach() + human_offsets['mean_offset'], mesh_neutral_pose[None, :, :].detach()) +
                                self.lap_reg(mesh_neutral_pose[None, :, :].detach() + human_offsets['mean_offset'] + human_offsets['mean_offset_offset'], mesh_neutral_pose[None, :, :].detach())) * 100000 * weight

            weight = torch.ones(
                (1, smpl_x.vertex_num_upsampled, 1)).float().cuda()
            weight[:, self.human_gaussian.is_rhand, :] = 100
            weight[:, self.human_gaussian.is_lhand, :] = 100
            loss['lap_rgb'] = (self.lap_reg(human_assets['rgb'], None) +
                               self.lap_reg(human_assets_refined['rgb'], None)) * weight
            loss['lap_rot'] = (self.lap_reg(human_assets['rotation'], None) +
                               self.lap_reg(human_assets_refined['rotation'], None)) * 0.001 * weight
            
            loss['lap_scale'] = (self.lap_reg(human_assets['radius'], None) + self.lap_reg(human_assets_refined['radius'], None)) * 1000 * weight
            
            # loss['lap_opacity'] = (self.lap_reg(human_assets['opacity'], None) +
            #                        self.lap_reg(human_assets_refined['opacity'], None)) * 10 * weight
            
            loss['hand_rgb_reg'] = (self.hand_rgb_reg(human_assets['rgb'], self.human_gaussian.is_rhand, self.human_gaussian.is_lhand) +
                                    self.hand_rgb_reg(human_assets_refined['rgb'], self.human_gaussian.is_rhand, self.human_gaussian.is_lhand)) * 0.01
            is_upper_arm, is_lower_arm = smpl_x.get_arm(
                mesh_neutral_pose, self.human_gaussian.skinning_weight)
            loss['arm_rgb_reg'] = (self.arm_rgb_reg(mesh_neutral_pose, is_upper_arm, is_lower_arm, human_assets['rgb']) +
                                   self.arm_rgb_reg(mesh_neutral_pose, is_upper_arm, is_lower_arm, human_assets_refined['rgb'])) * 0.1

            weight = torch.ones((smpl_x.joint_num, 3)).float().cuda() * 10
            weight[smpl_x.joint_part['lhand'], :] = 100
            weight[smpl_x.joint_part['rhand'], :] = 100
            loss['joint_offset_reg'] = (
                self.human_gaussian.joint_offset[subject_id] - smpl_x.joint_offset[subject_id].cuda()) ** 2 * weight
            loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(
                self.human_gaussian.joint_offset[subject_id])

            # loss['tv_loss'] = human_offsets['tv_loss'].mean() * 1 #0.1
            return loss
        else: 
            out = {}

            out['human_img'] = human_renders['img']

            out['human_img_refined'] = human_renders_refined['img']

            out['partition'] = human_renders['img_collect'][0]
            
            # out['alpha'] = human_renders_refined['alpha']
            # out['alpha_partition'] = human_renders_refined['alpha_collect'][0]
            
            # out['semantic_img'] = torch.argmax(torch.softmax(human_renders['semantic_img'].permute(0,2,3,1), dim=3), dim=3)
            # out['semantic_img_collect'] = torch.argmax(torch.softmax(human_renders_refined['semantic_img_collect'][0].permute(0,2,3,1), dim=3), dim=3)
            # out['normal_refined'] = human_renders_refined['normal']

            
            out['smplx_mesh'] = smplx_outputs
            
            # out['smplx_normal'] = normal_map 
            # out['sd_mask'] = sd_mask

            return out, human_assets
    
    def transfer(self, des_id, src_id1, src_id2, data, label, mode="transfer"):
        batch_size = data['cam_param']['R'].shape[0]
        img_height, img_width = data['img'].shape[2:]
        
        
        # in the training, increase SH degree following the schedule
        # in the testing stage, load and use it
        
        # background color for the human-only rendering
        if mode == 'train':
            bg = torch.rand(3).float().cuda()
        else:
            bg = torch.ones((3)).float().cuda()
           
        # get assets for the rendering and render
        human_assets, human_assets_refined = {}, {}
        human_renders, human_renders_refined = {}, {}
        smplx_outputs = []
        
        for i in range(batch_size):
            subject_id = data['subject_id'][i].item()
            assert subject_id == des_id
            # get assets form scene Gaussians            
            # get assets and offsets from human Gaussians            

            smplx_param = self.smplx_param_dict(str(subject_id), [data['frame_idx'][i]])[0]
            human_asset, human_asset_refined = self.human_gaussian.transfer(subject_id, src_id1, src_id2, smplx_param, {k: v[i].cuda() for k,v in data['cam_param'].items()}, label)
            
            # clamp scale in early of the training as garbage large scales from randomly initialized networks take HUGE GPU memory
            key_list = ['mean_3d', 'opacity', 'rotation', 'rgb', 'radius']

            # gather assets
            for key in key_list:
                if key not in human_assets:
                    human_assets[key] = [human_asset[key]]
                else:
                    human_assets[key].append(human_asset[key])

            # bg=torch.tensor(((0.8706, 0.9216, 0.9686))).float().cuda()
            # human render
            human_render = self.gaussian_renderer(human_asset, (img_height, img_width), {k: v[i].cuda() for k,v in data['cam_param'].items()}, bg)
            for key in ['img']:
                if key not in human_renders:
                    human_renders[key] = [human_render[key]]
                else:
                    human_renders[key].append(human_render[key])
            

            smplx_output = self.get_smplx_outputs(subject_id,
                smplx_param, {k: v[i].cuda() for k, v in data['cam_param'].items()})
            smplx_outputs.append(smplx_output)
            
        # aggregate assets and renders
        # do not perform any differentiable operations on mean_2d to get its gradients (we should make it the left node)
        human_assets = {k: torch.stack(v) for k,v in human_assets.items()}
        human_renders = {k: torch.stack(v) for k,v in human_renders.items()}
        smplx_outputs = torch.stack(smplx_outputs)
       
        
        out = {}

        out['human_img'] = human_renders['img']

        out['smplx_mesh'] = smplx_outputs

        return out

    def quick_forward(self, data):
        batch_size = data['cam_param']['R'].shape[0]
        img_height, img_width = data['img'].shape[2:]
        bg = torch.ones((3)).float().cuda()

        human_renders = {}
        for i in range(batch_size):
            subject_id = data['subject_id'][i].item()
            smplx_param = self.smplx_param_dict(str(subject_id), [data['frame_idx'][i]])[0]
            # print(data['cam_param']['R'].device)
            human_asset = self.human_gaussian.quick_deform(subject_id, smplx_param, {k: v[i].cuda() for k,v in data['cam_param'].items()})

            # human render
            human_render = self.gaussian_renderer(human_asset, (img_height, img_width), {
                                                  k: v[i].cuda() for k, v in data['cam_param'].items()}, bg)
            for key in ['img']:
                if key not in human_renders:
                    human_renders[key] = [human_render[key]]
                else:
                    human_renders[key].append(human_render[key])

        human_renders = {k: torch.stack(v) for k, v in human_renders.items()}
        out = {}

        out['human_img'] = human_renders['img']
        return out

def get_model(smplx_params):
    num_subject = len(smplx_params)
    human_gaussian = HumanGaussian(num_subject)
    if smplx_params is not None:
        smplx_param_dict = SMPLXParamDict()
        with torch.no_grad():
            smplx_param_dict.init(smplx_params)
    else:
        smplx_param_dict = None
    with torch.no_grad():
        human_gaussian.init()

    

    model = Model(human_gaussian, smplx_param_dict)
    return model
