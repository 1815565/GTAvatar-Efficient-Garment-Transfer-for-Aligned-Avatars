import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
import cv2
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from utils.vis import render_mesh
from obj_io import save_gaussians_as_ply
import trimesh
from utils.preprocessing import load_img, get_bbox, parse_outfits
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_matrix
from pytorch3d.renderer import look_at_view_transform
import math
import copy
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    parser.add_argument('--save_ply', action='store_true')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    assert args.motion_path, 'Motion path for the animation is required.'
    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()


        

    # load ID information
    root_path = osp.join('..', 'data', cfg.dataset, 'data')
    outfits = parse_outfits(cfg)
    data_dirs = {outfit: osp.join(root_path, outfit) for outfit in outfits['seen'].keys()}
    # for subject_id, (outfit, outfit_datadir) in enumerate(data_dirs.items()):
    #     with open(osp.join(outfit_datadir, 'smplx_optimized', 'shape_param.json')) as f:
    #         shape_param = torch.FloatTensor(json.load(f))
    #     with open(osp.join(outfit_datadir, 'smplx_optimized', 'face_offset.json')) as f:
    #         face_offset = torch.FloatTensor(json.load(f))
    #     with open(osp.join(outfit_datadir, 'smplx_optimized', 'joint_offset.json')) as f:
    #         joint_offset = torch.FloatTensor(json.load(f))
    #     with open(osp.join(outfit_datadir, 'smplx_optimized', 'locator_offset.json')) as f:
    #         locator_offset = torch.FloatTensor(json.load(f))
    #     smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

    # tester.smplx_params = [None] * len(data_dirs)
    # tester._make_model()

    motion_path = args.motion_path
    if motion_path[-1] == '/':
        motion_name = motion_path[:-1].split('/')[-1]
    else:        
        motion_name = motion_path.split('/')[-1]
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', '*.json'))])[::2]
    # print(frame_idx_list[0], type(frame_idx_list[0]))
    render_shape = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx_list[0]) + '.png')).shape[:2]

    save_root_path = cfg.result_dir
    os.makedirs(save_root_path, exist_ok=True)

    # cam_params = {}
    # with open(osp.join(args.motion_path, 'sparse', 'cameras.txt')) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if line[0] == '#':
    #             continue
    #         splitted = line.split()
    #         _, _, width, height, focal_x, focal_y, princpt_x, princpt_y = splitted
    #     focal = np.array((float(focal_x), float(focal_y)), dtype=np.float32) # shared across all frames
    #     princpt = np.array((float(princpt_x), float(princpt_y)), dtype=np.float32) # shared across all frames
    # with open(osp.join(args.motion_path, 'sparse', 'images.txt')) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if line[0] == '#':
    #             continue
    #         if 'png' not in line:
    #             continue
    #         splitted = line.split()
    #         frame_idx = int(splitted[-1][:-4])
    #         qw, qx, qy, qz = float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])
    #         tx, ty, tz = float(splitted[5]), float(splitted[6]), float(splitted[7])
    #         R = quaternion_to_matrix(torch.FloatTensor([qw, qx, qy, qz])).numpy()
    #         t = np.array([tx, ty, tz], dtype=np.float32)
    #         cam_params[frame_idx] = {'R': torch.FloatTensor(R).cuda(), 't': torch.FloatTensor(t).cuda(), 'focal': torch.FloatTensor(focal).cuda(), 'princpt': torch.FloatTensor(princpt).cuda()}

    
    for subject_id, (outfit, outfit_datadir) in enumerate(data_dirs.items()):
        if subject_id in [0,1,2]:
            continue
        # if subject_id != 0:
        #     continue
        # video_out = cv2.VideoWriter(osp.join(save_root_path, f'{subject_id}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 7, (render_shape[1], render_shape[0]))
        # video_out2 = cv2.VideoWriter(osp.join(save_root_path, f'{subject_id}_refine.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 7, (render_shape[1], render_shape[0]))
        for frame_idx in tqdm(frame_idx_list):
            with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
                cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
                # cam_param['focal'] = torch.FloatTensor([3000, 3000]).cuda()

            # cam_param = cam_params[frame_idx]

            with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params', str(frame_idx) + '.json')) as f:
                smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}
                smplx_param['expr'] = torch.zeros_like(smplx_param['expr']).cuda()


            # if frame_idx == 365:
                # root_pose = smplx_param['root_pose'].view(1,3)
                # body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint_part['body'])-1)*3)
                # jaw_pose = smplx_param['jaw_pose'].view(1,3)
                # leye_pose = smplx_param['leye_pose'].view(1,3)
                # reye_pose = smplx_param['reye_pose'].view(1,3)
                # lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint_part['lhand'])*3)
                # rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint_part['rhand'])*3)
                # expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
                # trans = smplx_param['trans'].view(1,3)
                # shape = tester.model.module.human_gaussian.shape_param[subject_id][None]
                # face_offset = (smpl_x.face_offset[subject_id].cuda())[None]
                # joint_offset = tester.model.module.human_gaussian.joint_offset[subject_id][None]
                # output = tester.model.module.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
                # mesh, root_joint_cam = output.vertices[0], output.joints[0][smpl_x.root_joint_idx]
                # mesh = torch.matmul(torch.inverse(cam_param['R']), (mesh - cam_param['t'].view(-1,3)).permute(1,0)).permute(1,0) # camera coordinate -> world coordinate
                # mesh_render = render_mesh(mesh, smpl_x.face, cam_param, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)
                # root_joint_world = torch.matmul(torch.inverse(cam_param['R']), root_joint_cam - cam_param['t']) # camera coordinate -> world coordinate

                #cv2.imwrite(osp.join(save_root_path, str(subject_id) + '_' + str(frame_idx) + '_human_pose.png'), mesh_render)

                # view_num = 60
                # save_path = './rest_pose'
                # os.makedirs(save_path, exist_ok=True)

                # for i in range(view_num):
                #     # make camera parmeters with look_at function
                #     azim = math.pi + math.pi*2*i/view_num # azim angle of the camera
                #     if i == 0:
                #         at_point_orig = root_joint_world.clone()
                #         at_point = root_joint_world # world coordinate
                #         cam_pos = torch.matmul(torch.inverse(cam_param['R']), -cam_param['t'].view(3,1)).view(3) # get camera position (world coordinate system)
                #         at_point_cam = root_joint_cam # camera coordinate
                #         elev = torch.arctan(torch.abs(at_point_cam[1])/torch.abs(at_point_cam[2])) # elev angle of the camera
                #         dist = torch.sqrt(torch.sum((cam_pos - at_point)**2)) # distance between camera and mesh
                #     mesh[:,[0,2]] = mesh[:,[0,2]] - root_joint_world[None,[0,2]] + at_point_orig[None,[0,2]]
                #     R, t = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=False, at=at_point[None,:], up=((0,1,0),)) 
                #     R = torch.inverse(R)
                #     cam_param_rot = {'R': R[0].cuda(), 't': t[0].cuda(), 'focal': cam_param['focal'], 'princpt': cam_param['princpt']}

                #     with torch.no_grad():
                #         human_asset, human_asset_refined, _, _ = tester.model.module.human_gaussian(subject_id, smplx_param, cam_param)
                #         human_asset['mean_3d'][:,[0,2]] = human_asset['mean_3d'][:,[0,2]] - root_joint_world[None,[0,2]] + at_point_orig[None,[0,2]]
                #         human_render = tester.model.module.gaussian_renderer(human_asset, render_shape, cam_param_rot)
                #     cv2.imwrite(osp.join(save_path, str(subject_id) + '_' + str(i) + '.png'), (human_render['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8))
    

            with torch.no_grad():
                human_asset, human_asset_refined, human_offset, mesh_neutral_pose = tester.model.module.human_gaussian(subject_id, smplx_param, cam_param)


            #     # is_palm, is_back = smpl_x.get_hand_palm_back(mesh_neutral_pose, tester.model.module.human_gaussian.skinning_weight)
            #     # human_asset['rgb'][is_palm] = torch.mean(human_asset['rgb'][is_back], dim=0)

            #     # out = tester.model.module.gaussian_renderer(human_asset_refined, render_shape, cam_param)
            #     # human_img_refined = out['img'].cpu().numpy()

                out = tester.model.module.gaussian_renderer(human_asset, render_shape, cam_param)
                human_img = out['img'].cpu().numpy()

            #     # cv2.imwrite(osp.join(save_root_path, str(subject_id) + '_' + str(frame_idx) + '_human_refined.png'), human_img_refined.transpose(1,2,0)[:,:,::-1]*255)
                cv2.imwrite(osp.join(save_root_path, str(subject_id) + '_' + str(frame_idx) + '_human.png'), human_img.transpose(1,2,0)[:,:,::-1]*255)
            #     # video_out.write((human_img.transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8))
            #     # video_out2.write((human_img_refined.transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8))


        # video_out.release()
        # video_out2.release()
        
        

        
    
if __name__ == "__main__":
    main()
