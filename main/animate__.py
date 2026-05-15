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
from base import Transfer
from utils.smpl_x import smpl_x
from utils.vis import render_mesh
from obj_io import save_gaussians_as_ply
from utils.preprocessing import load_img, get_bbox, parse_outfits
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_matrix
from pytorch3d.renderer import look_at_view_transform

import copy
import math
import trimesh
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

    transfer = Transfer(args.test_epoch)
    transfer._make_batch_generator()
    transfer._make_model()
    start_idx, end_idx = transfer.transet.data_length['0']

        
        

    # load ID information
    root_path = osp.join('..', 'data', cfg.dataset, 'data')
    outfits = parse_outfits(cfg)
    data_dirs = {outfit: osp.join(root_path, outfit) for outfit in outfits['seen'].keys()}
    save_root_path = cfg.result_dir
    os.makedirs(save_root_path, exist_ok=True)



    motion_path = args.motion_path
    if motion_path[-1] == '/':
        motion_name = motion_path[:-1].split('/')[-1]
    else:        
        motion_name = motion_path.split('/')[-1]
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params', '*.json'))])[::]
    render_shape = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx_list[0]) + '.png')).shape[:2]

    

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


    # for subject_id, (outfit, outfit_datadir) in enumerate(data_dirs.items()):
    # video_out = cv2.VideoWriter(motion_name + outfit + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (render_shape[1]*3, render_shape[0]))
    subject_id = 8
    for frame_idx in tqdm(frame_idx_list):
        with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}

        # cam_param = cam_params[frame_idx]

        with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', str(frame_idx) + '.json')) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}
            smplx_param['expr'] = torch.zeros_like(smplx_param['expr']).cuda()
        # forward
        if frame_idx == 514:
            smplx_param_fix = copy.deepcopy(smplx_param)
            smplx_param_fix['root_pose']=torch.FloatTensor([math.pi,0,0]).cuda()
            smplx_param_fix['trans']: torch.FloatTensor((0,0,0)).float().cuda() # type: ignore

            render_shape_fix = (1024, 1024)
            cam_param_fix = {'R': torch.FloatTensor(((1,0,0), (0,1,0), (0,0,1))).float().cuda(), \
                        't': torch.zeros((3)).float().cuda(), \
                        'focal': torch.FloatTensor((3000,3000)).cuda(), \
                        'princpt': torch.FloatTensor((render_shape_fix[1]/2, render_shape_fix[0]/2)).cuda()}
            
            mesh_cam = torch.matmul(axis_angle_to_matrix(smplx_param_fix['root_pose']), smpl_x.layer['neutral'].v_template.cuda().permute(1,0)).permute(1,0) + smplx_param_fix['trans'].view(1,3)
            at_point_cam = mesh_cam.mean(0)
            at_point = torch.matmul(torch.inverse(cam_param_fix['R']), (at_point_cam - cam_param_fix['t']))
            cam_pos = torch.matmul(torch.inverse(cam_param_fix['R']), -cam_param_fix['t'].view(3,1)).view(3)

            view_num = 60
            save_path = './rest_pose'
            os.makedirs(save_path, exist_ok=True)

            for i in range(view_num):
                azim = math.pi + math.pi*2*i/view_num # azim angle of the camera
                elev = -math.pi/12 
                dist = torch.sqrt(torch.sum((cam_pos - at_point)**2)) # distance between camera and mesh
                R, t = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=False, at=at_point[None,:], up=((0,1,0),)) 
                R = torch.inverse(R)
                cam_param_rot = {'R': R[0].cuda(), 't': t[0].cuda(), 'focal': cam_param_fix['focal'], 'princpt': cam_param_fix['princpt']}

                with torch.no_grad():
                    human_asset, human_asset_refined = transfer.model.module.human_gaussian.transfer(8, 2, 2, smplx_param_fix, cam_param_fix, ['upper_body', 'lower_body'])
                    human_render = transfer.model.module.gaussian_renderer(human_asset, render_shape_fix, cam_param_rot)
                cv2.imwrite(osp.join(save_path, str(subject_id) + '_' + str(i) + '.png'), (human_render['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8))
 


        # with torch.no_grad():
        #     human_asset, human_asset_refined = transfer.model.module.human_gaussian.transfer(7, 0, 3, smplx_param, cam_param, ['whole_body', 'foot'])
        #     human_render = transfer.model.module.gaussian_renderer(human_asset, render_shape, cam_param)
        #     human_img = human_render['img'].cpu().numpy()
        #     cv2.imwrite(osp.join(save_root_path, str(subject_id) + '_' + str(frame_idx) + '_human.png'), human_img.transpose(1,2,0)[:,:,::-1]*255)


            # smplx mesh render
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
            # face_offset = smpl_x.face_offset[subject_id].cuda()[None]
            # joint_offset = tester.model.module.human_gaussian.joint_offset[subject_id][None]
            # output = tester.model.module.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
            # mesh = output.vertices[0]
            # mesh_render = render_mesh(mesh, smpl_x.face, cam_param, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)

            # img = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx) + '.png'))
            # render = (human_render['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
        
            # font_size = 1.5
            # thick = 3
            # cv2.putText(img, 'image', (int(1/3*img.shape[1]), int(0.05*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
            # cv2.putText(mesh_render, 'rendered SMPL-X mesh', (int(1/5*mesh_render.shape[1]), int(0.05*mesh_render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2)
            # cv2.putText(render, 'render', (int(1/3*render.shape[1]), int(0.05*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 

            # out = np.concatenate((img, mesh_render, render),1).astype(np.uint8)
            # out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.05), int(out.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, 2)
            # video_out.write(out)
        
        # video_out.release()
        
if __name__ == "__main__":
    main()
