import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Transfer
import os
import os.path as osp
import cv2
from utils.smpl_x import smpl_x
from utils.vis import render_mesh
from pytorch3d.io import save_obj
import trimesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--fit_pose_to_test', dest='fit_pose_to_test', action='store_true')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--save_ply', action='store_true')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id, args.fit_pose_to_test)

    transfer = Transfer(args.test_epoch)
    transfer._make_batch_generator()
    transfer._make_model()
    start_idx, end_idx = transfer.transet.data_length['0']      
    
    # for batch in transfer.batch_generator:
    #     # batch 应该是一个字典，其中包含了 'img' 等键
    #     render_shape = batch['img'].shape[2:]
    #     break  # 如果你只需要第一个批次，使用 break 跳出循环
    # print(render_shape)
    save_root_path = cfg.result_dir
    os.makedirs(save_root_path, exist_ok=True)
    # video_out = cv2.VideoWriter(save_root_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (render_shape[1]*1, render_shape[0]))


    for itr, data in enumerate(tqdm(transfer.batch_generator)):
        if itr < start_idx: continue
        if itr >= end_idx: break

        # forward
        with torch.no_grad():
            out = transfer.model.module.transfer(0, 3, 5, data, ['upper_body', 'lower_body'])
        # save
        human_img = out['human_img'].cpu().numpy()
        

        batch_size = human_img.shape[0]
        for i in range(batch_size):
            subject_id = '4'
            frame_idx = int(data['frame_idx'][i])
            # cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_human_refined.png'), human_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            # cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_gt.png'), data['img'][i].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_human.png'), human_img[i].transpose(1,2,0)[:,:,::-1]*255)
            
            # mesh = out['smplx_mesh'][i]
        
            # mesh_render = render_mesh(mesh, smpl_x.face, {k: v[i].cuda() for k,v in data['cam_param'].items()}, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)

            # img = cv2.imread(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_gt.png'), cv2.IMREAD_UNCHANGED)
            # render = (human_img[i].transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
            # font_size = 1.5
            # thick = 3
            # cv2.putText(img, 'image', (int(1/3*img.shape[1]), int(0.05*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
            # cv2.putText(mesh_render, 'rendered SMPL-X mesh', (int(1/5*mesh_render.shape[1]), int(0.05*mesh_render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2)
            # cv2.putText(render, 'render', (int(1/3*render.shape[1]), int(0.05*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 

            # out = np.concatenate((img, mesh_render, render),1).astype(np.uint8)
            # out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.05), int(out.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, 2)
            # video_out.write(out)
        
    # video_out.release()

            # if args.save_ply:
            #     face_upsampled = smpl_x.face_upsampled
            #     xyz = human_assets_refined['mean_3d'].detach().cpu().numpy()
            #     mesh = trimesh.Trimesh(xyz, face_upsampled, process=False)
            #     mesh.export(osp.join(cfg.vis_dir, f'{itr}.obj'))

if __name__ == "__main__":
    main()
