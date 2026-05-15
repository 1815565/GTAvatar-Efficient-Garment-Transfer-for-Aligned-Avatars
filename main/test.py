import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Tester
import os
import os.path as osp
import cv2
from utils.smpl_x import smpl_x
from utils.vis import render_mesh
from pytorch3d.io import save_obj
import trimesh
from PIL import Image as PILImage
from PIL import Image
from obj_io import save_gaussians_as_ply
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

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id, args.fit_pose_to_test)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    # start_idx, end_idx = tester.testset.data_length['0']
    palette = get_palette(6)
    for batch in tester.batch_generator:
        # batch 应该是一个字典，其中包含了 'img' 等键
        render_shape = batch['img'].shape[2:]
        break  # 如果你只需要第一个批次，使用 break 跳出循环
    print(render_shape)
    save_root_path = cfg.result_dir
    os.makedirs(save_root_path, exist_ok=True)
    # video_out = cv2.VideoWriter(save_root_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 7, (render_shape[1]*3, render_shape[0]))
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    times = []
    for itr, data in enumerate(tqdm(tester.batch_generator)):
        # if itr < start_idx: continue
        # if itr >= end_idx: break
        iter_start.record()
        # forward
        with torch.no_grad():
            out, human_assets = tester.model(data, 'test', epoch=5)

        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)
        times.append(elapsed)
        # save
        human_img_refined = out['human_img_refined'].cpu().numpy()
        human_img = out['human_img'].cpu().numpy()
        partition_refined = out['partition'].cpu().numpy()
        # semantic_img = out['semantic_img'].cpu().numpy()
        # semantic_img_collect = out['semantic_img_collect'].cpu().numpy()
        
        # mask = (out['alpha']>0.5).cpu().numpy()
        # mask_parts = (out['alpha_partition']>0.5).cpu().numpy()

        # mask_reshaped = mask[:, 0, :, :]  
        # semantic_img[mask_reshaped] += 1  
        # mask_parts_reshaped = mask_parts[:, 0, :, :]  
        # semantic_img_collect[mask_parts_reshaped] += 1
        
        # sd_mask = out['sd_mask'].cpu().numpy()
        # smplx_verts = out['smplx_mesh'].cpu().numpy()

        batch_size = human_img_refined.shape[0]
        for i in range(batch_size):
            subject_id = str(data['subject_id'][i].item())
            frame_idx = int(data['frame_idx'][i].item())
            
            cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_gt.png'), data['img'][i].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)

            # cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_human_refined.png'), human_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_human.png'), human_img[i].transpose(1,2,0)[:,:,::-1]*255)

            # output_im = PILImage.fromarray(np.asarray(semantic_img[i], dtype=np.uint8))            
            # output_im.putpalette(palette)  # 重新应用调色板（确保一致性）
            # output_im.save(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + 'semantic.png'))
           
           
           # # mesh = out['smplx_mesh'][i]
            # mesh_render = render_mesh(mesh, smpl_x.face, {k: v[i].cuda() for k,v in data['cam_param'].items()}, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)

            # img = cv2.imread(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_gt.png'))
            # render = (human_img_refined[i].transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
            # font_size = 1.5
            # thick = 3
            # cv2.putText(img, 'image', (int(1/3*img.shape[1]), int(0.05*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
            # cv2.putText(mesh_render, 'rendered SMPL-X mesh', (int(1/5*mesh_render.shape[1]), int(0.05*mesh_render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2)
            # cv2.putText(render, 'render', (int(1/3*render.shape[1]), int(0.05*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 

            # out = np.concatenate((img, mesh_render, render),1).astype(np.uint8)
            # out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.05), int(out.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, 2)
            # video_out.write(out)
            for j in range(5):
                cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + f'_part{j}.png'), partition_refined[j].transpose(1,2,0)[:,:,::-1]*255)
                # output_im = PILImage.fromarray(np.asarray(semantic_img_collect[j], dtype=np.uint8))
                # output_im.putpalette(palette)
                # output_im.save(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + f'_sem_part{j}.png'))

            if args.save_ply:
                # face_upsampled = smpl_x.face_upsampled
                # xyz = human_assets_refined['mean_3d'].detach().cpu().numpy()
                # mesh = trimesh.Trimesh(xyz, face_upsampled, process=False)
                # mesh.export(osp.join(cfg.vis_dir, f'{itr}.obj'))
                gaussian_vals = {
                    'mean_3d': human_assets['mean_3d'][i],
                    'rgb': human_assets['rgb'][i],
                    'opacity': human_assets['opacity'][i],
                    'scale': human_assets['radius'][i],
                    'rotation': human_assets['rotation'][i],
                }
                save_gaussians_as_ply(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_human.ply'), gaussian_vals)

    # video_out.release()
    _time = np.mean(times[1:])
    print("Average time per iteration : {} ms".format(_time))
if __name__ == "__main__":
    main()
