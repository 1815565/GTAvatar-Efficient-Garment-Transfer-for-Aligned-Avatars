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

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    # start_idx, end_idx = tester.testset.data_length['0']

    tester.model.module.human_gaussian.init_()
    exit()

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    times = []
    
    save_root_path = cfg.result_dir
    os.makedirs(save_root_path, exist_ok=True)
 
    for itr, data in enumerate(tqdm(tester.batch_generator)):
        # if itr < start_idx: continue
        # if itr >= end_idx: break
        iter_start.record()
        # forward
        with torch.no_grad():
            out = tester.model.module.quick_forward(data)

        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)
        times.append(elapsed)

        human_img = out['human_img'].cpu().numpy()

        batch_size = human_img.shape[0]
        for i in range(batch_size):
            subject_id = str(data['subject_id'][i].item())
            frame_idx = int(data['frame_idx'][i].item())

            cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_gt.png'), data['img'][i].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_human.png'), human_img[i].transpose(1,2,0)[:,:,::-1]*255)

            if args.save_ply:
                face_upsampled = smpl_x.face_upsampled
                xyz = human_assets_refined['mean_3d'].detach().cpu().numpy()
                mesh = trimesh.Trimesh(xyz, face_upsampled, process=False)
                mesh.export(osp.join(cfg.vis_dir, f'{itr}.obj'))
    _time = np.mean(times[1:])
    print("Average time per iteration : {} ms".format(_time))
if __name__ == "__main__":
    main()
