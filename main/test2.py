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

def move_dict_to_device(data, device):
    for key1 in data:
        if isinstance(data[key1], torch.Tensor):
            data[key1] = data[key1].to(device)
        if isinstance(data[key1], dict):
            for key2 in data[key1]:
                if isinstance(data[key1][key2], torch.Tensor):
                    data[key1][key2] =  data[key1][key2].to(device)
    return data

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id, args.fit_pose_to_test)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model2()
    # start_idx, end_idx = tester.testset.data_length['0']

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
        data = move_dict_to_device(data, "cuda:0")
        iter_start.record()
        # forward
        with torch.no_grad():
            out = tester.model2(data, 'test')

        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)
        times.append(elapsed)
        # save
        human_img_body = out['render_img_body'].cpu().numpy()
        human_img_total = out['img'].cpu().numpy()
       
        subject_id = str(data['subject_id'][0].item())
        frame_idx = int(data['frame_idx'][0].item())


        cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_humanbody.png'), human_img_body.transpose(1,2,0)[:,:,::-1]*255)
        cv2.imwrite(osp.join(save_root_path, subject_id + '_' + str("%05d"%frame_idx) + '_humantotal.png'), human_img_total.transpose(1,2,0)[:,:,::-1]*255)
    
    _time = np.mean(times[1:])
    print("Average time per iteration : {} ms".format(_time))
if __name__ == "__main__":
    main()
