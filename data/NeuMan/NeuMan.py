import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.flame import flame
from utils.preprocessing import load_img, get_bbox, parse_outfits
from utils.transforms import transform_joint_to_other_db
from pytorch3d.transforms import quaternion_to_matrix
import json
import torchvision
import os


class NeuMan(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.data_split = data_split
        self.root_path = osp.join('..', 'data', 'NeuMan', 'data')
        outfits = parse_outfits(cfg)
        self.data_dirs = {outfit: osp.join(self.root_path, outfit) for outfit in outfits['seen'].keys()}
        self.transform = transform
        self.data = []
        self.smplx_params = []
        self.data_length = {}
        self.load_data()
        

    def load_data(self):
        start_idx = 0
        for subject_id, (outfit, outfit_datadir) in enumerate(self.data_dirs.items()):
            self.load_id_info(outfit_datadir)

            print('Loading {}..'.format(outfit))
            # read split file
            if cfg.fit_pose_to_test or self.data_split=='test' : #or self.data_split=='test'
                split_path = osp.join(outfit_datadir, 'test_split.txt')
            else:
                split_path = osp.join(outfit_datadir, 'train_split.txt')
            with open(split_path) as f:
                frame_idx_list = [int(x[:-5]) for x in f.readlines()]

            data_length = len(frame_idx_list)
            end_idx = start_idx + data_length
            self.data_length[str(subject_id)] = (start_idx, end_idx)
            print('{} {}: {} {}'.format(subject_id, outfit, start_idx, end_idx))
            start_idx = end_idx

            # load cameras
            cam_params = {}
            with open(osp.join(outfit_datadir, 'sparse', 'cameras.txt')) as f:
                lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                splitted = line.split()
                _, _, width, height, focal_x, focal_y, princpt_x, princpt_y = splitted
            focal = np.array((float(focal_x), float(focal_y)), dtype=np.float32) # shared across all frames
            princpt = np.array((float(princpt_x), float(princpt_y)), dtype=np.float32) # shared across all frames
            with open(osp.join(outfit_datadir, 'sparse', 'images.txt')) as f:
                lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                if 'png' not in line:
                    continue
                splitted = line.split()
                frame_idx = int(splitted[-1][:-4])
                qw, qx, qy, qz = float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])
                tx, ty, tz = float(splitted[5]), float(splitted[6]), float(splitted[7])
                R = quaternion_to_matrix(torch.FloatTensor([qw, qx, qy, qz])).numpy()
                t = np.array([tx, ty, tz], dtype=np.float32)
                cam_params[frame_idx] = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}
            # self.cam_params.append(cam_params)

            # load image paths
            img_paths = {}
            img_path_list = glob(osp.join(outfit_datadir, 'images', '*.png'))
            for img_path in img_path_list:
                frame_idx = int(img_path.split('/')[-1][:-4])
                img_paths[frame_idx] = img_path
            # self.img_paths.append(img_paths)

            # load mask paths
            mask_paths = {}
            mask_path_list = glob(osp.join(outfit_datadir, 'masks', '*.png'))
            for mask_path in mask_path_list:
                frame_idx = int(mask_path.split('/')[-1][:-4])
                mask_paths[frame_idx] = mask_path
            # self.mask_paths.append(mask_paths)

            parse_paths = {}
            parse_path_list = glob(osp.join(outfit_datadir, 'seg_class', 'videos', '*.npy'))
            for parse_path in parse_path_list:
                frame_idx = int(parse_path.split('/')[-1][:-4])
                parse_paths[frame_idx] = parse_path

            # normal_paths = {}
            # normal_path_list = glob(osp.join(outfit_datadir, 'normal', 'sapiens_1b', '*.png'))
            # for normal_path in normal_path_list:
            #     frame_idx = int(normal_path.split('/')[-1][:-4])
            #     normal_paths[frame_idx] = normal_path

            # load keypoints
            kpts = {}
            kpt_path_list = glob(osp.join(outfit_datadir, 'keypoints_whole_body', '*.json'))
            for kpt_path in kpt_path_list:
                frame_idx = int(kpt_path.split('/')[-1][:-5])
                with open(kpt_path) as f:
                    kpts[frame_idx] = np.array(json.load(f), dtype=np.float32)
            # self.kpts.append(kpts)

            for idx, frame_idx in enumerate(frame_idx_list):
                img = load_img(img_paths[frame_idx])
                img = self.transform(img.astype(np.float32))/255.

                # normal = load_img(normal_paths[frame_idx])
                # normal = self.transform(normal.astype(np.float32))/255.

                # load mask
                mask = cv2.imread(mask_paths[frame_idx])[:,:,0,None] / 255.
                mask = self.transform((mask > 0.5).astype(np.float32))

                parse_label = np.load(parse_paths[frame_idx])
                parse_mask = torch.from_numpy(self.one_hot(parse_label)).permute(2,0,1)
                part_mask = torch.from_numpy(self.get_partition(parse_label))

                # get bbox from 2D keypoints
                joint_img = kpts[frame_idx][:,:2]
                joint_valid = (kpts[frame_idx][:,2:] > 0.5).astype(np.float32)
                bbox = get_bbox(joint_img, joint_valid[:,0])

                self.data.append({'img': img, 'mask': mask, 'bbox': bbox,\
                                   'cam_param': cam_params[frame_idx], 'frame_idx': frame_idx, \
                                      'subject_id': subject_id, 'semantic_gt': parse_mask, 'part_mask': part_mask})

            # load smplx parameters
            smplx_params = {}
            smplx_param_path_list = glob(osp.join(outfit_datadir, 'smplx_optimized', 'smplx_params', '*.json'))
            for smplx_param_path in smplx_param_path_list:
                file_name = smplx_param_path.split('/')[-1]
                frame_idx = str(int(file_name[:-5]))
                with open(smplx_param_path) as f:
                    smplx_params[frame_idx] = {k: torch.FloatTensor(v) for k,v in json.load(f).items()}
            self.smplx_params.append(smplx_params)

        if self.data_split == 'train':
            self.data *= 100
    
    def load_id_info(self, path):
        with open(osp.join(path, 'smplx_optimized', 'shape_param.json')) as f:
            shape_param = torch.FloatTensor(json.load(f))
        with open(osp.join(path, 'smplx_optimized', 'face_offset.json')) as f:
            face_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(path, 'smplx_optimized', 'joint_offset.json')) as f:
            joint_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(path, 'smplx_optimized', 'locator_offset.json')) as f:
            locator_offset = torch.FloatTensor(json.load(f))
        smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

        texture_path = osp.join(path, 'smplx_optimized', 'face_texture.png')
        texture = torch.FloatTensor(cv2.imread(texture_path)[:,:,::-1].copy().transpose(2,0,1))/255
        texture_mask_path = osp.join(path, 'smplx_optimized', 'face_texture_mask.png')
        texture_mask = torch.FloatTensor(cv2.imread(texture_mask_path).transpose(2,0,1))/255
        flame.set_texture(texture, texture_mask)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def one_hot(self, label):

        H, W = label.shape
        result = np.zeros((H, W, 5), dtype=np.float32)
        # index_mappings = {0: 0, 1: 1, 2: 1, 5: 2, 6: 2, 7: 2, 9: 3, 12: 3, 8: 4, 18: 4, \
        #                   19: 4, 3: 5, 4: 5, 10: 5, 11: 5, 13: 5, 14: 5, 15: 5, 16: 5, 17: 5}
        
        index_mappings = {3: 0, 4: 0, 10: 0, 11: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, \
                          1: 1, 2: 1, \
                          5: 2, 6: 2, 7: 2, \
                          9: 3, 12: 3,\
                          8: 4, 18: 4, 19: 4}

        valid_classes = list(index_mappings.keys())
        mask = np.isin(label, valid_classes)
        rows, cols = np.where(mask)
        classes = label[mask]
        indices = np.array([index_mappings[c] for c in classes])
        result[rows, cols, indices] = 1.0
        return result
    
    def get_partition(self, label):
        target_id0 = [3,4,10,11,13,14,15,16,17]
        target_id1 = [1, 2]
        target_id2 = [5,6,7]
        target_id3 = [9,12]
        target_id4 = [8,18,19]
        
        binary_mask0 = np.isin(label, target_id0).astype(np.uint8)
        binary_mask1 = np.isin(label, target_id1).astype(np.uint8)
        binary_mask2 = np.isin(label, target_id2).astype(np.uint8)
        binary_mask3 = np.isin(label, target_id3).astype(np.uint8)
        binary_mask4 = np.isin(label, target_id4).astype(np.uint8)
        binary = np.stack([binary_mask0, binary_mask1, binary_mask2, binary_mask3, binary_mask4], axis=0)
        return binary




body_parse=['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
            'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
            'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']