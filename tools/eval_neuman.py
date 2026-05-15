# use original images (without crop and resize) following https://github.com/aipixel/GaussianAvatar/blob/main/eval.py

import cv2
import torch
import os.path as osp
import os
from glob import glob
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse
import sys

cur_dir = osp.dirname(os.path.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..', 'main'))
from config import cfg
from utils.preprocessing import load_img, get_bbox, parse_outfits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, dest='output_path')
    parser.add_argument('--include_bkg', dest='include_bkg', action='store_true')
    args = parser.parse_args()
    assert args.output_path, "Please set output_path."
    return args

# get path
args = parse_args()
output_path = args.output_path
include_bkg = args.include_bkg


psnr = PeakSignalNoiseRatio(data_range=1).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()


root_path = osp.join('..', 'data', cfg.dataset, 'data')
outfits = parse_outfits(cfg)
data_dirs = {outfit: osp.join(root_path, outfit) for outfit in outfits['seen'].keys()}
for subject_id, (outfit, outfit_datadir) in enumerate(data_dirs.items()):
    results = {'psnr': [], 'ssim': [], 'lpips': []}
    with open(osp.join(outfit_datadir, 'test_split.txt')) as f: #
        lines = f.readlines()
    for line in lines:
        frame_idx = int(line[:-5])

        # output image
        out_path = osp.join(output_path,  str(subject_id) + '_' + str("%05d"%frame_idx) +'_human_refined.png') #str(subject_id) + '_' +    # 0_scene_human_refined_composed #human_refined.png
        # out = cv2.imread(out_path)[:,:,::-1]/255.
        out = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
        
        
        # gt image
        gt_path = osp.join(outfit_datadir, 'images', '%05d.png' % frame_idx)
        # gt = cv2.imread(gt_path)[:,:,::-1]/255.
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        
        # gt mask
        mask_path = osp.join(outfit_datadir, 'masks', '%05d.png' % frame_idx)
        # mask = cv2.imread(mask_path)
        # mask = 1 - mask/255. # 0: bkg, 1: human
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask = torch.FloatTensor(mask).permute(2,0,1)[None,:,:,:].cuda()
        mask = mask != 0
        
        
        # exclude background pixels
        if not include_bkg:
            # out = out * mask + 1 * (1 - mask)
            # gt = gt * mask + 1 * (1 - mask)
            out[~mask] = 255.
            out = out / 255.
            gt[~mask] = 255.
            gt = gt / 255.

        out = torch.FloatTensor(out).permute(2,0,1)[None,:,:,:].cuda()
        gt = torch.FloatTensor(gt).permute(2,0,1)[None,:,:,:].cuda()

        out = out.clamp(0.0, 1.0)
        gt = gt.clamp(0.0, 1.0)

        results['psnr'].append(psnr(out, gt))
        results['ssim'].append(ssim(out, gt))
        results['lpips'].append(lpips(out*2-1, gt*2-1)) # normalize to [-1,1]

    print('output path: ' + output_path)
    print('subject: ' + outfit)
    print('include_bkg: ' + str(include_bkg))
    print({k: torch.FloatTensor(v).mean() for k,v in results.items()})

