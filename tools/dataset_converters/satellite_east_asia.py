# -*- encoding: utf-8 -*-
'''
@File    :   satellite_east_asia.py
@Time    :   2024/04/20 17:31:27
@Author  :   GauthierLi 
@Version :   1.0
@Contact :   lwklxh@163.com
@License :   Copyright (C) 2024 GauthierLi, All rights reserved.
'''

'''
Description here ...
'''

import os
import mmcv
import argparse
import shutil
import tempfile
import zipfile
import numpy as np
import os.path as osp

from tqdm import tqdm
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert satellite dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='satellite folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'Satellite dataset Ⅱ (East Asia)')
    else:
        out_dir = args.out_dir
        
    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
        
    dataset_path = os.path.join(dataset_path, '1. The cropped image data and raster labels')
    datadir = ['train', 'test']
    
    for subdir in datadir:
        data_subpath = os.path.join(dataset_path, subdir)
        for image_path in tqdm(os.listdir(os.path.join(data_subpath, 'image'))):
            image = mmcv.imread(os.path.join(data_subpath, 'image', image_path))
            label = mmcv.imread(os.path.join(data_subpath, 'label', image_path))
            label = np.where(label==0, 1, label)
            label = np.where(label==255, 2, label)
            # import pdb; pdb.set_trace()
            image_save_path = os.path.join(out_dir, 'img_dir', subdir.replace('test', 'val'), image_path.replace('.tif', '.png'))
            label_save_path = os.path.join(out_dir, 'ann_dir', subdir.replace('test', 'val'), image_path.replace('.tif', '.png'))
            # import pdb; pdb.set_trace()
            mmcv.imwrite(image.astype('uint8')[:,:,0], image_save_path)
            mmcv.imwrite(label.astype('uint8')[:,:,0], label_save_path)
    print('=> Done!')
    
if __name__ == '__main__':
    args = parse_args()
    main()