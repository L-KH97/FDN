import os

import cv2
import numpy as np
import json
import random

from src.datahandler.denoise_dataset import DenoiseDataSet
from src.datahandler.tools import *
from . import regist_dataset


@regist_dataset
class UNIT(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.template_aug = Augmentation(4, 0.05, 0.0, 0.0, 1.0)
        self.search_aug = Augmentation(64, 0.18, 0.0, 0.0, 1.0)
        self.anchor_target = AnchorTarget()

    def _scan(self):
        # check if the dataset exists
        dataset_path = ''
        anno_path = ''
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path
        
        with open(anno_path, 'r') as f:
            self.meta_data = json.load(f)

        # scan all image path & info in dataset path
        for folder_name in os.listdir(dataset_path):
            # parse folder name of each shot
            n_img = len(os.listdir(os.path.join(dataset_path, folder_name)))
            assert n_img%2==0, 'Datasets should be paired'

            x_list = []
            z_list = []

            for img_name in os.listdir(os.path.join(dataset_path, folder_name)):
                if 'z.jpg' in img_name:
                    z_list.append(img_name)
                elif 'x.jpg' in img_name:
                    x_list.append(img_name)
            
            random.shuffle(z_list)
            random.shuffle(x_list)


            for img_z, img_x in list(zip(z_list, x_list)):
                info = {}
                info['img_seq'] = folder_name
                info['img_z_path'] = os.path.join(dataset_path, folder_name, img_z)
                info['img_x_path'] = os.path.join(dataset_path, folder_name, img_x)
                info['img_z_idx'] = img_z.split('.')[0]
                info['img_x_idx'] = img_x.split('.')[0]
                self.img_paths.append(info)



    def _load_data(self, data_idx):
        file_info = self.img_paths[data_idx]

        template_image = cv2.imread(file_info['img_z_path'])
        search_image = cv2.imread(file_info['img_x_path'])
        
        template_box = self._get_bbox(template_image, self.meta_data[file_info['img_seq']]['00'][file_info['img_z_idx']])
        search_box = self._get_bbox(search_image, self.meta_data[file_info['img_seq']]['00'][file_info['img_x_idx']])
        
        template, _ = self.template_aug(template_image, template_box, 127)
        
        search, bbox = self.search_aug(search_image, search_box, 287)

        cls, delta, delta_weight, overlap = self.anchor_target(bbox, 21)

        return {'real_noisy_template': self._load_img_from_np(template),
                'real_noisy_search': self._load_img_from_np(search),
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bbox)}
    
    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = 127
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox
