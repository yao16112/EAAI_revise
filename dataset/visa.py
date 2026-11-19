import json
import os
import random
import csv
import numpy as np
import torch.utils.data as data
from PIL import Image
from .base import BaseDataset

'''dataset source: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'''

VISA_CLS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum',
]

from config import DATASET_ROOTS
VISA_ROOT = DATASET_ROOTS['visa']


class VisaSolver(object):
    """VisA 专用的数据解析器，读取 meta.json"""
    def __init__(self, root, clsnames):
        self.root = root
        self.CLSNAMES = clsnames
        self.meta_path = os.path.join(root, 'meta.json')

    def run(self):
        if not os.path.exists(self.meta_path):
            return self._run_from_split_csv()

        # 读取 meta.json
        with open(self.meta_path, 'r') as f:
            meta_info = json.load(f)
        
        # 只返回指定类别的数据
        info_required = {'train': {}, 'test': {}}
        
        for cls in self.CLSNAMES:
            if cls in meta_info.get('train', {}):
                info_required['train'][cls] = meta_info['train'][cls]
            else:
                info_required['train'][cls] = []
                
            if cls in meta_info.get('test', {}):
                info_required['test'][cls] = meta_info['test'][cls]
            else:
                info_required['test'][cls] = []
        
        return info_required

    def _run_from_split_csv(self):
        split_path = os.path.join(self.root, 'split_csv', '1cls.csv')
        info_required = {'train': {}, 'test': {}}
        for cls in self.CLSNAMES:
            info_required['train'][cls] = []
            info_required['test'][cls] = []

        with open(split_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row['category']
                if cls not in self.CLSNAMES:
                    continue
                split = row['split']
                label = row['label']
                anomaly = 0 if label == 'normal' else 1
                info_required[split][cls].append({
                    'img_path': row['image'],
                    'mask_path': row.get('mask') or '',
                    'cls_name': cls,
                    'specie_name': label,
                    'anomaly': anomaly,
                })

        return info_required


class VisaDataset(BaseDataset):
    def __init__(self, class_names, transform, target_transform, synthesis_anomalies=True,
                 root=VISA_ROOT, training=True, white_noise=False,
                 mode='FS',
                 k_shot=5):
        # 不调用父类的 __init__，自己处理
        assert mode in ['FS', 'ZS']

        self.mode = mode
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.k_shot = k_shot
        self.data_all = []
        self.training = training
        self.white_noise = white_noise
        self.synthesis_anomalies = synthesis_anomalies
        self.class_names = class_names

        # 使用 VisA 专用的 Solver
        self.solver = VisaSolver(root, class_names)
        meta_info = self.solver.run()

        if self.mode == 'ZS':
            meta_info = meta_info['test']
        else:
            meta_info = self._select_normal_from_train(meta_info['train'], k_shot) if training \
                else meta_info['test']

        self.meta_info = meta_info

        # 收集数据
        if meta_info and meta_info != 0:
            for cls_name in self.class_names:
                if cls_name in meta_info:
                    self.data_all.extend(meta_info[cls_name])

        self._count_number(self.data_all)
        self.length = len(self.data_all)

        # 异常源图像路径（用于合成异常）
        from config import DATA_ROOT
        import glob
        anomaly_source_path = os.path.join(DATA_ROOT, 'dtd', 'images')
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(anomaly_source_path, "**/*.jpg"), recursive=True))

        import imgaug.augmenters as iaa
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]

    def _select_normal_from_train(self, meta_info: dict, k_shot):
        normal_info = {}
        for cls in meta_info.keys():
            normal_info[cls] = []
            for info in meta_info[cls]:
                anomaly = info['anomaly']
                if anomaly == 0:
                    normal_info[cls].append(info)

            current_len = len(normal_info[cls])
            if current_len == 0:
                print(f"Warning: No training data found for class {cls}")
                continue

            k = min(current_len, k_shot) if k_shot > 0 else current_len
            if k == 0:
                return 0

            normal_info[cls] = random.sample(normal_info[cls], k)
        return normal_info

    def _count_number(self, data):
        normal_number = 0
        abnormal_number = 0
        for _data in data:
            anomaly = _data['anomaly']
            if anomaly == 0:
                normal_number += 1
            else:
                abnormal_number += 1
        print(f'Training: {self.training}, Normal: {normal_number}, Abnormal: {abnormal_number}')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']

        full_img_path = os.path.join(self.root, img_path)
        img = Image.open(full_img_path).convert('RGB')

        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            full_mask_path = os.path.join(self.root, mask_path)
            if mask_path and os.path.exists(full_mask_path):
                img_mask = np.array(Image.open(full_mask_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')

        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask

        return {
            'img': img,
            'img_mask': img_mask,
            'specie_name': specie_name,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': full_img_path
        }
