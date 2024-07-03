import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import os
import csv

T_ITEM = Tuple[int, str, Tensor, Tensor, Tensor, Tensor]

# for img2shape
# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def mask2bbox(mask):
    # mask: w x h
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # return rmin, rmax, cmin, cmax
    return cmin, rmin, cmax, rmax

# ref: pix2vox: https://github.com/hzxie/Pix2Vox/blob/f1b82823e79d4afeedddfadb3da0940bcf1c536d/utils/data_transforms.py
def crop_square(img, bbox, img_size_h=256, img_size_w=256):
    # from pix2vox
    img_height, img_width, c = img.shape

    x0, y0, x1, y1 = bbox

    # Calculate the size of bounding boxes
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    bbox_x_mid = (x0 + x1) * .5
    bbox_y_mid = (y0 + y1) * .5

    # Make the crop area as a square
    square_object_size = max(bbox_width, bbox_height)
    x_left = int(bbox_x_mid - square_object_size * .5)
    x_right = int(bbox_x_mid + square_object_size * .5)
    y_top = int(bbox_y_mid - square_object_size * .5)
    y_bottom = int(bbox_y_mid + square_object_size * .5)

    # If the crop position is out of the image, fix it with padding
    pad_x_left = 0
    if x_left < 0:
        pad_x_left = -x_left
        x_left = 0
    pad_x_right = 0
    if x_right >= img_width:
        pad_x_right = x_right - img_width + 1
        x_right = img_width - 1
    pad_y_top = 0
    if y_top < 0:
        pad_y_top = -y_top
        y_top = 0
    pad_y_bottom = 0
    if y_bottom >= img_height:
        pad_y_bottom = y_bottom - img_height + 1
        y_bottom = img_height - 1

    # Padding the image and resize the image
    processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                mode='edge')
    
    pil_img = Image.fromarray(processed_image)
    pil_img = pil_img.resize((img_size_w, img_size_h))
    # processed_image = cv2.resize(processed_image, (img_size_w, img_size_h))

    return pil_img

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _transform_rgb(n_px):
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        Resize((n_px, n_px)),
    ])


class UDFs3d(Dataset):
    def __init__(self, name: str, root: Path, split: str, cond: str) -> None:
        super().__init__()

        self.root = str(root)
        self.ids = []
        self.npz_list = []
        self.name2text = {}
        self.text2name_all = {}
        self.text2name = {}
        self.cond = cond
        self.id2cat = {}
        self.cat2garment_type = {}
        self.name2npz = {}
        self.split = split
        self.name = name
        if 'sketch' in self.cond:
            self.clip_preprocess = _transform(224)
        elif 'img' in self.cond:
            print('here')
            self.clip_preprocess = _transform_rgb(224)

        if 'text' in self.cond:
            with open('./dataset/ShapeNet/text2shape/captions.tablechair_train.csv') as f:
                
                reader = csv.reader(f, delimiter=',')
                self.header = next(reader, None)
                data = [row for row in reader]
            for d in data:
                _, model_id, text, cat_i, synset, subSynsetId = d
                self.name2text[model_id] = text
                self.text2name_all[text] = model_id

        if 'category' in self.cond:
            with open('./dataset/Deepfashion3D/garment_type_list.txt', 'r') as f:
                data = f.readlines()
            for i, line in enumerate(data):
                line = line.rstrip()
                line = line.split(' ')
                for l in line[1:]:
                    self.id2cat[l] = i
                self.cat2garment_type[i] = line[0]
            print(self.cat2garment_type)

        if name in ['shapenet', 'deepfashion3d']:

            if name == 'deepfashion3d':
                self.data_root = os.path.join(self.root, 'udfs', 'train')
                self.sketch_root = os.path.join(self.root, 'images', 'train', 'sketch')
            else:
                self.data_root = os.path.join(self.root, 'train')
            ids = os.listdir(self.data_root)

            for id in ids:

                assert id.endswith(".npz")
                self.ids.append(id[:-4])
                self.npz_list.append(os.path.join(self.data_root, id))
        elif name == 'text2shape':
            data_root_chair = os.path.join(self.root, '03001627', 'train')
            ids_chair = os.listdir(data_root_chair)
            for id in ids_chair:
                #print(id)
                assert id.endswith(".npz")
                self.ids.append(id[:-4])
                self.npz_list.append(os.path.join(data_root_chair, id))

            data_root_table = os.path.join(self.root, '04379243', 'train')
            ids_table = os.listdir(data_root_table)
            for id in ids_table:
                #print(id)
                assert id.endswith(".npz")
                self.ids.append(id[:-4])
                self.npz_list.append(os.path.join(data_root_table, id))

            for npz_ in self.npz_list:
                self.name2npz[npz_.split('/')[-1][:-4]] = npz_

            for t in self.text2name_all.keys():
                if self.text2name_all[t] in self.ids:
                    self.text2name[t] = self.text2name_all[t]

            self.info_text = list(self.text2name.keys())


        elif name == 'pix3d':
            cats = os.listdir(os.path.join(self.root, 'udfs', split))
            for cat in cats:
                ids = os.listdir(os.path.join(self.root, 'udfs', split, cat))
                for id in ids:
                    self.ids.append(id[:-4])
                    self.npz_list.append(os.path.join(self.root, 'udfs', split, cat, id))

            self.img_root = os.path.join(self.root, 'images', 'train')
        


    def __len__(self) -> int:
        if 'text' in self.name:
            return (len(self.text2name.keys()))
        else:
            return len(self.ids)

    def __getitem__(self, index: int) -> T_ITEM:
        if not 'text' in self.name:
            item_id = self.ids[index]
        if 'sketch' in self.cond:            
            sketch_view_index = 0

            sketch_path = os.path.join(self.sketch_root, item_id, f'sketch_{sketch_view_index}.png')
            img = Image.open(sketch_path)
            img = self.clip_preprocess(img)
        elif 'img' in self.cond:
            cat = self.npz_list[index].split('/')[-2]
            all_imgs = os.listdir(os.path.join(self.img_root, cat, item_id))
            select_img = np.random.choice(all_imgs, 1)[0]
    
            img_path = os.path.join(self.img_root, cat, item_id, select_img)
            img_np = np.array(Image.open(img_path).convert('RGB'))
            mask_path = os.path.join(self.root, 'mask', cat, select_img.split('.')[0]+'.png')
            mask_np = np.array(Image.open(mask_path).convert('1'))
            
            x0, y0, x1, y1 = mask2bbox(mask_np)
            bbox = [x0, y0, x1, y1]
                
            try:
                img_clean = img_np * mask_np[:, :, None]
            except:
                print(img_path, mask_path)
            img_clean = crop_square(img_clean.astype(np.uint8), bbox)

            img = self.clip_preprocess(img_clean)


        if 'text' in self.name:
            text = self.info_text[index]
            item_id = self.text2name[text]
            npz = np.load(self.name2npz[item_id])
        else:
            npz = np.load(self.npz_list[index])
        pcd = torch.from_numpy(npz["pcd"])
        coords = torch.from_numpy(npz["coords"])
        labels = torch.from_numpy(npz["labels"])
        gradients = torch.from_numpy(npz["gradients"])

        if 'text' in self.cond:
            return index, item_id, pcd, coords, labels, gradients, text

        if 'sketch' in self.cond or 'img' in self.cond:
            return index, item_id, pcd, coords, labels, gradients, img

        if 'category' in self.cond:
            cat = self.id2cat[item_id.split('-')[0]]
            return index, item_id, pcd, coords, labels, gradients, cat

        return index, item_id, pcd, coords, labels, gradients

    def get_mesh(self, index: int) -> Tuple[Tensor, Tensor]:
        npz = np.load(self.root / f"{self.ids[index]}.npz")
        v = torch.from_numpy(npz["vertices"])
        t = torch.from_numpy(npz["triangles"])
        return v, t
    
