import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import os

T_ITEM = Tuple[int, str, Tensor, Tensor, Tensor, Tensor]


class UdfsDataset(Dataset):
    def __init__(self, name: str, root: Path, split: str) -> None:
        super().__init__()

        self.root = str(root)
        self.ids = []
        self.npz_list = []
        self.id2category = {}
        self.training_idxes = []
        self.name = name
        

        if name in ['shapenet', 'deepfashion3d']:
            self.data_root = os.path.join(self.root, 'train')
            ids = os.listdir(self.data_root)
            print(split, len(ids))
            for id in ids:
                #print(id)
                assert id.endswith(".npz")
                self.ids.append(id[:-4])
                self.npz_list.append(os.path.join(self.data_root, id))
        elif 'text2shape' in name:
            data_root_chair = os.path.join(self.root, '03001627', 'train')
            ids_chair = os.listdir(data_root_chair)

            data_root_table = os.path.join(self.root, '04379243', 'train')
            ids_table = os.listdir(data_root_table)

            for id in ids_chair:
                self.ids.append(id[:-4])
                self.npz_list.append(os.path.join(data_root_chair, id))

            for id in ids_table:
                self.ids.append(id[:-4])
                self.npz_list.append(os.path.join(data_root_table, id))

            self.ids = sorted(self.ids)
            self.npz_list = sorted(self.npz_list)


        elif name == 'pix3d':
            cats = os.listdir(os.path.join(self.root, split))
            for cat in cats:
                ids = os.listdir(os.path.join(self.root, split, cat))
                for id in ids:
                    self.ids.append(id[:-4])
                    self.npz_list.append(os.path.join(self.root, split, cat, id))


    def __len__(self) -> int:
        return len(self.ids)

    def get_training_idxes(self):
        return self.training_idxes

    def update_training_idxes(self, new_idxes):
        self.training_idxes = self.training_idxes + new_idxes
        with open('./training_idxes.txt', 'w') as f:
            for info in self.training_idxes:
                f.write(f'{info}\n')
    
    def val_del_idxes(self):
        self.ids
        self.npz_list

    def __getitem__(self, index: int) -> T_ITEM:

        item_id = self.npz_list[index].split('/')[-1][:-4]
        npz = np.load(self.npz_list[index])
        pcd = torch.from_numpy(npz["pcd"])
        coords = torch.from_numpy(npz["coords"])
        labels = torch.from_numpy(npz["labels"])
        gradients = torch.from_numpy(npz["gradients"])

        return index, item_id, pcd, coords, labels, gradients

    def get_mesh(self, index: int) -> Tuple[Tensor, Tensor]:
        npz = np.load(self.root / f"{self.ids[index]}.npz")
        v = torch.from_numpy(npz["vertices"])
        t = torch.from_numpy(npz["triangles"])

        return v, t
