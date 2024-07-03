import sys

sys.path.append("..")

from pathlib import Path
import os

import numpy as np
from utils import (
    compute_udf_from_mesh,
    get_o3d_mesh_from_tensors,
    get_tensor_pcd_from_o3d,
    progress_bar,
    read_mesh,
)

np.random.seed(1024)

cat2id = {
    "chair": "03001627",
    "bench": "02828884",
    "cabinet": "02933112",
    "car": "02958343",
    "airplane": "02691156",
    "display": "03211117",
    "lamp": "03636649",
    "speaker": "03691459",
    "rifle": "04090263",
    "sofa": "04256520",
    "table": "04379243",
    "phone": "04401088",
    "watercraft": "04530566"
}

data_root = sys.argv[1]
out_dir = sys.argv[2]
name = sys.argv[3]
assert name in ['shapenet', 'deepfashion3d', 'pix3d']
if name == 'shapenet':
    if len(sys.argv) != 5:
        print("Usage: python preprocess_udfs.py </path/to/meshes> </out/path> <name of dataset> <category>")
        exit(1)
    category = sys.argv[4]

    assert category in cat2id.keys()
    id = cat2id[category]
    out_dir = os.path.join(out_dir, id)
    sub_ids = os.listdir(os.path.join(data_root, id))
else:
    if len(sys.argv) != 4:
        print("Usage: python preprocess_udfs.py </path/to/meshes> </out/path> <name of dataset> <category>")
        exit(1)
    

os.makedirs(out_dir, exist_ok=True)


if name == 'shapenet':
    lst_dir = '../dataset_info_files/ShapeNet_filelists'

    with open(lst_dir+"/"+str(id)+"_test.lst", "r") as f:
        list_obj_test = f.readlines()

    with open(lst_dir+"/"+str(id)+"_train.lst", "r") as f:
        list_obj_train = f.readlines()

    list_obj_test = [f.rstrip() for f in list_obj_test]
    list_obj_train = [f.rstrip() for f in list_obj_train]

elif name == 'deepfashion3d':
    lst_dir = '../dataset_info_files/Deepfashion3d'

    with open(lst_dir+"/"+"deepfashion3d_test.txt", "r") as f:
        list_obj_test = f.readlines()

    with open(lst_dir+"/"+"deepfashion3d_train.txt", "r") as f:
        list_obj_train = f.readlines()

    list_obj_test = [f.rstrip('\n') for f in list_obj_test]
    list_obj_train = [f.rstrip('\n') for f in list_obj_train]
    print(len(list_obj_train), len(list_obj_train))
    check_dict = {}
    for id in list_obj_train:
        if check_dict.__contains__(id):
            print(id)
        else:
            check_dict[id] = 1

elif name == 'pix3d':
    list_obj_train = []
    list_obj_test = []

    data_root = './dataset/pix3d/models'

    cats_train = os.listdir(os.path.join(data_root, 'train'))
    for cat in cats_train:
        ids = os.listdir(os.path.join(data_root, 'train', cat))
        for id in ids:
            model_path = os.path.join(data_root, 'train', cat, id, 'model.obj')
            list_obj_train.append(model_path)

    cats_test = os.listdir(os.path.join(data_root, 'test'))
    for cat in cats_test:
        ids = os.listdir(os.path.join(data_root, 'test', cat))
        for id in ids:
            model_path = os.path.join(data_root, 'test', cat, id, 'model.obj')
            list_obj_test.append(model_path)


def PrepareOneUDF(sub_id, split):
    if name == 'shapenet':
        mesh_path = os.path.join(data_root, id, sub_id, 'model.obj')

    elif name == 'deepfashion3d':
        mesh_path = os.path.join(data_root, f'{sub_id}.obj')

    elif name == 'pix3d':
        mesh_path = os.path.join(data_root, f'{sub_id}.obj')


    out_dir_split = os.path.join(out_dir, split)

    os.makedirs(out_dir_split, exist_ok=True)

    v, t = read_mesh(mesh_path)
    mesh_o3d = get_o3d_mesh_from_tensors(v, t)

    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=100_000)
    pcd = get_tensor_pcd_from_o3d(pcd_o3d)[:, :3]

    coords, labels, gradients = compute_udf_from_mesh(
        mesh_o3d,
        num_queries_on_surface=250_000,
        num_queries_per_std=[250_000, 200_000, 25_000, 25_000],

    )

    if name == 'pix3d':
        out_file = os.path.join(out_dir_split, sub_id.split('/')[-3], f"{sub_id.split('/')[-2]}.npz")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    else:
        out_file = os.path.join(out_dir_split, f"{sub_id}.npz")


    print('train: ', len(list_obj_train), 'test: ', len(list_obj_test))
    print(out_file)
    np.savez(
        out_file,
        vertices=v.numpy(),
        triangles=t.numpy(),
        pcd=pcd.numpy(),
        coords=coords.numpy(),
        labels=labels.numpy(),
        gradients=gradients.numpy(),
    )


for sub_id in progress_bar(list_obj_train):
    PrepareOneUDF(sub_id, 'train')

for sub_id in progress_bar(list_obj_test):
    PrepareOneUDF(sub_id, 'test')
