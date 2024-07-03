import sys

sys.path.append("..")

from pathlib import Path
from typing import Any, Dict


import torch
from hesiod import hcfg, hmain
from torch import Tensor
from torch.utils.data import DataLoader

from AutoEncoder.data.dataset import UdfsDataset
from meshudf.meshudf import get_mesh_from_udf
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
from utils import get_o3d_mesh_from_tensors, progress_bar, random_point_sampling

import open3d as o3d 
import os
import numpy as np


import trimesh
from utils.utils import GridFiller

if len(sys.argv) != 2:
    print("Usage: python export_meshes.py <run_cfg_file>")
    exit(1)

@hmain(
    base_cfg_dir="../cfg/bases",
    run_cfg_file=sys.argv[1],
    parse_cmd_line=False,
    out_dir_root="../logs",
)
def main() -> None:
    seed = 10
    import random
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)


    ckpt_path = hcfg("ckpt_path", str)

    ckpt = torch.load(ckpt_path)

    latent_size = hcfg("latent_size", int)
    num_points_pcd = hcfg("num_points_pcd", int)
    udf_max_dist = hcfg("udf_max_dist", float)

    watertight = hcfg("watertight", bool)
    resolution = hcfg("resolution", int)

    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    encoder = encoder.cuda()
    encoder.eval()

    coords_encoder = CoordsEncoder()

    decoder_cfg = hcfg("decoder", Dict[str, Any])
    decoder = CbnDecoder(
        coords_encoder.out_dim,
        latent_size,
        decoder_cfg["hidden_dim"],
        decoder_cfg["num_hidden_layers"],
    )
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder = decoder.cuda()
    decoder.eval()


    dset_root = Path(hcfg("dset.root", str))
    name = hcfg("dset.name", str)

    bs = 1
    test_dset = UdfsDataset(name, dset_root, 'train') #deepfasion3d
    test_loader = DataLoader(test_dset, bs, num_workers=2, shuffle=False)


    for batch in progress_bar(test_loader, "Exporting"):

        _, item_ids, pcds, _, _, _ = batch
        
        bs = pcds.shape[0]
        pcds = pcds.cuda()
        
        pcds = random_point_sampling(pcds, num_points_pcd)
        

        with torch.no_grad():

            latent_codes = encoder(pcds)

        for i in progress_bar(range(bs), "Meshing"):

            lat = latent_codes[i].unsqueeze(0)

            def udf_func(c: Tensor) -> Tensor:

                c_ = coords_encoder.encode(c.unsqueeze(0))
                p = decoder(c_, lat).squeeze(0)
                p = torch.sigmoid(p)
                p = (1 - p) * udf_max_dist

                return p

            if watertight:
                size = resolution #256 faster
                fast_grid_filler = GridFiller(size)
                udf, gradients = fast_grid_filler.fill_grid(udf_func, max_batch=2**16)
                udf[udf < 0] = 0

                import mcubes
                vertices, faces = mcubes.marching_cubes(udf.detach().cpu().numpy(), 0.01)
                mesh = trimesh.Trimesh(vertices, faces)
                components = mesh.split(only_watertight=True)
                bbox = []
                for k, c in enumerate(components):
                    bbmin = c.vertices.min(0)
                    bbmax = c.vertices.max(0)
                    bbox.append((bbmax - bbmin).max())
                max_component = np.argmax(bbox)
                mesh = components[max_component]
                mesh.vertices = mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]
                mesh_path = f"./outputs/AE/{hcfg('dset.exp_name', str)}/meshes_test/{item_ids[i]}mc.obj"
                os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
                trimesh.Trimesh(vertices=vertices, faces=faces).export(mesh_path)

            else:
                v, t, udf, gradients = get_mesh_from_udf(
                udf_func,
                coords_range=(-1, 1),
                max_dist=udf_max_dist,
                N=resolution,
                max_batch=2**16,
                differentiable=False,
                use_fast_grid_filler=True
                )
                pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)
                mesh_path = f"./outputs/AE/{hcfg('dset.exp_name', str)}/meshes_test/{item_ids[i]}_meshudf.obj"
                os.makedirs(os.path.dirname(mesh_path), exist_ok=True)

                o3d.io.write_triangle_mesh(str(mesh_path), pred_mesh_o3d)
    

if __name__ == "__main__":
    main()
