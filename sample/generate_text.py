
import os

import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from models.cfg_sampler import ClassifierFreeSampleModel

from AutoEncoder.models.coordsenc import CoordsEncoder
from AutoEncoder.models.cbndec import CbnDecoder
from meshudf.meshudf import get_mesh_from_udf
from utils.utils import get_o3d_mesh_from_tensors

import trimesh
import open3d as o3d
from PIL import Image
from torch import Tensor
from utils.utils import GridFiller
import pymeshlab as ml

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip


def load_and_freeze_clip(clip_version):
    clip_model, _ = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    # clip.model.convert_weights(
    #     clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def main():
    args = generate_args()

    out_path = args.output_dir
    os.makedirs(out_path, exist_ok=True)

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    print(args.guidance_param)
    if args.guidance_param != 1: ## 3 for text
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)


    model.to(dist_util.dev())
    model.eval()  # disable random masking

    print("You are running conditional generaion")

    import csv
    text2name = {}

    cond = {}
    cond['y'] = {}
    

    ckpt = torch.load(args.ae_dir)
    print(f'Load AutoEncoder From: {args.ae_dir}')

    latent_size = 64
    coords_encoder = CoordsEncoder()

    hidden_dim = 512
    num_hidden_layers = 5
    decoder = CbnDecoder(
        coords_encoder.out_dim,
        latent_size,
        hidden_dim,
        num_hidden_layers,
    )
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder = decoder.cuda()
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    cond['y']['text'] = [args.prompt]*args.batch_size
    # for name in names:
    print("You are running conditional generaion")
    print("Conidtion sample: ", cond['y']['text'][0])

    args.batch_size = len(cond['y']['text'])
    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model,
        (
        args.batch_size, 1, latent_size),
        clip_denoised=False,
        model_kwargs=cond,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    bs = len(cond['y']['text'])
    udf_max_dist = 0.1
    for k in range(bs):
        id = k
        lat = sample[k]
        def udf_func(c: Tensor) -> Tensor:
            c = coords_encoder.encode(c.unsqueeze(0))
            p = decoder(c, lat).squeeze(0)
            p = torch.sigmoid(p)
            p = (1 - p) * udf_max_dist
            return p

        mesh_path = os.path.join(args.output_dir, cond['y']['text'][k].replace(" ", "-").replace(".", "")[:100]+f'_{id}.obj')

        if args.watertight:
            size = args.resolution
            fast_grid_filler = GridFiller(size)
            udf, _ = fast_grid_filler.fill_grid(udf_func, max_batch=2**16)
            udf[udf < 0] = 0

            # keep the max component of the extracted mesh
            import mcubes
            vertices, faces = mcubes.marching_cubes(udf.detach().cpu().numpy(), 0.01)
            mesh = trimesh.Trimesh(vertices, faces)
            components = mesh.split(only_watertight=False)
            bbox = []
            for k, c in enumerate(components):
                bbmin = c.vertices.min(0)
                bbmax = c.vertices.max(0)
                bbox.append((bbmax - bbmin).max())
                
            max_component = np.argmax(bbox)
            mesh = components[max_component]
            mesh.vertices = mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]
            os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
            trimesh.Trimesh(vertices=vertices, faces=faces).export(mesh_path)
            ms = ml.MeshSet()
            ms.set_verbosity(False)
            ms.load_new_mesh(mesh_path)
            ms.meshing_remove_connected_component_by_face_number(mincomponentsize=5000)
            ms.save_current_mesh(mesh_path)
        else:
            v, t = get_mesh_from_udf(
                udf_func,
                coords_range=(-1, 1),
                max_dist=udf_max_dist,
                N=args.resolution,
                max_batch=2**16,
                differentiable=False,
            )
            pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)
            os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
            o3d.io.write_triangle_mesh(mesh_path, pred_mesh_o3d)

    print(f"saved to: {args.output_dir}")



if __name__ == "__main__":
    main()
