
from utils.fixseed import fixseed
import os
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from models.cfg_sampler import ClassifierFreeSampleModel

from AutoEncoder.models.coordsenc import CoordsEncoder
from AutoEncoder.models.cbndec import CbnDecoder
from meshudf.meshudf import get_mesh_from_udf
from utils.utils import get_o3d_mesh_from_tensors

import open3d as o3d
from torch import Tensor
import pymeshlab as ml
import torch.nn.functional as F


def main():
    args = generate_args()

    out_path = args.output_dir
    os.makedirs(out_path, exist_ok=True)

    dist_util.setup_dist(args.device)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples


    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    cond = {}
    cond["y"] = {}
    print("You are running unconditional generaion")


    ckpt = torch.load(args.ae_dir)
    print(f'Load AutoEncoder From: {args.ae_dir}')

    latent_size = 32
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

    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model,
        (
        args.batch_size, 1, latent_size), # torch.Size([1, 1, 128, 128, 128])
        clip_denoised=False,
        model_kwargs=cond,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )


    bs = args.batch_size
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


        v, t = get_mesh_from_udf(
            udf_func,
            coords_range=(-1, 1),
            max_dist=udf_max_dist,
            N=args.resolution,
            max_batch=2**16,
            differentiable=False,
        )

        pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)
        mesh_path = os.path.join(args.output_dir, f'{id}.obj')
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        o3d.io.write_triangle_mesh(mesh_path, pred_mesh_o3d)
        ms = ml.MeshSet()
        ms.set_verbosity(False)
        ms.load_new_mesh(mesh_path)
        ms.apply_coord_laplacian_smoothing()
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=2500)
        ms.save_current_mesh(mesh_path)
    print(f'saved results to {mesh_path}')





if __name__ == "__main__":
    main()