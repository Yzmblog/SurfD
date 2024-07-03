import torch
from typing import Iterable, List, Tuple, Union, Callable
from torch import Tensor
import open3d as o3d
import numpy as np
import math

def batchify(inputs: List[Tensor], required_dim: int) -> Tuple[bool, List[Tensor]]:
    """Batchify input tensors if needed.
    All the input tensors with a number of dimensions smaller than
    required_dim will be expanded with a leading batch dimension.
    Args:
        inputs: The tensors to batchify.
        required_dim: The required number of dimensions.
    Returns:
        - A flag that indicates wether one of the inputs has been batchified.
        - The batchified tensors.
    """
    results: List[Tensor] = []
    has_changed = False

    for t in inputs:
        has_changed = len(t.shape) < required_dim or has_changed
        batched_t = torch.unsqueeze(t, dim=0) if has_changed else t
        results.append(batched_t)

    return has_changed, results


def unbatchify(inputs: List[Tensor]) -> List[Tensor]:
    """Remove batch dimension from input tensors.
    Args:
        inputs: The tensors to unbatchify.
    Returns:
        The unbatchified tensors.
    """
    results: List[Tensor] = []
    for t in inputs:
        unbatched_t = torch.squeeze(t, dim=0)
        results.append(unbatched_t)

    return results

def random_point_sampling(pcd: Tensor, num_points: int, inds=None) -> Tensor:
    """Sample the requested number of points from the given point cloud(s).
    Points are sampled randomly. If num_points is greater than NUM_POINTS,
    then points are sampled with replacement.
    Args:
        pcd: The input point cloud(s) with shape ([B,] NUM_POINTS, D).
        num_points: The number of points to sample.
    Returns:
        The sampled points with shape ([B,] NUM_SAMPLED_POINTS, D).
    """
    #print(pcd.shape)
    batched, [pcd] = batchify([pcd], 3)
    
    batch_size, original_num_points, _ = pcd.shape
    #print(batch_size, original_num_points)
    #torch.random.manual_seed(10)
    if inds is None:
        #print(original_num_points, num_points)
        weights = torch.ones((batch_size, original_num_points), dtype=torch.float)
        weights = weights.to(pcd.device)
        replacement = original_num_points < num_points
        indices_to_sample = torch.multinomial(weights, num_points, replacement=replacement)
    else:
        #print(original_num_points, num_points)
        indices_to_sample = inds
    #print(indices_to_sample)

    batch_indices = torch.arange(batch_size).reshape(batch_size, 1)
    sampled_points = pcd[batch_indices, indices_to_sample]

    if batched:
        [sampled_points] = unbatchify([sampled_points])

    return sampled_points

def get_o3d_mesh_from_tensors(
    vertices: Union[Tensor, np.ndarray],
    triangles: Union[Tensor, np.ndarray],
) -> o3d.geometry.TriangleMesh:
    """Get open3d mesh from either numpy arrays or torch tensors.
    The input vertices must have shape (NUM_VERTICES, D), where D
    can be 3 (only X,Y,Z), 6 (X,Y,Z and normals) or 9 (X,Y,Z, normals and colors).
    The input triangles must have shape (NUM_TRIANGLES, D), where D can be 3
    (only vertex indices) or 6 (vertex indices and normals).
    Args:
        vertices: The numpy array or torch tensor with vertices
            with shape (NUM_VERTICES, D).
        triangles: The numpy array or torch tensor with triangles
            with shape (NUM_TRIANGLES, D).
    Returns:
        The open3d mesh.
    """
    mesh_o3d = o3d.geometry.TriangleMesh()

    if isinstance(vertices, Tensor):
        v = vertices.clone().detach().cpu().numpy()
    else:
        v = np.copy(vertices)

    if isinstance(triangles, Tensor):
        t = triangles.clone().detach().cpu().numpy()
    else:
        t = np.copy(triangles)

    mesh_o3d.vertices = o3d.utility.Vector3dVector(v[:, :3])

    if v.shape[1] == 6:
        mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(v[:, 3:6])

    if v.shape[1] == 9:
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(v[:, 6:9])

    mesh_o3d.triangles = o3d.utility.Vector3iVector(t[:, :3])

    if t.shape[1] == 6:
        mesh_o3d.triangle_normals = o3d.utility.Vector3dVector(t[:, 3:6])

    return mesh_o3d

def compute_gradients(x: Tensor, y: Tensor) -> Tensor:
    grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True)[0]
    return grads

def sample_udf(
    udf_func: Callable[[Tensor], Tensor],
    coords: Tensor,
    max_batch: int,
    grad: bool = False,
) -> Tensor:
    udf = torch.zeros(coords.shape[0]).cuda()
    start = 0

    while start < coords.shape[0]:
        end = min(start + max_batch, coords.shape[0])
        p = coords[start:end]
        if grad:
            udf[start:end] = udf_func(p)
        else:
            with torch.no_grad():
                udf[start:end] = udf_func(p)
        start = end

    return udf



class GridFiller:
    """
    Coarse to fine method for querying an SDF network, using cached grids.
    #
    We start by evaluating the field on a low resolution grid, and then
    iteratively subdivide each voxel and re-evaluate the field only
    where needed until we reach a desired grid resolution.
    We subdivide voxels if the field absolute value on any of the voxel corners
    is smaller than the voxel diagonal √2∆x, where ∆x denotes voxel size.
    #
    In practice the coarsest level is here hardcoded to be 32**3.
    """

    def __init__(
        self,
        final_resolution: int,
        voxel_origin: Tuple[int, int, int] = (-1, -1, -1),
        cube_side_length: float = 2.0,
    ):
        # Save attributes
        self.N_max = final_resolution
        self.num_samples = final_resolution**3
        self.N_levels = [32 * (2**i) for i in range(int(math.log2(self.N_max) - 4))]
        self.voxel_origin = voxel_origin
        self.cube_side_length = cube_side_length

        # Construct grid, and precompute sparse masks, from 32 (coarsest grid) to final_resolution
        """
        Create one empty grid (N,N,N,7) where the 7 channels are (x,y,z, UDF, +3 for gradients).
        """
        voxel_size = self.cube_side_length / (self.N_max - 1)
        self.voxel_size = voxel_size
        overall_index = torch.arange(0, self.N_max**3, 1, out=torch.LongTensor())
        samples = torch.zeros(self.N_max**3, 7)

        # Transform the first 3 columns to be the x, y, z indices.
        samples[:, 2] = overall_index % self.N_max
        samples[:, 1] = (
            torch.div(overall_index, self.N_max, rounding_mode="floor") % self.N_max
        )
        samples[:, 0] = (
            torch.div(
                torch.div(overall_index, self.N_max, rounding_mode="floor"),
                self.N_max,
                rounding_mode="floor",
            )
            % self.N_max
        )

        # Then transform the first 3 columns to be the x, y, z coordinates.
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        samples.requires_grad = False
        #samples.pin_memory()
        #self.samples = samples.cuda()
        self.samples = samples

        """
        Precompute binary masks for adressing the above grid at different resolutions.
        """
        mask = torch.zeros(self.N_max**3).bool()
        mask = mask.reshape(self.N_max, self.N_max, self.N_max)

        # Fill dictionaries with precomputed masks.
        self.masks_coarse = {}
        self.masks_coarse_no_recompute = {}
        self.idxs_coarse_neighbors_blocks = {}
        for i, N in enumerate(self.N_levels):
            #### 1: Subsample coarsely.
            mask_coarse = mask.clone()
            mask_coarse[
                :: self.N_max // N, :: self.N_max // N, :: self.N_max // N
            ] = True

            # (N_max**3) array, with True only for indices of the coarse sampling (N**3 locations):
            mask_coarse = mask_coarse.reshape(-1)
            self.masks_coarse[i] = mask_coarse.clone().cuda()

            #### 2: Compute the indices of neighboring blocks.
            neighbors_block_coarse = mask.clone()
            neighbors_block_coarse[
                : self.N_max // N, : self.N_max // N, : self.N_max // N
            ] = True
            neighbors_block_coarse = neighbors_block_coarse.reshape(-1)
            # Shape (N**3 / 64, 64): idxs_coarse_neighbors_blocks[i] represents the (N_max // N)**3 indices covered by coarse point i.
            idxs_coarse_neighbors_blocks = torch.where(mask_coarse)[0].reshape(
                -1, 1
            ) + torch.where(neighbors_block_coarse)[0].reshape(1, -1)
            self.idxs_coarse_neighbors_blocks[
                i
            ] = idxs_coarse_neighbors_blocks.clone().cuda()

            #### 3: For levels finer than the coarsest one, do not recompute already queried SDFs.
            if i > 0:
                mask_coarse_no_recompute = mask_coarse.clone()
                mask_coarse_no_recompute[self.masks_coarse[i - 1]] = False
                self.masks_coarse_no_recompute[
                    i
                ] = mask_coarse_no_recompute.clone().cuda()

    def fill_grid(
        self, udf_func: Callable[[Tensor], Tensor], max_batch: int
    ) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            samples = self.samples.clone()
            close_surface_masks = {}
            idxs_coarse_neighbors_blocks_LOCAL = {}

            for level, N in enumerate(self.N_levels):
                """Prepare masks based on previous levels"""
                if level == 0:
                    mask_coarse = self.masks_coarse[level]
                    idxs_coarse_neighbors_blocks = self.idxs_coarse_neighbors_blocks[
                        level
                    ].clone()
                    mask_coarse_no_recompute = self.masks_coarse[level]
                else:
                    # Mask using previous queries: binary mask.
                    mask_coarse = self.masks_coarse[level].clone()
                    for l in range(level):
                        mask_coarse[
                            idxs_coarse_neighbors_blocks_LOCAL[l][
                                ~close_surface_masks[l]
                            ]
                        ] = False

                    # Compute the corresponding indices tensor.
                    if N < self.N_max:
                        idxs_coarse_neighbors_blocks = (
                            self.idxs_coarse_neighbors_blocks[level].clone()
                        )
                        idxs_coarse_neighbors_blocks = idxs_coarse_neighbors_blocks[
                            mask_coarse[self.masks_coarse[level]]
                        ]
                    else:
                        idxs_coarse_neighbors_blocks = (
                            self.idxs_coarse_neighbors_blocks[level]
                        )

                    # The no_recompute version does not query the decoder for nodes that have
                    # already been computed at coarser levels.
                    mask_coarse_no_recompute = self.masks_coarse_no_recompute[
                        level
                    ].clone()
                    for l in range(level):
                        mask_coarse_no_recompute[
                            idxs_coarse_neighbors_blocks_LOCAL[l][
                                ~close_surface_masks[l]
                            ]
                        ] = False
                idxs_coarse_neighbors_blocks_LOCAL[level] = idxs_coarse_neighbors_blocks

                """ Query the network """
                xyz = samples[mask_coarse_no_recompute, 0:3].cuda()
                # Query and fill grid.
                samples[mask_coarse_no_recompute, 3] = sample_udf(
                    udf_func, xyz, max_batch=max_batch
                ).cpu()

                """ Prepare next levels queries """
                if N < self.N_max:
                    ## Which samples are close to the surface?
                    step_size = 2.0 / N
                    close_surface_mask = (
                        torch.abs(samples[mask_coarse, 3]) < 1.5 * 1.7 * step_size
                    )
                    close_surface_masks[level] = close_surface_mask

                    # For those far of the surface, we can ignore them for the future and copy the high value to their neighbors
                    samples[
                        idxs_coarse_neighbors_blocks[~close_surface_mask], 3
                    ] = samples[mask_coarse, 3][~close_surface_mask].unsqueeze(-1)

            udf_values = samples[:, 3]
            udf_values = udf_values.reshape(self.N_max, self.N_max, self.N_max)

        #torch.cuda.empty_cache()
        # Compute gradients only where the predicted udf value is small.
        # mask_gradients = samples[:, 3] < (2.5 * self.cube_side_length / self.N_max)
        # samples[mask_gradients, 4:] = sample_grads(
        #     udf_func, samples[mask_gradients, :3].cuda(), max_batch=max_batch
        # ).cpu()
        # gradients = samples[:, 4:]
        # gradients = gradients.reshape(self.N_max, self.N_max, self.N_max, 3)
        gradients = None
        del samples
        torch.cuda.empty_cache()
        return udf_values, gradients