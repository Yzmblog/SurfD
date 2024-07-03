""" 
Meshing algorithm for UDFs from
"Meshudf: Fast and differentiable meshing of unsigned distance field networks." 
Guillard, Benoit, Federico Stella, and Pascal Fua. ECCV 2022.

Original implementation: https://github.com/cvlab-epfl/MeshUDF
"""

import math
from collections import defaultdict
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.sparse import coo_matrix
from torch import Tensor

from meshudf._marching_cubes_lewiner import udf_mc_lewiner


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
        self.samples = samples.cuda()

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
                )

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

        # Compute gradients only where the predicted udf value is small.
        mask_gradients = samples[:, 3] < (2.5 * self.cube_side_length / self.N_max)
        samples[mask_gradients, 4:] = sample_grads(
            udf_func, samples[mask_gradients, :3], max_batch=max_batch
        )
        gradients = samples[:, 4:]
        gradients = gradients.reshape(self.N_max, self.N_max, self.N_max, 3)
        return udf_values, gradients


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


def sample_grads(
    udf_func: Callable[[Tensor], Tensor],
    coords: Tensor,
    max_batch: int,
) -> Tensor:
    grads = torch.zeros(coords.shape[0], 3).cuda()
    start = 0

    while start < coords.shape[0]:
        end = min(start + max_batch, coords.shape[0])
        p = coords[start:end].detach().clone()
        p.requires_grad = True
        udf = udf_func(p)
        udf.sum().backward()
        g = p.grad
        # norms = torch.norm(g, dim=-1)
        # g[norms < 0.5] = torch.zeros(1, 3).cuda()
        grads[start:end] = -F.normalize(g, dim=1)
        start = end

    return grads


def get_udf_and_grads(
    udf_func: Callable[[Tensor], Tensor],
    coords_range: Tuple[float, float],
    max_dist: float,
    N: int,
    max_batch: int,
) -> Tuple[Tensor, Tensor]:
    """
    Fills a dense N*N*N regular grid by querying the given function.

    Args:
        udf_func: Function to call to get the udf values.
        coords_range: The udf coordinates range.
        max_dist: The udf clipping distance.
        N: Grid resolution.
        max_batch: The maximum number of points that we can evaluate simultaneously.

    Returns:
        - (N, N, N) tensor representing udf values on the grid.
        - (N, N, N, 3) tensor representing gradients values on the grid.
    """
    # compute origin of the volume and voxel size
    origin = [coords_range[0]] * 3
    spacing = (coords_range[1] - coords_range[0]) / (N - 1)

    # prepare grid coordinates, each axis goes from 0 to (N - 1)
    x = torch.arange(0, N)
    coords_x, coords_y, coords_z = torch.meshgrid(x, x, x, indexing="ij")
    coords = torch.stack((coords_x, coords_y, coords_z), dim=-1).float()

    # scale and shift coordinates so that each axis goes from coords_range[0] to coords_range[1]
    coords *= spacing
    coords += torch.tensor(origin)
    coords = coords.reshape(N**3, 3)

    # comput udf for every corner of the grid
    zeros = torch.zeros(coords.shape[0], 4)
    samples = torch.cat([coords, zeros], dim=-1).cuda()
    samples[:, 3] = sample_udf(udf_func, samples[:, :3], max_batch)

    # compute gradients only where the predicted udf value is small
    mask = samples[:, 3] < (max_dist - 1e-3)
    samples[mask, 4:] = sample_grads(udf_func, samples[mask, :3], max_batch // 4)

    # separate values in udf and gradients
    udf = samples[:, 3]
    udf = udf.reshape(N, N, N)
    grads = samples[:, 4:]
    grads = grads.reshape(N, N, N, 3)

    return udf, grads


def get_mesh_from_udf(
    udf_func: Callable[[Tensor], Tensor],
    coords_range: Tuple[float, float],
    max_dist: float,
    N: int = 128,
    smooth_borders: bool = True,
    differentiable: bool = True,
    max_batch: int = 2**12,
    use_fast_grid_filler: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Computes a triangulated mesh from udf.

    Args:
        udf_func: Function to call to get the udf values.
        coords_range: The udf coordinates range.
        max_dist: The udf clipping distance.
        N: Grid resolution.
        smooth_borders: Do we smooth borders with a Laplacian?
        differentiable: Do we need the mesh to be differentiable wrt the UDF?
        max_batch: The maximum number of points that we can evaluate simultaneously.
        use_fast_grid_filler: Use coarse to fine UDF grid evaluator, with cached coordinates.

    Returns:
        - Vertices of the mesh.
        - Faces of the mesh.
    """
    # sample udf grid
    if not use_fast_grid_filler:
        udf, gradients = get_udf_and_grads(
            udf_func, coords_range, max_dist, N, max_batch
        )
    else:
        fast_grid_filler = GridFiller(N)
        udf, gradients = fast_grid_filler.fill_grid(udf_func, max_batch)
    udf[udf < 0] = 0

    # run custom marching cubes on it
    N = udf.shape[0]
    spacing = (coords_range[1] - coords_range[0]) / (N - 1)
    udf = udf.cpu().detach().numpy()
    gradients = gradients.cpu().detach().numpy()
    vertices, faces, _, _ = udf_mc_lewiner(udf, gradients, spacing=[spacing] * 3)

    # shift vertices according to the given range
    vertices += coords_range[0]

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    # remove faces whose vertices feature big udf values
    # check not only vertices but also points in the middle of edges
    points = np.vstack(
        (
            mesh.vertices[mesh.edges[:, 0]],
            mesh.vertices[mesh.edges[:, 1]],
            (mesh.vertices[mesh.edges[:, 0]] + mesh.vertices[mesh.edges[:, 1]]) / 2,
        )
    )
    face_idxs = np.hstack([mesh.edges_face] * 3)

    points = torch.from_numpy(points).float().cuda()
    udf = sample_udf(udf_func, points, max_batch)
    udf = udf.cpu().numpy()

    # th_dist is the threshold udf to consider a point on the surface.
    th_dist = 1 / N

    mask = udf > th_dist
    face_idxs_to_remove = np.unique(face_idxs[mask])
    face_mask = np.full(mesh.faces.shape[0], True)
    face_mask[face_idxs_to_remove] = False
    filtered_faces = mesh.faces[face_mask]
    mesh = trimesh.Trimesh(mesh.vertices, filtered_faces)

    # remove NaNs, flat triangles, duplicate faces
    mesh = mesh.process(validate=False)
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    # fill single triangle holes
    mesh.fill_holes()

    # re-process the mesh until it is stable:
    mesh_2 = trimesh.Trimesh(mesh.vertices, mesh.faces)
    n_verts, n_faces, n_iter = 0, 0, 0
    while (n_verts, n_faces) != (
        len(mesh_2.vertices),
        len(mesh_2.faces),
    ) and n_iter < 10:
        mesh_2 = mesh_2.process(validate=False)
        mesh_2.remove_duplicate_faces()
        mesh_2.remove_degenerate_faces()
        (n_verts, n_faces) = (len(mesh_2.vertices), len(mesh_2.faces))
        n_iter += 1
        mesh_2 = trimesh.Trimesh(mesh_2.vertices, mesh_2.faces)

    mesh = trimesh.Trimesh(mesh_2.vertices, mesh_2.faces)

    if smooth_borders:
        # identify borders: those appearing only once
        border_edges = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)

        # build a dictionnary of (u,l): l is the list of vertices that are adjacent to u
        neighbours = defaultdict(lambda: [])
        for u, v in mesh.edges_sorted[border_edges]:
            neighbours[u].append(v)
            neighbours[v].append(u)
        border_vertices = np.array(list(neighbours.keys()))

        # build a sparse matrix for computing laplacian
        pos_i, pos_j = [], []
        for k, ns in enumerate(neighbours.values()):
            for j in ns:
                pos_i.append(k)
                pos_j.append(j)

        sparse = coo_matrix(
            (np.ones(len(pos_i)), (pos_i, pos_j)),  # put ones at these locations
            shape=(len(border_vertices), len(mesh.vertices)),
        )

        # smoothing operation:
        lambda_ = 0.3
        for _ in range(20):
            border_neighbouring_averages = sparse @ mesh.vertices / sparse.sum(axis=1)
            laplacian = border_neighbouring_averages - mesh.vertices[border_vertices]
            mesh.vertices[border_vertices] = (
                mesh.vertices[border_vertices] + lambda_ * laplacian
            )

    final_verts = torch.tensor(mesh.vertices).float().cuda()
    final_faces = torch.tensor(mesh.faces).long().cuda()

    if differentiable:
        # use the mesh to compute normals
        normals = trimesh.geometry.weighted_vertex_normals(
            vertex_count=len(mesh.vertices),
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            face_angles=mesh.face_angles,
        )

        # evaluate the udf around each vertex, based on normals
        normals = torch.tensor(normals).float().cuda()
        verts = torch.tensor(mesh.vertices).float().cuda()
        xyz_s1 = verts + th_dist * normals
        xyz_s2 = verts - th_dist * normals
        s1 = sample_udf(udf_func, xyz_s1, max_batch, True).unsqueeze(-1)
        s2 = sample_udf(udf_func, xyz_s2, max_batch, True).unsqueeze(-1)
        # re-plug differentiability here, by this rewriting trick
        z1 = th_dist * s1 * normals - th_dist * s2 * normals
        z2 = (th_dist * s1 * normals - th_dist * s2 * normals).detach()
        new_verts = verts - z1 + z2

        # identify borders
        border_edges = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)

        # build a dictionnary of (u,v) edges, such that each vertex on the border
        # gets associated to exactly one border edge
        border_edges_dict = {}
        for u, v in mesh.edges_sorted[border_edges]:
            border_edges_dict[u] = v
            border_edges_dict[v] = u
        u_v_border = np.array(list(border_edges_dict.items()))
        u_border = u_v_border[:, 0]  # split border edges (u,v) into u and v arrays
        v_border = u_v_border[:, 1]

        # for each vertex on the border, take the cross product between
        # its normal and the border's edge
        normals_border = normals[u_border]
        edge_border = mesh.vertices[v_border] - mesh.vertices[u_border]
        edge_border = torch.tensor(edge_border).float().cuda()
        out_vec = torch.cross(edge_border, normals_border, dim=1)
        out_vec = out_vec / (
            torch.norm(out_vec, dim=1, keepdim=True) + 1e-6
        )  # make it unit length

        # then we need to orient the out_vec such that they point outwards
        # to do so, we evaluate at +- their offset, and take the corresponding max udf value
        border_verts = torch.tensor(mesh.vertices[u_border]).float().cuda()
        xyz_s1_border = border_verts + 3 * th_dist * out_vec
        xyz_s2_border = border_verts - 3 * th_dist * out_vec
        s1_border = sample_udf(udf_func, xyz_s1_border, max_batch, True).unsqueeze(-1)
        s2_border = sample_udf(udf_func, xyz_s2_border, max_batch, True).unsqueeze(-1)
        s1s2 = torch.stack((s1_border, s2_border))
        sign_out_vec = -torch.argmax(s1s2, dim=0) * 2 + 1
        out_vec = sign_out_vec * out_vec

        # filter out the verts borders for which a displacement of out_vec still present
        # a udf < th_dist, i.e. verts classified as borders which are not really so
        mask = ((s1_border + s2_border)[:, 0] > th_dist).detach().cpu().numpy()
        u_border_filtered = u_border[mask]
        out_vec_filtered = out_vec[(s1_border + s2_border)[:, 0] > th_dist]
        out_df_filtered = torch.max(s1_border, s2_border)[
            (s1_border + s2_border) > th_dist
        ]

        # plug gradients to verts positions (fake zero, just to pass grads)
        s_border = (th_dist * (out_df_filtered - out_df_filtered.detach())).unsqueeze(
            -1
        )
        new_verts[u_border_filtered] = (
            new_verts[u_border_filtered] - s_border * out_vec_filtered
        )

        final_verts = new_verts
        final_faces = torch.tensor(mesh.faces).long().cuda()

    return final_verts, final_faces
