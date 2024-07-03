from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from einops import repeat
from torch import Tensor
from tqdm import tqdm


def read_mesh(
    mesh_path: Union[str, Path],
    dtype: torch.dtype = torch.float,
) -> Tuple[Tensor, Tensor]:
    """Read a mesh from a given file.
    The mesh is returned as a tuple of torch tensors, containing:
    - The vertices with shape (N, D). D can be 3 if only coordinates are available,
      6 if also normals or colors are available, 9 if both normals
      and colors are available.
    - The trianges with shape (M, D). D can be 3 if normals are not available,
      6 otherwise.
    Args:
        mesh_path: The path of the mesh file.
        dtype: The data type for the output tensors.
    Raises:
        ValueError: If the given file doesn't exist.
    Returns:
        - The tensor with the mesh vertices with shape (N, D).
        - The tensor with the mesh triangles with shape (M, D).
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise ValueError(f"The mesh file {str(mesh_path)} does not exists.")

    mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
    return get_tensor_mesh_from_o3d(mesh_o3d, dtype)


def get_tensor_mesh_from_o3d(
    mesh_o3d: o3d.geometry.TriangleMesh,
    dtype: torch.dtype = torch.float,
) -> Tuple[Tensor, Tensor]:
    """Convert an open3d mesh to a tuple of torch tensors.
    The mesh is returned as a tuple of torch tensors, containing:
    - The vertices with shape (N, D). D can be 3 if only coordinates are available,
      6 if also normals or colors are available, 9 if both normals
      and colors are available.
    - The trianges with shape (M, D). D can be 3 if normals are not available,
      6 otherwise.
    Args:
        mesh_o3d: The open3d mesh.
    Returns:
        - The tensor with the mesh vertices with shape (N, D).
        - The tensor with the mesh triangles with shape (M, D).
    """
    vertices = torch.tensor(np.asarray(mesh_o3d.vertices), dtype=dtype)

    if len(mesh_o3d.vertex_normals) > 0:
        vertices_normals = torch.tensor(
            np.asarray(mesh_o3d.vertex_normals), dtype=dtype
        )
        vertices = torch.cat((vertices, vertices_normals), dim=-1)

    if len(mesh_o3d.vertex_colors) > 0:
        vertices_colors = torch.tensor(np.asarray(mesh_o3d.vertex_colors), dtype=dtype)
        vertices = torch.cat((vertices, vertices_colors), dim=-1)

    triangles = torch.tensor(np.asarray(mesh_o3d.triangles), dtype=dtype)

    if len(mesh_o3d.triangle_normals) > 0:
        triangles_normals = torch.tensor(
            np.asarray(mesh_o3d.triangle_normals), dtype=dtype
        )
        triangles = torch.cat((triangles, triangles_normals), dim=-1)

    return vertices, triangles


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


def progress_bar(iterable: Iterable, desc: str = "", num_cols: int = 60) -> Iterable:
    """Decorate an iterable object using a progress bar.
    Args:
        iterable: the iterable to decorate.
        desc: the description to print. Defaults to "".
        num_cols: The width of the entire output message. Defaults to 60.
    Returns:
        The decorated iterable.
    """
    bar_format = "{percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    if len(desc) > 0:
        bar_format = "{desc}: " + bar_format
    return tqdm(iterable, desc=desc, bar_format=bar_format, ncols=num_cols, leave=False)


def get_tensor_pcd_from_o3d(
    pcd_o3d: o3d.geometry.PointCloud,
    dtype: torch.dtype = torch.float,
) -> Tensor:
    """Convert an open3d point cloud to a torch tensor.
    The point cloud is returned as a torch tensor with shape (NUM_POINTS, D).
    D can be 3 (only XYZ coordinates), 6 (XYZ coordinates and
    normals/colors) or 9 (XYZ coordinates, normals and colors).
    Args:
        pcd_o3d: The open3d point cloud.
    Returns:
        A torch tensor with the loaded point cloud with shape (NUM_POINTS, D).
    """
    pcd_torch = torch.tensor(np.asarray(pcd_o3d.points), dtype=dtype)

    if len(pcd_o3d.normals) > 0:
        normals_torch = torch.tensor(np.asarray(pcd_o3d.normals), dtype=dtype)
        pcd_torch = torch.cat((pcd_torch, normals_torch), dim=-1)

    if len(pcd_o3d.colors) > 0:
        colors_torch = torch.tensor(np.asarray(pcd_o3d.colors), dtype=dtype)
        pcd_torch = torch.cat((pcd_torch, colors_torch), dim=-1)

    return pcd_torch


def sample_points_around_pcd(
    pcd: Tensor,
    stds: List[float],
    num_points_per_std: List[int],
    coords_range: Tuple[float, float],
    device: str = "cpu",
) -> Tensor:
    """Sample points around the given point cloud.
    Points are sampled by adding gaussian noise to the input cloud,
    according to the given standard deviations. Additionally, points
    are also sampled uniformly in the given range.
    Args:
        pcd: The point cloud tensor with shape (N, 3).
        stds: A list of standard deviations to compute the gaussian noise
            to obtain the points.
        num_points_per_std: A list with the number of points to sample for each
            standard deviation. The last number refers to points sampled uniformly
            in the given range (i.e., len(num_points_per_std) = len(stds) + 1).
        coords_range: The range for the points coordinates.
        device: The device for the sampled points. Defaults to "cpu".
    Returns:
        The sampled points with shape (M, 3).
    """
    coords = torch.empty(0, 3).to(device)
    num_points_pcd = pcd.shape[0]

    for sigma, num_points in zip(stds, num_points_per_std[:-1]):
        mul = num_points // num_points_pcd

        if mul > 0:
            coords_for_sampling = repeat(pcd, "n d -> (n r) d", r=mul).to(device)
        else:
            coords_for_sampling = torch.empty(0, 3).to(device)

        still_needed = num_points % num_points_pcd
        if still_needed > 0:
            weights = torch.ones(num_points_pcd, dtype=torch.float).to(device)
            indices_random = torch.multinomial(weights, still_needed, replacement=False)
            pcd_random = pcd[indices_random].to(device)
            coords_for_sampling = torch.cat((coords_for_sampling, pcd_random), dim=0)

        offsets = torch.randn(num_points, 3).to(device) * sigma
        coords_i = coords_for_sampling + offsets

        coords = torch.cat((coords, coords_i), dim=0)

    random_coords = torch.rand(num_points_per_std[-1], 3).to(device)
    random_coords *= coords_range[1] - coords_range[0]
    random_coords += coords_range[0]
    coords = torch.cat((coords, random_coords), dim=0)

    coords = torch.clip(coords, min=coords_range[0], max=coords_range[1])

    return coords


def compute_udf_and_gradients(
    mesh_o3d: o3d.geometry.TriangleMesh,
    queries: Tensor,
) -> Tuple[Tensor, Tensor]:
    scene = o3d.t.geometry.RaycastingScene()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    triangles = np.asarray(mesh_o3d.triangles, dtype=np.uint32)
    _ = scene.add_triangles(vertices, triangles)

    #signed_distance = scene.compute_signed_distance(query_point)
    closest_points = scene.compute_closest_points(queries.numpy())["points"]
    closest_points = torch.tensor(closest_points.numpy())

    q2p = queries - closest_points
    udf = torch.linalg.vector_norm(q2p, dim=-1)
    gradients = F.normalize(q2p, dim=-1)

    return udf, gradients

def compute_sdf_and_gradients(
    mesh_o3d: o3d.geometry.TriangleMesh,
    queries: Tensor,
) -> Tuple[Tensor, Tensor]:
    scene = o3d.t.geometry.RaycastingScene()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    triangles = np.asarray(mesh_o3d.triangles, dtype=np.uint32)
    _ = scene.add_triangles(vertices, triangles)

    signed_distance = scene.compute_signed_distance(queries.numpy())
    closest_points = scene.compute_closest_points(queries.numpy())["points"]
    closest_points = torch.tensor(closest_points.numpy())
    signed_distance = torch.tensor(signed_distance.numpy())

    #print(signed_distance.shape)

    #gradients = np.zeros((signed_distance.shape[0], 3))
    q2p = queries - closest_points
    gradients = F.normalize(q2p, dim=-1)
    gradients = np.sign(signed_distance)[:, None] * gradients
    #print(gradients.shape)

    return signed_distance, gradients



def compute_udf_from_mesh(
    mesh_o3d: o3d.geometry.TriangleMesh,
    num_surface_points: int = 100_000,
    num_queries_on_surface: int = 10_000,
    queries_stds: List[float] = [0.003, 0.01, 0.1],
    num_queries_per_std: List[int] = [5_000, 4_000, 500, 500],
    coords_range: Tuple[float, float] = (-1.0, 1.0),
    max_dist: float = 0.1,
    convert_to_bce_labels: bool = False,
    use_cuda: bool = True,
    input_queries = None
) -> Tuple[Tensor, Tensor, Tensor]:
    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=num_surface_points)
    pcd = get_tensor_pcd_from_o3d(pcd_o3d)[:, :3]

    device = "cuda" if use_cuda else "cpu"
    if input_queries is not None:
        queries = input_queries
    else:
        queries = sample_points_around_pcd(
            pcd,
            queries_stds,
            num_queries_per_std,
            coords_range,
            device,
        )
    queries = queries.cpu()

    udf, gradients = compute_udf_and_gradients(mesh_o3d, queries)
    values = torch.clip(udf, min=0, max=max_dist)

    # q_on_surf_o3d = mesh_o3d.sample_points_uniformly(
    #     number_of_points=num_queries_on_surface
    # )
    # queries_on_surface = get_tensor_pcd_from_o3d(q_on_surf_o3d)[:, :3]
    # values_on_surface = torch.zeros(num_queries_on_surface)
    # gradients_on_surface = torch.zeros(num_queries_on_surface, 3)

    # queries = torch.cat([queries_on_surface, queries], dim=0)
    # values = torch.cat([values_on_surface, values], dim=0)
    # gradients = torch.cat([gradients_on_surface, gradients], dim=0)

    # if convert_to_bce_labels:
    #     values /= max_dist
    #     values = 1 - values

    return queries, values, gradients


def compute_sdf_from_mesh(
    mesh_o3d: o3d.geometry.TriangleMesh,
    num_surface_points: int = 100_000,
    num_queries_on_surface: int = 10_000,
    queries_stds: List[float] = [0.003, 0.01, 0.1],
    num_queries_per_std: List[int] = [5_000, 4_000, 500, 500],
    coords_range: Tuple[float, float] = (-1.0, 1.0),
    max_dist: float = 0.1,
    convert_to_bce_labels: bool = False,
    use_cuda: bool = True,
    input_queries = None
) -> Tuple[Tensor, Tensor, Tensor]:
    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=num_surface_points)
    pcd = get_tensor_pcd_from_o3d(pcd_o3d)[:, :3]

    device = "cuda" if use_cuda else "cpu"
    if input_queries is not None:
        queries = input_queries
    else:
        queries = sample_points_around_pcd(
            pcd,
            queries_stds,
            num_queries_per_std,
            coords_range,
            device,
        )
    queries = queries.cpu()

    sdf, gradients = compute_sdf_and_gradients(mesh_o3d, queries)
    values = torch.clip(sdf, min=-max_dist, max=max_dist)

    q_on_surf_o3d = mesh_o3d.sample_points_uniformly(
        number_of_points=num_queries_on_surface
    )
    queries_on_surface = get_tensor_pcd_from_o3d(q_on_surf_o3d)[:, :3]
    values_on_surface = torch.zeros(num_queries_on_surface)
    gradients_on_surface = torch.zeros(num_queries_on_surface, 3)

    queries = torch.cat([queries_on_surface, queries], dim=0)
    values = torch.cat([values_on_surface, values], dim=0)
    gradients = torch.cat([gradients_on_surface, gradients], dim=0)

    if convert_to_bce_labels:
        values /= max_dist
        values = 1 - values

    return queries, values, gradients

def compute_gradients(x: Tensor, y: Tensor) -> Tensor:
    grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True)[0]
    return grads


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


def random_point_sampling(pcd: Tensor, num_points: int) -> Tensor:
    """Sample the requested number of points from the given point cloud(s).
    Points are sampled randomly. If num_points is greater than NUM_POINTS,
    then points are sampled with replacement.
    Args:
        pcd: The input point cloud(s) with shape ([B,] NUM_POINTS, D).
        num_points: The number of points to sample.
    Returns:
        The sampled points with shape ([B,] NUM_SAMPLED_POINTS, D).
    """
    batched, [pcd] = batchify([pcd], 3)

    batch_size, original_num_points, _ = pcd.shape

    weights = torch.ones((batch_size, original_num_points), dtype=torch.float)
    weights = weights.to(pcd.device)

    replacement = original_num_points < num_points

    indices_to_sample = torch.multinomial(weights, num_points, replacement=replacement)
    
    batch_indices = torch.arange(batch_size).reshape(batch_size, 1)
    sampled_points = pcd[batch_indices, indices_to_sample]

    if batched:
        [sampled_points] = unbatchify([sampled_points])

    return sampled_points
