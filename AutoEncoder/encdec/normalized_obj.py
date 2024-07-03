import trimesh
import os

def obj_normalization(input_mesh_file, output_file_path):
    '''
    Normalize the obj file based on the information from the first frame (T-pose).
    :param path: the path of original obj file
    :return:
        normalized obj file [-1,1] at the origin (0,0,0).
    '''
    obj = trimesh.load(input_mesh_file, force='mesh')
    v = obj.vertices
    v = v - v.mean(0)
    # v = v*2
    print(v.mean(0), v.max(0), v.min(0))

    trimesh.Trimesh(vertices=v, faces=obj.faces).export(output_file_path)
    return True


def box_center_normalization(input_mesh_file, output_file_path):
    '''
    Normalize the obj file based on the information from the first frame (T-pose).
    :param path: the path of original obj file
    :return:
        normalized obj file [-1,1] at the origin (0,0,0).
    '''
    obj = trimesh.load(input_mesh_file, force='mesh')
    v = obj.vertices
    max_v = v.max(0)
    min_v = v.min(0)
    lhw = max_v = min_v
    box_center = min_v + (lhw / 2)

    print(v.mean(0), v.max(0), v.min(0))
    v = v - box_center

    trimesh.Trimesh(vertices=v, faces=obj.faces).export(output_file_path)
    return True


data_root = './dataset/Deepfashion3D/filtered_registered_mesh'
ids = os.listdir(data_root)
output_root = './dataset/Deepfashion3D/norm_objs'
os.makedirs(output_root, exist_ok=True)

for id in ids:
    input_mesh_file = os.path.join(data_root, id, 'model_cleaned.obj')
    output_file_path = os.path.join(output_root, id+'.obj')
    obj_normalization(input_mesh_file, output_file_path)

