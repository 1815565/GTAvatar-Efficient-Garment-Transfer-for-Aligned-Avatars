import torch
import numpy as np
from utils.sh_utils import RGB2SH, SH2RGB
from utils.general_utils import inverse_sigmoid
from plyfile import PlyData, PlyElement
import os


def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1] * _features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(_features_rest.shape[1] * _features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_gaussians_as_ply(path, gaussian_vals: dict):
    os.makedirs(os.path.dirname(path), exist_ok = True)

    xyz = gaussian_vals['mean_3d'].detach().cpu().numpy()
    # center = np.mean(xyz, axis = 0)
    # xyz = xyz - center[None]
    normals = np.zeros_like(xyz)
    fused_color = RGB2SH(gaussian_vals['rgb'].detach())
    features = torch.zeros((fused_color.shape[0], 3, (0 + 1) ** 2))
    features[:, :3, 0] = fused_color
    features_dc = features[:, :, 0:1].transpose(1, 2)
    features_rest = features[:, :, 1:].transpose(1, 2)
    f_dc = features_dc.transpose(1, 2).flatten(start_dim = 1).contiguous().cpu().numpy()
    f_rest = features_rest.transpose(1, 2).flatten(start_dim = 1).contiguous().cpu().numpy()
    opacities = inverse_sigmoid(gaussian_vals['opacity'].detach()).cpu().numpy()
    scale = torch.log(gaussian_vals['scale'].detach()).cpu().numpy()
    rotation = gaussian_vals['rotation'].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scale, rotation)]

    elements = np.empty(xyz.shape[0], dtype = dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis = 1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def load_gaussians_from_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis = 1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # self._xyz = nn.Parameter(torch.tensor(xyz, dtype = torch.float, device = "cuda").requires_grad_(True))
    # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype = torch.float, device = "cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype = torch.float, device = "cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # self._opacity = nn.Parameter(torch.tensor(opacities, dtype = torch.float, device = "cuda").requires_grad_(True))
    # self._scaling = nn.Parameter(torch.tensor(scales, dtype = torch.float, device = "cuda").requires_grad_(True))
    # self._rotation = nn.Parameter(torch.tensor(rots, dtype = torch.float, device = "cuda").requires_grad_(True))
    #
    # self.active_sh_degree = self.max_sh_degree

    return {
        'positions': torch.tensor(xyz, dtype = torch.float, device = "cuda"),
        'colors': torch.tensor(SH2RGB(features_dc)[:, [2, 1, 0]], dtype = torch.float, device = "cuda").squeeze(-1),
        'opacity': torch.sigmoid(torch.tensor(opacities, dtype = torch.float, device = "cuda")),
        'scales': torch.exp(torch.tensor(scales, dtype = torch.float, device = "cuda")),
        'rotations': torch.nn.functional.normalize(torch.tensor(rots, dtype = torch.float, device = "cuda")),
        'features_extr': torch.tensor(features_extra, dtype = torch.float, device = "cuda")
    }


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm