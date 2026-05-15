
import numpy as np
import torch
import cv2
import trimesh

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    PerspectiveCameras,
    AmbientLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from threestudio.utils.smpl_x import smpl_x

class NormalRenderer():
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)


        # self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
        #                     ambient_color=((0.8,0.8,0.8),),diffuse_color=((0.5,0.5,0.5),),specular_color=((0,0,0),))

        self.lights = AmbientLights(device=self.device,ambient_color=((0.6, 0.6, 0.6),))  # 纯环境光，无方向属性


        mesh = trimesh.load("/home/ljr/Downloads/Cloth_avatar/assets/smplx0.ply", process=False)  # process=False 保留原始数据
        
        self.faces = mesh.faces        # numpy数组，形状为 [M, 3]，表示M个三角形面   


        vmin = mesh.vertices.min(0)
        vmax = mesh.vertices.max(0)
        self.ori_center = (vmax + vmin) / 2
        vertices = mesh.vertices - self.ori_center
        # coordinate system: opengl --> blender (switch y/z)
        
        self.verts = torch.tensor(vertices, dtype=torch.float32)[None].cuda()    

    def render_mesh(self, cam_param):
        '''
        mode: normal, phong, texture
        '''
        # print(cam_param['focal'].shape)
        # print(verts.shape)
        # print(faces.shape)

        # verts_hom = torch.cat([self.verts, torch.ones_like(self.verts[..., :1]).cuda()], dim=-1)  # 形状变为 [1, N, 4]

        # verts_hom_t = verts_hom.transpose(1, 2)  # 形状变为 [1, 4, N]

        # wv_transform = cam_param.world_view_transform.unsqueeze(0).cuda()  # 若为 [4,4]，添加 batch 维度变为 [1,4,4]
        # verts_cam_hom_t = torch.matmul(wv_transform, verts_hom_t)  # 形状 [1, 4, N]
        # verts = verts_cam_hom_t.transpose(1, 2)[..., :3]  # 形状 [1, N, 3]
     
        verts = torch.bmm(cam_param.R[None], self.verts.permute(0,2,1)).permute(0,2,1) + cam_param.T.view(-1,1,3)
        batch_size = verts.shape[0]
        faces = torch.from_numpy(self.faces).cuda()[None,:,:].repeat(batch_size,1,1)
        verts = torch.stack((-verts[:,:,0], -verts[:,:,1], verts[:,:,2]),2).float()

        # mesh = trimesh.Trimesh(vertices=verts[0].cpu().numpy(), faces=self.faces)
        # mesh.export(f'smplx_out{id}.obj')
        cameras = PerspectiveCameras(focal_length=cam_param.focal.view(1,2),
                                principal_point=cam_param.princpt.view(1,2),
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(cam_param.image_shape).cuda().view(1,2))
        raster_settings = RasterizationSettings(image_size=cam_param.image_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=64, max_faces_per_bin=100000)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
        shader = HardPhongShader(device=self.device, cameras=cameras, lights=self.lights)

        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())

            # normals_transformed = torch.bmm(normals, cam_param.R[None].transpose(1, 2))  # (1, N, 3)
            # normals_transformed = torch.stack((-normals_transformed[:,:,0], -normals_transformed[:,:,1], normals_transformed[:,:,2]), 2)
            
            # 可视化时重新归一化
            normals_vis = (normals * 0.5 + 0.5).float()  # (1, N, 3)

            # normals_vis = (normals* 0.5 + 0.5).float()
            mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
            with torch.cuda.amp.autocast(enabled=False):
                image_normal = renderer(mesh_normal)
            
            alpha_mask = image_normal[..., 3] == 0  # 找到背景像素
            image_normal_rgb = image_normal[..., :3]  # 提取RGB通道
            # image_normal_rgb[alpha_mask] = torch.tensor([0.0, 0.0, 0.0], device=self.device) 
  

        return image_normal_rgb[:, :, :, :3]
    

    # def render_mesh(self, cam_param):
    #     # 顶点相机坐标变换（与原代码保持一致）
    #     verts = torch.bmm(cam_param.R[None], self.verts.permute(0,2,1)).permute(0,2,1) + cam_param.T.view(-1,1,3)
    #     batch_size = verts.shape[0]
    #     faces = torch.from_numpy(self.faces).cuda()[None,:,:].repeat(batch_size,1,1)
    #     # 坐标翻转（与原代码保持一致，适配渲染坐标系）
    #     verts = torch.stack((-verts[:,:,0], -verts[:,:,1], verts[:,:,2]), 2).float()

    #     # 初始化相机（与原代码保持一致）
    #     cameras = PerspectiveCameras(
    #         focal_length=cam_param.focal.view(1,2),
    #         principal_point=cam_param.princpt.view(1,2),
    #         device='cuda',
    #         in_ndc=False,
    #         image_size=torch.LongTensor(cam_param.image_shape).cuda().view(1,2)
    #     )

    #     # ---------------------- 渲染深度图 ----------------------
    #     # 配置 rasterizer 以输出深度缓冲区（zbuf）
    #     raster_settings_depth = RasterizationSettings(
    #         image_size=cam_param.image_shape,
    #         blur_radius=0.0,
    #         faces_per_pixel=1,
    #         bin_size=64,
    #         max_faces_per_bin=100000,
    #         z_clip_value=1e-3  # 避免近平面裁剪问题
    #     )
    #     rasterizer_depth = MeshRasterizer(
    #         cameras=cameras, 
    #         raster_settings=raster_settings_depth
    #     ).cuda()

    #     # 执行 rasterization 并获取深度信息
    #     with torch.no_grad():
    #         with torch.cuda.amp.autocast(enabled=False):
    #             mesh = Meshes(verts, faces)
    #             #  rasterize 输出包含 pix_to_face（像素对应的面）和 zbuf（深度缓冲区）
    #             raster_out = rasterizer_depth(mesh)
    #         zbuf = raster_out.zbuf  # 形状: [batch_size, H, W, 1]，值为相机坐标系下的深度（z坐标）

    #         # 处理深度图（归一化到 [0, 1] 范围以便可视化）
    #         # 过滤无效深度值（未被网格覆盖的像素，zbuf 为 -inf）
    #         valid_mask = zbuf != -float('inf')
    #         min_depth = torch.min(zbuf[valid_mask]) if torch.any(valid_mask) else 0.0
    #         max_depth = torch.max(zbuf[valid_mask]) if torch.any(valid_mask) else 1.0

    #         # 归一化深度值到 [0, 1]
    #         depth_map = (zbuf - min_depth) / (max_depth - min_depth + 1e-8)
    #         # 无效像素设为 0（或其他值，如 1.0）
    #         depth_map[~valid_mask] = 0.0
    #         # 扩展为 3 通道（便于与 RGB 图像兼容）
    #         depth_map_vis = depth_map.repeat(1, 1, 1, 3)  # 形状: [batch_size, H, W, 3]
    #     return depth_map_vis

    def sample_smplx_points(self):
        verts_upsampled = smpl_x.upsample_mesh(self.verts[0][:,:3])
        return verts_upsampled.detach()

torch.cuda.set_device(torch.device("cuda:0"))
renderer = NormalRenderer()

def render(verts, faces, cam_param, render_shape, colors=None):
    return renderer.render_mesh(verts, faces, cam_param, render_shape, colors)

def render_trimesh(mesh, cam_param, render_shape, mode='npta'):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    image = renderer.render_mesh(verts, faces, cam_param, render_shape, colors=colors, mode=mode)[0]
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    return image


image_size = 1024
def render_joint(smpl_jnts, bone_ids):
    marker_sz = 6
    line_wd = 2

    image = np.ones((image_size, image_size,3), dtype=np.uint8)*255 
    smpl_jnts[:,1] += 0.3
    smpl_jnts[:,1] = -smpl_jnts[:,1] 
    smpl_jnts = smpl_jnts[:,:2]*image_size/2 + image_size/2

    for b in bone_ids:
        if b[0]<0 : continue
        joint = smpl_jnts[b[0]]
        cv2.circle(image, joint.astype('int32'), color=(0,0,0), radius=marker_sz, thickness=-1)

        joint2 = smpl_jnts[b[1]]
        cv2.circle(image, joint2.astype('int32'), color=(0,0,0), radius=marker_sz, thickness=-1)

        cv2.line(image, joint2.astype('int32'), joint.astype('int32'), color=(0,0,0), thickness=int(line_wd))

    return image



def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = [ 'pink', #0
                'blue', #1
                'green', #2
                'red', #3
                'pink', #4
                'pink', #5
                'pink', #6
                'green', #7
                'blue', #8
                'red', #9
                'pink', #10
                'pink', #11
                'pink', #12
                'blue', #13
                'green', #14
                'red', #15
                'cyan', #16
                'darkgreen', #17
                'pink', #18
                'pink', #19
                'blue', #20
                'green', #21
                'pink', #22
                'pink' #23
    ]


    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[1],
                    'green':cmap.colors[3],
                    'pink': [1,1,1],
                    'red':cmap.colors[5],
                    }

    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]# [1x24x3]
    verts_colors = weights[:,:,None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors