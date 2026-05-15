import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # 防止 Qt 尝试加载 GUI
 
import matplotlib
matplotlib.use('Agg')  # 确保在导入 pyplot 之前设置后端
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
import os
from scipy.spatial.transform import Rotation as R

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

# t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(10, 10), fps=120, radius=2,
                   vis_mode='gt', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        # fig.suptitle(title, fontsize=10)
        # ax.grid(b=False)

        
        ax.grid(None)
        
        # 设置X、Y、Z面的背景是白色
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        # 设置坐标轴不可见
        ax.axis('off')


    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    # colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_blue = ["#4773C4", "#4773C4", "#4773C4", "#4773C4", "#4773C4"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        #         ax =
        # plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
        #              MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 16.0
            else:
                linewidth = 10.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)


        # plt.savefig(save_path, dpi=1000)
        # plt.show()
    # update(2)


        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps, writer='ffmpeg')
    # # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()



if __name__ == "__main__":
    import smplx
    import torch
    smpl_model_path = '/home/ljr/Downloads/HUMAN_MODELS/'
    # smplx_model_path = '/home/ljr/Downloads/HUMAN_MODELS/models_smplx_v1_1/models/'
    smplx_model = smplx.create(
        smpl_model_path, model_type='SMPL', gender='neutral', batch_size=21)
    save_path = "/home/ljr/Downloads/SelfAvatar_ori/bike.mp4"
    kinematic = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    title = "pose_show"
    dataset = "humanml"


    # pose_seattle =  [[0.09782122820615768, 0.10309028625488281, 0.1554555594921112], [0.15125219523906708, 0.010869174264371395, 0.0521777868270874], [0.14460262656211853, -0.007442197762429714, 0.042239390313625336], [0.09169267863035202, 0.3885538578033447, 0.07629899680614471], [-0.03633676841855049, -0.14421546459197998, 0.03324032202363014], [-0.002186780096963048, 0.049292467534542084, -0.16871777176856995], [0.10909721255302429, 0.21393223106861115, -0.18970896303653717], [0.09734640270471573, -0.07147820293903351, 0.1614588052034378], [-0.01597161404788494, -0.008653847500681877, 0.04508869722485542], [0.0006394194788299501, 0.0010789503576233983, 0.0013731217477470636], [0.0008449442102573812, -0.0010088824201375246, -0.0026770932599902153], [0.11913483589887619, -0.16889260709285736, 0.19460627436637878], [0.12104861438274384, -0.0069404346868395805, 0.19740627706050873], [0.019206074997782707, -0.08456533402204514, -0.042385928332805634], [-0.1938398778438568, -0.10573986917734146, 0.012842513620853424], [0.3802987039089203, -0.07850359380245209, -0.7098459601402283], [0.36113792657852173, -0.051448460668325424, 1.0253641605377197], [-1.7880290746688843, 4.0776238441467285, 0.9910356402397156], [0.315129816532135, 1.212407112121582, 0.3051651418209076], [0.007935339584946632, -0.35236015915870667, 0.43007349967956543], [-0.18102549016475677, 0.22588469088077545, -0.3589305579662323]]
    # pose_bike = [[0.08226235210895538, 0.08392085134983063, 0.01562969572842121], [0.05646846443414688, -0.08588071167469025, -0.08464903384447098], [0.18548014760017395, -0.0243303831666708, 0.07046976685523987], [0.052270080894231796, 0.0073435017839074135, 0.1419684886932373], [0.19453896582126617, -0.16721384227275848, 0.06035524979233742], [-0.05175807699561119, 0.0800669938325882, -0.2483557015657425], [-0.0913797914981842, 0.18637071549892426, -0.12178733199834824], [-0.11309372633695602, -0.4463815689086914, 0.02260279469192028], [0.0981382504105568, 0.029209965839982033, 0.11860565096139908], [0.0003618336922954768, 0.001694758771918714, 0.0009560206672176719], [0.00014600537542719394, -0.0020041377283632755, -0.001017649075947702], [0.11547251045703888, -0.14167770743370056, 0.18766841292381287], [0.03806104511022568, 0.011632893234491348, 0.054999660700559616], [-0.07475090771913528, -0.026225825771689415, 0.1193091869354248], [-0.24981756508350372, 0.009836381301283836, -0.13751821219921112], [0.15798945724964142, -0.2983056306838989, -1.0727766752243042], [0.21046608686447144, 0.1536610871553421, 1.172120451927185], [0.425554484128952, -0.5465456247329712, -0.09333678334951401], [0.31301748752593994, 0.7630741596221924, -0.08080245554447174], [-0.25822243094444275, 0.007705972529947758, 0.15414589643478394], [-0.04664479196071625, 0.0059643881395459175, -0.08821912854909897]]
    # pose_citron = [[0.3071582615375519, 0.7321499586105347, -0.02541813999414444], [-0.2071744054555893, -0.003135785460472107, 0.07533368468284607], [0.31711670756340027, -0.020651312544941902, 0.061445631086826324], [0.39842283725738525, -0.5301931500434875, 0.22507861256599426], [0.5532160997390747, 0.09833168238401413, 0.06122633442282677], [0.05506841465830803, 0.10121388733386993, -0.17818492650985718], [-0.7942678332328796, -0.21276219189167023, 0.1990012228488922], [-0.06985890120267868, 0.3398444354534149, 0.0982634648680687], [0.06475789844989777, 0.011782271787524223, -0.0038007255643606186], [-0.0003554911236278713, 0.0006159970653243363, 0.0012491296511143446], [-0.00017287391528952867, -5.702935595763847e-05, -0.0004917510086670518], [0.15581993758678436, -0.37675023078918457, 0.1834551841020584], [-0.020989255979657173, -0.17538809776306152, -0.14896352589130402], [-0.15695984661579132, 0.004916180390864611, 0.23316442966461182], [-0.18902787566184998, -0.2809983193874359, -0.05698801204562187], [0.1296548992395401, -0.6019797921180725, -0.8066131472587585], [0.2628324031829834, 0.13616032898426056, 1.1320527791976929], [0.37663522362709045, -0.43135666847229004, -0.1123712807893753], [0.43407556414604187, 0.34948617219924927, -0.08778262883424759], [-0.29574450850486755, 0.2545064687728882, 0.3193861246109009], [0.4147186279296875, 0.12767243385314941, -0.36559081077575684]]
    # pose_seq = torch.stack([torch.tensor(pose_seattle), torch.tensor(pose_bike), torch.tensor(pose_citron)]).reshape(-1, 21*3)
    
    smpl_data = np.load(os.path.join("/home/ljr/Downloads/InstantAvatar/data/custom/bike", 'poses_optimized.npz'), allow_pickle=True)
    smpl_data = dict(smpl_data)
    smpl_data = {k: torch.from_numpy(v.astype(np.float32))
                    for k, v in smpl_data.items()}
    pose_seq = smpl_data['body_pose'][:21,:].reshape(-1, 23*3)
    global_ori = smpl_data['global_orient'][:21,:].reshape(-1, 3)
    rotation = R.from_rotvec(global_ori)
    rotation_matrices = rotation.as_matrix()
    
    angle_rad = np.pi  # 90 度转换为弧度
    rotation_z_90 = R.from_euler('z', angle_rad, degrees=False).as_matrix()
    
    # 将每个旋转矩阵与绕 Z 轴旋转 90 度的矩阵相乘
    rotation_matrices_rotated = np.einsum('ijk,kl->ijl', rotation_matrices, rotation_z_90)
    
    # angle_rad = np.pi  # 90 度转换为弧度
    # rotation_y_90 = R.from_euler('y', angle_rad, degrees=False).as_matrix()
    # rotation_matrices_rotated = np.einsum('ijk,kl->ijl', rotation_matrices_rotated, rotation_y_90)   

    # 对旋转矩阵进行转置
    rotation_matrices_transposed = rotation_matrices_rotated.transpose(0,2,1)  # 假设 rotation_matrices 的形状是 (21, 3, 3)
    
    # 将转置后的旋转矩阵转换回轴角
    rotation_transposed = R.from_matrix(rotation_matrices_transposed)
    rot_vecs_transposed = torch.from_numpy(rotation_transposed.as_rotvec())
    rot_vecs_transposed_tensor = torch.tensor(rot_vecs_transposed, dtype=torch.float32) 
         
    print(pose_seq.shape)
    output = smplx_model(body_pose=pose_seq, global_orient=rot_vecs_transposed_tensor)
    joints = output.joints.detach().numpy()[:, :24]
    print(joints.shape)
    plot_3d_motion(save_path, kinematic, joints, title, dataset, fps=4)