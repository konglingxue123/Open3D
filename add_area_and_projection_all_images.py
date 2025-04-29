import json
from PIL import Image
import open3d as o3d
import numpy as np

json_dir = 'datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/transforms.json'
# json_dir = 'datasets/tea_pot_transform_mode/transforms_train.json'
# 读取json文件
with open(json_dir, 'r') as f:
    data = json.load(f)

# 打印数据
frames_dict = data['frames']

# 将frames_dict中包含的file_path的图像复制到新文件夹中
import os
import shutil

new_dir = 'datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/images_colmap_sparsed'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
    
for frame_id in frames_dict:
    
    file_path = frame_id['file_path'] # 这是路径
    if os.path.exists(file_path):
        shutil.copy(file_path, new_dir)


frames_data = data['frames']


# load a scene point cloud
scene = o3d.io.read_point_cloud('/home/ubunto/Project/konglx/pcd/projection/datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/colmap_sparse/0/sparse.ply')
scene_points = np.asarray(scene.points)
scene_colors = np.asarray(scene.colors)


#合并frames_data[0]['R'], frames_data[0]['t']为pytorch格式
R_list = []
t_list = []
transform_matrix_list = []
path_list = []
image_list = []

for frame in frames_data:
    R = frame['R']
    t = frame['t']
    transform_matrix = frame['transform_matrix']
    image_path = frame['file_path'].split('./')[-1]
    image_pil = Image.open(image_path)
    
    
    R_list.append(R)
    t_list.append(t)
    transform_matrix_list.append(transform_matrix)
    path_list.append(image_path)
    image_list.append(image_pil)



from PIL import Image, ImageDraw
import open3d as o3d
import numpy as np
import json



# load a scene point cloud
scene = o3d.io.read_point_cloud('/home/ubunto/Project/konglx/pcd/projection/datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/colmap_sparse/0/sparse.ply')
# scene = o3d.io.read_triangle_mesh('/home/ubunto/Project/konglx/pcd/image_to_3d/TRELLIS/trellis-outputs/tea-pot_letter/sample.glb')
# 可视化坐标轴. The x, y, z axis will be rendered as red, green, and blue arrows respectively.
coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])   
vizualizer = o3d.visualization.Visualizer()
WIDTH = 1920
HEIGHT = 1080
vizualizer.create_window(width=WIDTH, height=HEIGHT)


scene_colors = np.asarray(scene.colors)


#########################################################################################
def pixel_to_world(uv, depth, K, R, T):
    # 畸变校正（假设已校正，否则需调用cv2.undistortPoints）
    # 归一化坐标计算
    inv_K = np.linalg.inv(K)
    inv_R = np.linalg.inv(R)
    homogeneous_pixel = uv[..., np.newaxis]
    ndc = inv_K @ homogeneous_pixel  # 归一化坐标（未乘深度）
    # ndc = ndc.squeeze(-1)
    # 应用深度
    camera_coord = ndc * depth
    # camera_coord = ndc
    
    # 转换为世界坐标
    world_coord = (R.transpose(0,2,1) @ camera_coord).squeeze(-1) - \
                    (R.transpose(0,2,1) @ T[..., np.newaxis]).squeeze(-1) 
    return world_coord
##########################################################################################

## 原始点云 ##
pcd = o3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(scene_points)
# 添加原始点云的颜色
# colors = scene_colors
# pcd.colors = open3d.utility.Vector3dVector(colors)
intrinsics = np.array([
    [data['fl_x'], 0,            data['cx'], 0],
    [0,            data['fl_y'], data['cy'], 0],
    [0,            0,            1,          0],
    [0,            0,            0,          1]
])



# json_dir = 'datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/transforms.json'
# # 读取json文件
# with open(json_dir, 'r') as f:
#     data = json.load(f)
# frames_data = data['frames']
# print(data['h'])

# load a scene point cloud
# scene = o3d.io.read_point_cloud('/home/ubunto/Project/konglx/pcd/projection/datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/colmap_sparse/0/sparse.ply')
# scene = o3d.io.read_triangle_mesh('/home/ubunto/Project/konglx/pcd/image_to_3d/TRELLIS/trellis-outputs/tea-pot_letter/sample.glb')
# 可视化坐标轴. The x, y, z axis will be rendered as red, green, and blue arrows respectively.
# coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])  
# coor.scale(10.0) 
# vizualizer = o3d.visualization.Visualizer()
# vizualizer.create_window(width=WIDTH, height=HEIGHT)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

id = 0
for frame in zip(R_list, t_list, transform_matrix_list, path_list, image_list):
    R = frame[0]
    t = frame[1]
    transform_matrix = frame[2]
    image_path = frame[3]
    print(id)
    # print(R, t, transform_matrix, image_path)
    image_pil = Image.open(os.path.join(current_dir, image_path))

    add_xy1 = [600, 600]
    add_xy2 = [700, 700]

    draw_image = ImageDraw.Draw(image_pil)
    draw_image.rectangle((add_xy1[0], add_xy1[1], add_xy2[0], add_xy2[1]), fill='red', outline='red')
    
    draw_img_np = np.asarray(image_pil) / 255.0
    draw_img_np.shape, draw_img_np.shape[0]*draw_img_np.shape[1]
    draw_img_np_reshape = draw_img_np.reshape(-1, 3)
    
    ################################
    # 内参
    # 增加红色的新点云
    # 生成网格点（注意 y 在前，x 在后，与图像的行列索引一致）---->2D网格点
    ################################
    y_range = slice(0, image_pil.size[1])  # y 对应行索引
    x_range = slice(0, image_pil.size[0])  # x 对应列索引
    # print(x_range)
    # 生成网格点矩阵
    y, x = np.mgrid[y_range, x_range]
    # print(x.shape)
    # print(y.shape)
    # 组合为二维坐标点，并调整形状为 [30000, 2]
    selected_area_np = np.column_stack((x.ravel(), y.ravel()))
    selected_area_np_qici = np.hstack((selected_area_np, np.ones((selected_area_np.shape[0], 1))))
    print(selected_area_np_qici.shape)
    # 内参重复10000次，shape为[10000, 4, 4]
    intrinsics_area = np.tile(intrinsics, (selected_area_np_qici.shape[0], 1, 1))
    ################################
    # 外参
    # 转置 t 并调整为列向量（3x1），然后与 r 水平拼接
    ################################

    top = np.hstack([np.asarray(R), np.asarray(t).reshape(3, 1)])  # 3x4

    # 创建最后一行 [0,0,0,1]
    bottom = np.array([[0, 0, 0, 1]])

    # 垂直拼接生成 4x4 矩阵
    extrinsics = np.vstack([top, bottom])
    extrinsics_area = np.tile(extrinsics, (selected_area_np_qici.shape[0], 1, 1))
    
    world_coords = pixel_to_world(selected_area_np_qici, 1, 
                              intrinsics_area[:,:3, :3], 
                              extrinsics_area[:, :3, :3], 
                              extrinsics_area[:,:3, 3])
    scene_points = np.vstack((scene_points, world_coords))
    pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_colors = np.vstack((scene_colors, draw_img_np_reshape))
    pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    
    # core code. Set up a set of lines to represent the camera.
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=int(data['w']), view_height_px=int(data['h']), 
                                                                   intrinsic=intrinsics[:3,:3], extrinsic=extrinsics,
                                                                   scale=0.5)
    
    vizualizer.add_geometry(cameraLines)
    id += 1
    # vizualizer.add_geometry(scene)
    # vizualizer.add_geometry(coor)
vizualizer.add_geometry(pcd)
vizualizer.add_geometry(coor)
vizualizer.run()
