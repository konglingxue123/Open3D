import json
from PIL import Image
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw

add_xy1 = [600, 600]
add_xy2 = [700, 700]

img_dir = '/home/ubunto/Project/konglx/pcd/projection/datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/images/image_54_1080（复件）.jpg'
# 在图片上绘制一块矩形红色区域
im = Image.open(img_dir)
draw = ImageDraw.Draw(im)
draw.rectangle((add_xy1[0], add_xy1[1], add_xy2[0], add_xy2[1]), fill='red', outline='red')



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


import numpy as np
import open3d

# 图像的rgb点
img_np = np.asarray(im) / 255.0
img_np.shape, img_np.shape[0]*img_np.shape[1]
img_np_reshape = img_np.reshape(-1, 3)



## 原始点云 ##
pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(scene_points)

# pcd.colors = open3d.utility.Vector3dVector(colors)
intrinsics = np.array([
    [data['fl_x'], 0,            data['cx'], 0],
    [0,            data['fl_y'], data['cy'], 0],
    [0,            0,            1,          0],
    [0,            0,            0,          1]
])

###########################################
# 增加红色的新点云
# 生成网格点（注意 y 在前，x 在后，与图像的行列索引一致）---->2D网格点
y_range = slice(0, im.size[1])  # y 对应行索引
x_range = slice(0, im.size[0])  # x 对应列索引
# print(x_range)
# 生成网格点矩阵
y, x = np.mgrid[y_range, x_range]
print(x.shape)
print(y.shape)
# 组合为二维坐标点，并调整形状为 [30000, 2]
selected_area_np = np.column_stack((x.ravel(), y.ravel()))
selected_area_np_qici = np.hstack((selected_area_np, np.ones((selected_area_np.shape[0], 1))))
print(selected_area_np_qici.shape)
# 内参重复10000次，shape为[10000, 4, 4]
intrinsics_area = np.tile(intrinsics, (selected_area_np_qici.shape[0], 1, 1))
print(intrinsics_area.shape)

# 找外参数矩阵
selected_image_name = 'image_54_1080'
for frame in frames_data:
    if selected_image_name in frame['file_path']:
        R = np.array(frame['R'])
        t = np.array(frame['t'])
        # 转置 t 并调整为列向量（3x1），然后与 r 水平拼接
        top = np.hstack([R, t.reshape(3, 1)])  # 3x4

        # 创建最后一行 [0,0,0,1]
        bottom = np.array([[0, 0, 0, 1]])

        # 垂直拼接生成 4x4 矩阵
        extrinsics = np.vstack([top, bottom])
    else:
        continue

extrinsics_area = np.tile(extrinsics, (selected_area_np_qici.shape[0], 1, 1))
print(extrinsics_area.shape)

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

world_coords = pixel_to_world(selected_area_np_qici, 1, 
                              intrinsics_area[:,:3, :3], 
                              extrinsics_area[:, :3, :3], 
                              extrinsics_area[:,:3, 3])


# np_rand_dot = np.random.randn(100, 3)
scene_points = np.vstack((scene_points, world_coords))
pcd.points = open3d.utility.Vector3dVector(scene_points)

# np_rand_dot_color = np.array([[1, 0, 0]] * world_coords.shape[0])
np_rand_dot_color = img_np_reshape
# 添加原始点云的颜色
colors = scene_colors
scene_colors = np.vstack((scene_colors, np_rand_dot_color))
pcd.colors = open3d.utility.Vector3dVector(scene_colors)

###########################################


# o3d.visualization.draw_geometries([pcd])


import open3d as o3d
import numpy as np
import json

WIDTH = 1920
HEIGHT = 1080


json_dir = 'datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/transforms.json'
# 读取json文件
with open(json_dir, 'r') as f:
    data = json.load(f)
frames_data = data['frames']
# print(data['h'])

# load a scene point cloud
# scene = o3d.io.read_point_cloud('/home/ubunto/Project/konglx/pcd/projection/datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/colmap_sparse/0/sparse.ply')
# scene = o3d.io.read_triangle_mesh('/home/ubunto/Project/konglx/pcd/image_to_3d/TRELLIS/trellis-outputs/tea-pot_letter/sample.glb')
# 可视化坐标轴. The x, y, z axis will be rendered as red, green, and blue arrows respectively.
coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])  
# coor.scale(10.0) 
vizualizer = o3d.visualization.Visualizer()
vizualizer.create_window(width=WIDTH, height=HEIGHT)


for frame in frames_data:
    R = np.array(frame['R'])
    t = np.array(frame['t'])
    # 转置 t 并调整为列向量（3x1），然后与 r 水平拼接
    top = np.hstack([R, t.reshape(3, 1)])  # 3x4

    # 创建最后一行 [0,0,0,1]
    bottom = np.array([[0, 0, 0, 1]])

    # 垂直拼接生成 4x4 矩阵
    extrinsics = np.vstack([top, bottom])

    # core code. Set up a set of lines to represent the camera.
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=int(data['w']), view_height_px=int(data['h']), 
                                                                   intrinsic=intrinsics[:3,:3], extrinsic=extrinsics,
                                                                   scale=0.5)
    
    vizualizer.add_geometry(cameraLines)
    # vizualizer.add_geometry(scene)
    # vizualizer.add_geometry(coor)
vizualizer.add_geometry(pcd)
vizualizer.add_geometry(coor)
vizualizer.run()
# 存储点云
o3d.io.write_point_cloud('scene_added_area.ply', pcd)
vizualizer.destroy_window()