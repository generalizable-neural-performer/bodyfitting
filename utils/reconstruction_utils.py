"""
reconstruction utilities
"""
import numpy as np
import os
import cv2
import json
# import poseviz

import constants
import utils.camera as camera

copy2cpu = lambda tensor: tensor.detach().cpu().numpy()


def ColorizeDepth(img_depth, near=1000, far=8000):
    median = np.median(img_depth)
    depth_norm = ((img_depth - near) / (far - near) * 255)
    depth_norm[depth_norm > 255] = 255
    depth_norm[depth_norm < 0] = 0
    depth_norm = depth_norm.astype(np.uint8)
    color_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return color_depth


def projection(points, fx, fy, cx, cy):
    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[1, 1] = fy
    K[2, 2] = 1.
    K[:-1, -1] = [cx, cy]
    p2d = np.dot(points, K.T)
    d = np.expand_dims(p2d[:, 2], axis=1)
    p2d = p2d[:, :2] / (d + 1e-5)
    return p2d


def get_keypoints_from_json(filename):
    np.set_printoptions(suppress=True)
    image_idx = "000018"
    imagename = "rgb_{}".format(image_idx)
    imagedir = os.path.join("/data/human_mesh/apose-jsonfile/APose_zzh_0306", imagename)
    output_dir = os.path.join("/home/SENSETIME/caozhijie/workspace/code/SPIN/examples/apose", imagename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    openpose_file = os.path.join(imagedir, "0_keypoints.json")
    with open(openpose_file, 'r') as f:
        full_json = json.load(f)
        keypoints = full_json['people'][0]['pose_keypoints_2d']
        keypoints = np.reshape(np.array(keypoints), (-1, 3))
    with open(os.path.join(output_dir, "0_keypoints.json"), "w") as f:
        json.dump(full_json, f)
    keypoints_xy = keypoints[:, :-1]
    keypoints_conf = keypoints[:, -1].reshape(-1, 1)
    img = cv2.imread(os.path.join(imagedir, imagename + ".png"))
    img = cv2.transpose(img)
    img = cv2.flip(img, 1)
    print("img", img.shape)
    cv2.imshow("examples/test/img.png", img)
    cv2.waitKey()
    return keypoints


def remove_outlier(pcd, nb_neighbors, std_ratio):
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                        std_ratio=std_ratio)
    inlier_cloud = voxel_down_pcd.select_down_sample(ind)
    return inlier_cloud


def get_pointcloud_from_depthmap(depthmap, seg_img, K, width, height, ratio=1.0, img=None, vis=False):
    import open3d as o3d
    seg_img = cv2.resize(seg_img.copy(), (0, 0), fx=ratio, fy=ratio)
    mask = seg_img > 0
    depthmap[~mask] = 0
    fx, fy = K[0][0] * ratio, K[1][1] * ratio
    cx, cy = K[0][2] * ratio, K[1][2] * ratio
    depth_camera = {'height': height * ratio,
                    'width': width * ratio,
                    'rgb': np.array([[fx, 0.00000, cx],
                                     [0.00000, fy, cy],
                                     [0.00000, 0.00000, 1.00000]]),
                    'dist_rgb': np.array([0, 0, 0, 0, 0])
                    }
    s2d = []
    depth_list = []
    for i in range(depthmap.shape[0]):
        for j in range(depthmap.shape[1]):
            s2d.append([j, i])
            dp = depthmap[i][j]
            depth_list.append(dp)
    s3d = camera.reproject3d(np.array(s2d), np.array(depth_list), depth_camera)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(s3d)
    pcd = remove_outlier(pcd, 50, 0.8)
    pcd = remove_outlier(pcd, 1000, 1.0)
    if vis:
        o3d.visualization.draw_geometries([pcd])
    cloud = np.asarray(pcd.points)
    cloud = cloud / 1000
    rot = cv2.Rodrigues(np.array([-2.0499682, 2.1432286, -0.1052672]).reshape(1, 3))[0]
    rot2 = cv2.Rodrigues(np.array([3.1415927, 0, 0]).reshape(1, 3))[0]
    cloud = np.expand_dims(cloud, axis=0)
    if img is not None and vis:
        cloud_vis = projection(s3d, fx, fy, cx, cy)
        img_vis = cv2.resize(img.copy(), (0, 0), fx=ratio, fy=ratio)
        depth_vis = depthmap.copy()
        depth_vis = ColorizeDepth(depth_vis)
        color_mask = np.array([[255, 255, 0]], dtype=np.uint8)
        depth_vis[mask] = depth_vis[mask] * 0.3 + color_mask * 0.3 + img_vis[mask] * 0.3
        cv2.imshow("examples/test/depth.png", depth_vis)
        cv2.waitKey()
        import poseviz
        cp = poseviz.draw_s2d_as_points(cloud_vis, image=img_vis, text=False)
        cv2.imshow("cp", cp)
        cv2.waitKey()
    return cloud


def transfer_camera_translation(camera_t, crop_center, crop_scale):
    cx, cy = constants.CX, constants.CY
    fx, fy = constants.FX, constants.FY
    vfx = constants.FOCAL_LENGTH
    scale = 1 / vfx / crop_scale / 200 * 224
    camera_t = camera_t + np.array([(crop_center[0] - cx) * scale,
                    (crop_center[1] - cy) * scale, 0]) * camera_t[2]
    camera_t = camera_t * np.array([1, 1, fx * scale])
    return camera_t


def visual_cloud(points, fx, fy, cx, cy, vis_cloud=False):
    import open3d as o3d
    # points_arr = copy2cpu(points)
    # points_vis = np.squeeze(points_arr, axis=0)
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(points)
    points_vis = points + [0, 0, 3]
    src_vis = projection(points_vis, fx, fy, cx, cy)
    print(src_vis)
    import poseviz
    cp = poseviz.draw_s2d_as_points(src_vis, size=1000, text=True)
    cp = cv2.resize(cp, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("cp", cp)
    cv2.waitKey()
    if vis_cloud:
        o3d.visualization.draw_geometries([pcd_a])


def visualize_clouds(vert_src_list, p3d=None, joint3d=None):
    import open3d as o3d
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    pcd_list = []
    color_list = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for idx, vert_src in enumerate(vert_src_list):
        vert_src_arr = vert_src.cpu().detach().numpy()
        vert_src_vis = np.squeeze(vert_src_arr, axis=0)
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(vert_src_vis)
        pcd_a.colors = o3d.utility.Vector3dVector(np.ones((vert_src_vis.shape[0], 3)) * color_list[idx % 3])
        pcd_list.append(pcd_a)
        viz.add_geometry(pcd_a)
    if p3d is not None:
        p3d_list = []
        for i in range(p3d.shape[0]):
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.01)
            trans = np.eye(4)
            trans[:3, 3] = p3d[i]
            print(trans)
            mesh.transform(trans)
            p3d_list.append(mesh)
            viz.add_geometry(mesh)
    if joint3d is not None:
        p3d_list = []
        for i in range(joint3d.shape[0]):
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.01)
            trans = np.eye(4)
            trans[:3, 3] = joint3d[i]
            print(trans)
            mesh.transform(trans)
            color = np.array([1.0, 0.0, 0.0])
            mesh.paint_uniform_color(color)
            p3d_list.append(mesh)
            viz.add_geometry(mesh)
    viz.run()


def compact_bound_p2d(p2d):
    maxx, maxy = np.max(p2d, axis=0)
    minx, miny = np.min(p2d, axis=0)
    center = [0.5 * (maxx + minx), 0.5 * (maxy + miny)]
    scale = 0.5 * max(maxx - minx, maxy - miny) * 1.2
    c2d = (p2d - center) / scale
    return c2d


def draw_s3d_as_points(p3d=None):
    import open3d as o3d
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    if p3d is not None:
        p3d_list = []
        for i in range(p3d.shape[0]):
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=0.01)
            trans = np.eye(4)
            trans[:3, 3] = p3d[i]
            mesh.transform(trans)
            p3d_list.append(mesh)
            viz.add_geometry(mesh)
    viz.run()


def draw_s3d_as_points_list(list):
    import open3d as o3d
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    for p3d in list:
        p3d_list = []
        for i in range(p3d.shape[0]):
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=0.01)
            trans = np.eye(4)
            trans[:3, 3] = p3d[i]
            mesh.transform(trans)
            p3d_list.append(mesh)
            viz.add_geometry(mesh)
    viz.run()