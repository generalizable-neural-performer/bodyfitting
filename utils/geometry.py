import os
import torch
from torch.nn import functional as F
import numpy as np
import cv2
# import poseviz

import constants
from utils.reconstruction_utils import copy2cpu


def batch_euler2rot(eulers):
    sin, cos = torch.sin, torch.cos
    phi, theta, psi = eulers[:, 0], eulers[:, 1], eulers[:, 2]
    batch = eulers.shape[0]
    R1 = torch.eye(3)[None,...].expand([batch, 3, 3]).to(eulers.device)
    R1[:, 1, 1] = cos(phi)
    R1[:, 1, 2] = sin(phi)
    R1[:, 2, 1] = -sin(phi)
    R1[:, 2, 2] = cos(phi)
    R2 = torch.eye(3)[None,...].expand([batch, 3, 3]).to(eulers.device)
    R2[:, 0, 0] = cos(theta)
    R2[:, 0, 2] = -sin(theta)
    R2[:, 2, 0] = sin(theta)
    R2[:, 2, 2] = cos(theta)
    R3 = torch.eye(3)[None,...].expand([batch, 3, 3]).to(eulers.device)
    R3[:, 0, 0] = cos(psi)
    R3[:, 0, 1] = sin(psi)
    R3[:, 1, 0] = -sin(psi)
    R3[:, 1, 1] = cos(psi)
    R = R1 @ R2 @ R3
    return R

def euler2rot(euler):
    sin, cos = np.sin, np.cos
    phi, theta, psi = euler[0], euler[1], euler[2]
    R1 = np.array([[1, 0, 0],
                   [0, cos(phi), sin(phi)],
                   [0, -sin(phi), cos(phi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)],
                   [0, 1, 0],
                   [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(psi), sin(psi), 0],
                   [-sin(psi), cos(psi), 0],
                   [0, 0, 1]])
    R = R1 @ R2 @ R3
    return R

def rot2euler(R):
    phi = np.arctan2(R[1,2], R[2,2])
    theta = -np.arcsin(R[0,2])
    psi = np.arctan2(R[0,1], R[0,0])
    return np.array([phi, theta, psi])


"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""
def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center, K=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    if K is None:
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:,0,0] = focal_length
        K[:,1,1] = focal_length
        K[:,2,2] = 1.
        K[:,:-1, -1] = camera_center
    else:
        K = torch.tensor(K, dtype=torch.float32, device=points.device)[None,...].expand([batch_size,-1,-1])

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    # projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, points)
    projected_points = projected_points / projected_points[:,:,-1].unsqueeze(-1)

    return projected_points[:, :, :-1]


def perspective_projection_ext(points, rotation, translation,
                           focal, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs, fx, fy) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal[0]
    K[:,1,1] = focal[1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, img_size=224):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000., img_size=224.):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device
    # Use only joints 25:49 (GT joints)
    S = S[:, 25:, :].cpu().numpy()
    joints_2d = joints_2d[:, 25:, :].cpu().numpy()
    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size)
    return torch.from_numpy(trans).to(device)


def projection(points, fx, fy, cx, cy):
    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[1, 1] = fy
    K[2, 2] = 1.
    K[0, 2] = cx
    K[1, 2] = cy
    p2d = np.dot(points, K.T)
    d = np.expand_dims(p2d[:, 2], axis=1)
    p2d = p2d[:, :2] / (d + 1e-5)
    return p2d


def uncrop_kpts(p2d, crop_center, crop_scale, crop_img_center=[112, 112]):
    crop_scale = crop_scale * 200 / 224
    uncrop_p2d = (p2d - crop_img_center) * crop_scale + crop_center
    return uncrop_p2d


def visualize_clouds_with_source(vert_src, vert_dst, img, K, output_folder):
    import open3d as o3d
    cx, cy = K[0][2], K[1][2]
    fx, fy = K[0][0], K[1][1]
    vert_src_arr = copy2cpu(vert_src)
    vert_dst_arr = copy2cpu(vert_dst)
    vert_src_vis = np.squeeze(vert_src_arr, axis=0)
    vert_dst_vis = np.squeeze(vert_dst_arr, axis=0)
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(vert_src_vis)
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(vert_dst_vis)
    src_vis = projection(vert_src_vis, fx, fy, cx, cy)
    dst_vis = projection(vert_dst_vis, fx, fy, cx, cy)
    cp = poseviz.draw_s2d_as_points(src_vis, image=img, text=False)
    cp = poseviz.draw_s2d_as_points(dst_vis, image=cp, text=False, color="#FF00FF")
    cp = cv2.resize(cp, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("cp", cp)
    cv2.waitKey()
    o3d.visualization.draw_geometries([pcd_a, pcd_b])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    o3d.io.write_point_cloud(os.path.join(output_folder, "smpl_world_coord.xyz"), pcd_a)
    o3d.io.write_point_cloud(os.path.join(output_folder, "cloud_world_coord.xyz"), pcd_b)


def visualize_clouds(vert_src_list, p3d=None, joint3d=None):
    import open3d as o3d
    pcd_list = []
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    for vert_src in vert_src_list:
        vert_src_arr = vert_src.cpu().detach().numpy()
        vert_src_vis = np.squeeze(vert_src_arr, axis=0)
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(vert_src_vis)
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


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        # >>> input = torch.rand(2, 3, 4)  # Nx4x4
        # >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        # >>> quaternion = torch.rand(2, 4)  # Nx4
        # >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        # >>> input = torch.rand(4, 3, 4)  # Nx3x4
        # >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def convert_hom_to_angle(pred_rotmat, batch_size, device):
    # Convert predicted rotation matrices to axis-angle
    pred_rotmat_hom = torch.cat(
        [pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0, 0, 1], dtype=torch.float32,
                                                                    device=device).view(1, 3, 1).expand(batch_size * 24,
                                                                                                        -1, -1)],
        dim=-1)
    pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
    # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
    pred_pose[torch.isnan(pred_pose)] = 0.0
    return pred_pose


def get_global_camera_translation(camera_translation, zoom_scale, crop_center, crop_scale, K):
    cx = K[0][0][2]
    cy = K[0][1][2]
    fx = K[0][0][0]
    vfx = constants.FOCAL_LENGTH
    scale = 1 / vfx / crop_scale / 200 * 224
    camera_translation_global = camera_translation + np.array([(crop_center[0] - cx) * scale,
                                        (crop_center[1] - cy) * scale, 0]) * camera_translation[:, 2]
    camera_translation_global = camera_translation_global * np.array([1, 1, fx * scale]) * zoom_scale
    return camera_translation_global


# def get_local_camera_translation(camera_translation_global, zoom_scale, crop_center, crop_scale, K):
#     cx = K[0][2]
#     cy = K[1][2]
#     fx = K[0][0]
#     vfx = 5000
#     scale = 1 / vfx / crop_scale / 200 * 224
#     camera_translation_local = camera_translation_global / np.array([1, 1, fx * scale]) / zoom_scale
#     camera_translation_local = camera_translation_local - np.array([(crop_center[0] - cx) * scale,
#                                                                (crop_center[1] - cy) * scale, 0]) * \
#                                camera_translation_local[:, 2]
#     return camera_translation_local

def get_local_camera_translation(camera_translation_global, zoom_scale, crop_center, crop_scale, K):
    cx = K[0][0][2]
    cy = K[0][1][2]
    fx = K[0][0][0]
    vfx = constants.FOCAL_LENGTH
    scale = 1 / vfx / crop_scale / 200 * 224
    camera_translation_local = camera_translation_global / np.array([1, 1, fx * scale]) / zoom_scale
    camera_translation_local = camera_translation_local - np.array([(crop_center[0] - cx) * scale,
                                                               (crop_center[1] - cy) * scale, 0]) * \
                               camera_translation_local[:, 2]
    return camera_translation_local


def convert_scale_to_space(pred_camera):
    rtn = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                 2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                dim=-1)
    return rtn


def convert_space_to_scale(pred_cam_t, img_res=constants.IMG_RES):
    rtn = torch.stack([2 * constants.FOCAL_LENGTH / (img_res * pred_cam_t[:, 2] + 1e-9),
            pred_cam_t[:, 0], pred_cam_t[:, 1]], dim=-1)
    return rtn


def draw_s3d_as_points(p3d_list=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    color = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    ax.set_xlabel('X')
    ax.set_xlim(-100, 100)
    ax.set_ylabel('Y')
    ax.set_ylim(-100, 100)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)
    for idx, p3d in enumerate(p3d_list):
        xs, ys, zs = [], [], []
        label = []
        for i in range(p3d.shape[0]):
            x, y, z = p3d[i]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            label.append(i)
            ax.text(x, y, z, i)
        ax.scatter(xs, ys, zs, c=color[idx % 3])
    plt.show()


def draw_s3d_as_points_o3d(vertices=None, p3d_list=None):
    import open3d as o3d
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    if vertices is not None:
        vert_src_arr = vertices.cpu().detach().numpy()
        vert_src_vis = np.squeeze(vert_src_arr, axis=0)
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(vert_src_vis)
        viz.add_geometry(pcd_a)
    p3d_viz = []
    color = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for idx, p3d in enumerate(p3d_list):
        for i in range(p3d.shape[0]):
            mesh = o3d.geometry.TriangleMesh.create_cylinder()
            trans = np.eye(4)
            trans[:3, 3] = p3d[i]
            mesh.transform(trans)
            mesh.paint_uniform_color(color[idx%3])
            p3d_viz.append(mesh)
            viz.add_geometry(mesh)
    viz.run()


def check_in_view(nx, ny, shape):
    if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
        return True
    return False


def modify_rend_depth(rend_depth):
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    expend_rend_depth = np.zeros_like(rend_depth, dtype=np.int)
    for i in range(rend_depth.shape[0]):
        for j in range(rend_depth.shape[1]):
            if rend_depth[i][j] < 100.0:
                is_edge_point = False
                for k in range(4):
                    nx = i + dx[k]
                    ny = j + dy[k]
                    if check_in_view(nx, ny, rend_depth.shape) and rend_depth[nx][ny] <= 0.0:
                        is_edge_point = True
                if is_edge_point:
                    expend_rend_depth[i][j] = 1
    rend_depth[expend_rend_depth == 1] = 0.0
    return rend_depth


def visualize_clouds_with_points(vert_src_list, p3d=None, joint3d_list=None):
    import open3d as o3d
    pcd_list = []
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    for vert_src in vert_src_list:
        vert_src_arr = vert_src.cpu().detach().numpy()
        vert_src_vis = np.squeeze(vert_src_arr, axis=0)
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(vert_src_vis)
        pcd_list.append(pcd_a)
        viz.add_geometry(pcd_a)
    if p3d is not None:
        p3d_list = []
        for i in range(p3d.shape[0]):
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.015)
            trans = np.eye(4)
            trans[:3, 3] = p3d[i]
            mesh.transform(trans)
            p3d_list.append(mesh)
            viz.add_geometry(mesh)
    if joint3d_list is not None:
        for joint3d in joint3d_list:
            p3d_list = []
            for i in range(joint3d.shape[0]):
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.015)
                trans = np.eye(4)
                trans[:3, 3] = joint3d[i]
                mesh.transform(trans)
                color = np.array([1.0, 0.0, 0.0])
                mesh.paint_uniform_color(color)
                p3d_list.append(mesh)
                viz.add_geometry(mesh)
    viz.run()
    # o3d.visualization.draw_geometries(pcd_list)
