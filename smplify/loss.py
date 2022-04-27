import os
import pickle

import numpy as np
import cv2
# import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
# import poseviz
from torchgeometry import angle_axis_to_rotation_matrix
from utils.mesh_grid_searcher import MeshGridSearcher

# from utils.geometry import projection, uncrop_kpts, compute_normal
from utils.reconstruction_utils import copy2cpu
import constants
SKELETON_LENGTH = 25
HANDS_LENGTH = 42
FACE_LENGTH = 68
FACE_MAPPING = list(range(17, 17+51)) + list(range(0, 17))

def perspective_projection(points, rotation, translation, K):
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
    if isinstance(K, np.ndarray):
        K = torch.tensor(K, dtype=torch.float32, device=points.device)
    K = K[None,...].expand([batch_size,-1,-1])

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = torch.einsum('bij,bkj->bki', K, points)
    projected_points = projected_points / projected_points[:,:,-1].unsqueeze(-1)

    return projected_points[:, :, :-1]

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    # return torch.exp(pose[:, [55-3, 58-3, 12-3, 13-3, 14-3, 15-3, 16-3, 17-3]] * torch.tensor(np.ones(8, dtype=np.float32), device=pose.device)) ** 2
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def get_cos(vec_x, vec_y):
    ans = -torch.cos(torch.mul(vec_x, vec_y) / torch.norm(vec_x) / torch.norm(vec_y))
    return ans


def get_direction(vecx):
    norm = torch.norm(vecx, dim=1)
    return vecx / norm

def extract_countours(masks):
    """
    Extract the external contour of silhouette points from mask
    """
    contours = []
    for mask in masks:
        _, contour, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contour[np.argmax(np.array([a.shape[1] for a in contour]))]
        contour = torch.tensor(contour, dtype=torch.float32, device=masks.device)
        contours.append(contour)
    return contours

def multview_mask_loss(contours, masks, smpl_verts, smpl_faces, w2cs, Ks, mask_frames, epsilon=10, imsize=512):
    # print(masks.shape, smpl_verts.shape, smpl_faces.shape, smpl_faces.shape, Ks.shape)
    """
    Multiview maks loss for projected smpl vertices and image contours

    contours: the outer contours extracted from per view mask
    smpl_verts: smpl vertices [1, N, 3]
    smpl_faces: smpl faces [1, N, 3]
    w2cs: world to camera rotation matrix wrt. contour view
    Ks: intrinsic matrix wrt. contour view
    epsilon: penalty coefficient of 2d icp loss for points outside the mask
    imsize: image size
    """
    scale_coeff = 1 # imsize / 256  # for image size invariant
    smpl_verts = smpl_verts.squeeze(0)[::4]
    smpl_faces = torch.tensor(smpl_faces[0], dtype=torch.int64, device=smpl_verts.device)
    losses, uvs = [], []
    for i, frame_id in enumerate(mask_frames):
        pose, K, contour, mask = w2cs[i], Ks[i], contours[i], masks[i]
        projected_points = perspective_projection(smpl_verts.unsqueeze(0), pose[None, :3, :3], pose[None, :3, 3], K).squeeze(0)
        inside = torch.prod((projected_points < imsize) & (projected_points >= 0), dim=1).squeeze(0) > 0
        inside_points = projected_points[inside]
        uvs.append(projected_points)

        # find the contour's closest smpl points 
        dist = torch.cdist(inside_points.unsqueeze(0)/scale_coeff, contour.unsqueeze(0)/scale_coeff).squeeze(0)        
        mindist, index = torch.min(dist, 1)
        
        # closest points inside the mask or not
        clost_points = inside_points[index[:, 0]].long()
        outside_mask = (mask[clost_points[:, 1], clost_points[:, 0]] < 0.1).float()[:, None]
        coeff = outside_mask * (epsilon - 1) + 1
        closest_dist = torch.sum(mindist * coeff)
        losses.append(closest_dist)

    masks_loss = torch.stack(losses).sum()

    # add differentiable binary mask loss to regularize the contour loss
    uvs = torch.stack(uvs, dim=0)
    uvs = uvs.view(len(masks), -1, 1, 2) / imsize * 2 - 1
    masks = masks[:, None]
    binary_dist = torch.nn.functional.grid_sample(1 - masks, uvs)
    binary_dist = torch.sum(binary_dist) * epsilon
    masks_loss += binary_dist

    return masks_loss

def reprojection_loss(cord, cord_gt, conf, scale_coeff, sigma):
    reprojection_error = gmof((cord_gt - cord) / scale_coeff, sigma)
    reprojection_loss = (conf ** 2) * reprojection_error.sum(dim=-1)
    reprojection_loss = reprojection_loss.sum(dim=-1)
    return reprojection_loss


def multiview_keypoint_loss(w2cs, Ks, keypoints, model_joints, poses, betas, use_frames, pose_prior, sigma=100, 
                            shape_prior_weight=5, angle_prior_weight=15.2, output='sum', debug=False, imsize=512, 
                            pose_prior_weight=4.78, use_hand_face=False, output_folder=None, verts=None):
    """
    Multiview keypoint loss for projected smpl joints and detected keypoints in each view
    
    w2cs: world to camera matrices
    Ks: intrinsic matrices
    keypoints: keypoints dictionary loaded from openpose\
    model_joints: smpl joints 3d locations
    poses, betas: smpl parameters for regularization
    pose_prior: GMM prior model
    use_frames: list of frames used
    """
    device = model_joints.device
    body_loss, hand_loss, face_loss = [], [], []
    scale_coeff = imsize / 1024 # for image size invariant
    for i, frame_id in enumerate(use_frames):
        if keypoints[i] is not None:
            w2c = w2cs[i].to(device)
            projected_joints = perspective_projection(model_joints, w2c[:3,:3].unsqueeze(0), w2c[:3, 3].unsqueeze(0), Ks[i])
            skeleton_keypoints = torch.from_numpy(keypoints[i]['pose']).float().to(device)
            cord_gt, conf = torch.split(skeleton_keypoints, [2,1], dim=-1)
            conf = conf.squeeze(-1)
            cord = projected_joints[0, :SKELETON_LENGTH]
            body_loss.append(reprojection_loss(cord, cord_gt, conf, scale_coeff, sigma))
            if use_hand_face:
                if 'hand_left' in keypoints[i].keys():
                    hand_keypoints = torch.from_numpy(keypoints[i]['hand_left']).float().to(device)
                    cord_gt, conf = torch.split(hand_keypoints, [2,1], dim=-1)
                    cord = projected_joints[0, SKELETON_LENGTH:SKELETON_LENGTH+HANDS_LENGTH//2]
                    hand_loss.append(reprojection_loss(cord, cord_gt, conf, scale_coeff, sigma))
                if 'hand_right' in keypoints[i].keys():
                    hand_keypoints = torch.from_numpy(keypoints[i]['hand_right']).float().to(device)
                    cord_gt, conf = torch.split(hand_keypoints, [2,1], dim=-1)
                    cord = projected_joints[0, SKELETON_LENGTH+HANDS_LENGTH//2:SKELETON_LENGTH+HANDS_LENGTH]
                    hand_loss.append(reprojection_loss(cord, cord_gt, conf, scale_coeff, sigma))
                if 'face' in keypoints[i].keys():
                    face_keypoints = keypoints[i]['face']
                    face_keypoints = torch.from_numpy(face_keypoints[FACE_MAPPING]).float().to(device)
                    cord_gt, conf = torch.split(face_keypoints, [2,1], dim=-1)
                    cord = projected_joints[0, SKELETON_LENGTH+HANDS_LENGTH:]
                    face_loss.append(reprojection_loss(cord, cord_gt, conf, scale_coeff, sigma))
            
        # # To visualize the reprojection error, please uncomment this block
        # if debug and output_folder is not None:
        #     proj_verts = perspective_projection(verts, w2c[:3,:3].unsqueeze(0), w2c[:3, 3].unsqueeze(0), Ks[i])
        #     cp = cv2.imread(os.path.join(os.path.dirname(output_folder), 'images/{:02d}.png'.format(frame_id)))
        #     cp = cv2.putText(cp, f'{i}', (100, 100), cv2.FONT_ITALIC, 2, (0,0,255), 6)
        #     pj_cpu = proj_verts.detach().cpu().numpy().astype(np.int32)[0]
        #     j2_cpu = skeleton_keypoints[:, :2].detach().cpu().numpy().astype(np.int32)
        #     pj_cpu = np.clip(pj_cpu, 0, imsize-1)
        #     j2_cpu = np.clip(j2_cpu, 0, imsize-1)
        #     cp[pj_cpu[:,1], pj_cpu[:,0]] = [0, 255, 0]
        #     cp[j2_cpu[:,1], j2_cpu[:,0]] = [255, 0, 0]
        #     cv2.imshow("cp", cp)
        #     cv2.waitKey()

    body_loss = torch.sum(torch.stack(body_loss, dim=0)) / len(use_frames)
    loss_2d = body_loss
    if use_hand_face:
        hand_loss = torch.sum(torch.stack(hand_loss, dim=0)) / len(use_frames)
        loss_2d += hand_loss
        face_loss = torch.sum(torch.stack(face_loss, dim=0)) / len(use_frames)
        loss_2d += face_loss
    # print(body_loss, hand_loss, face_loss)

    if use_hand_face:
        poses = torch.cat([poses, torch.zeros_like(poses[:,:6])], dim=-1)
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(poses, None)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(poses).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = loss_2d + pose_prior_loss + angle_prior_loss + \
                 shape_prior_loss

    losses = {
        "reprojection_loss": copy2cpu(loss_2d),
        "pose_prior_loss": copy2cpu(pose_prior_loss),
        "angle_prior_loss": copy2cpu(angle_prior_loss),
        "shape_prior_loss": copy2cpu(shape_prior_loss),
    }
    # print("reprojection_loss: ", loss_2d.item(), "pose_prior_loss", pose_prior_loss.item(), \
    #       "angle_prior_loss: ", angle_prior_loss.item(), "shape_prior_loss", shape_prior_loss.item())
    if output == 'sum':
        return total_loss.sum(), losses
    elif output == 'reprojection':
        return reprojection_loss, losses


def point_cloud_loss_mesh_grid(mesh_grid_searcher: MeshGridSearcher, points: torch.Tensor):
    """
    Point cloud loss (point to mesh distance) to fit SMPL to a mesh
    mesh_grid_searcher: a MeshGridSearch who has loaded the target mesh
    points: points to fit
    """
    closest_mesh_point, closest_mesh_face = mesh_grid_searcher.nearest_points(points.view(-1,3))
    closest_distance = torch.norm(points.view(-1,3) - closest_mesh_point.detach(), p=2)
    point_cloud_loss = torch.mean(closest_distance)
    return point_cloud_loss


def point_cloud_loss_chamfer_naive(pts_src, pts_tar, samples=500):
    """
    Point cloud loss (chamfer distance between two point clouds) to fit SMPL to mesh vertices
    verts_src: SMPL vertices
    verts_tar: mesh vertices
    """
    pts_src, pts_tar = pts_src.view(-1,3), pts_tar.view(-1,3)
    src_samples_idx = torch.randint(len(pts_src), (samples,), device=pts_src.device)
    tar_samples_idx = torch.randint(len(pts_src), (samples,), device=pts_src.device)
    src, tar = pts_src[src_samples_idx], pts_tar[tar_samples_idx]
    dist_mat = torch.cdist(src.unsqueeze(0), tar.unsqueeze(0), p=2) ** 2
    chamfer_distance = torch.min(dist_mat, dim=1)[0] + torch.min(dist_mat, dim=2)[0]
    chamfer_loss = torch.mean(chamfer_distance, dim=-1)[0]
    return chamfer_loss

def normal_loss_mesh_grid(mesh_grid_searcher: MeshGridSearcher, points: torch.Tensor, 
                          face_norm_mesh: torch.Tensor, point_norm: torch.Tensor):
    """
    Normal loss (point normal to plan normal distance) to fit deformed SMPL to a mesh
    mesh_grid_searcher: a MeshGridSearch who has loaded the target mesh
    points: points to fit
    """
    closest_mesh_point, closest_mesh_face = mesh_grid_searcher.nearest_points(points.view(-1,3))
    closest_face_norm = face_norm_mesh[closest_mesh_face.long()]
    normal_loss = torch.mean(1-torch.sum(closest_face_norm * point_norm, dim=-1))
    
    return normal_loss

def normal_laplacian_smoothness(norms, faces):
    """
    Laplacian smoothness loss (1-hoc) 
    norms: normals of vertices
    faces: topology for vertices connections
    """
    mse = lambda x, y: torch.sum((x-y)**2, dim=-1)
    norms, faces = norms.view(-1,3), faces.view(-1,3)

    # Compute the 1-hoc graph edge weight summation as an 
    # implementation of F-norm of Laplacian matrix
    na = torch.index_select(norms, 0, faces[:,0])
    nb = torch.index_select(norms, 0, faces[:,1])
    nc = torch.index_select(norms, 0, faces[:,2])
    smoothness_loss = torch.mean(mse(na, nb) + mse(nc, na) + mse(nb, nc))
    return smoothness_loss