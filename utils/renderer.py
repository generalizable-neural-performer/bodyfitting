import torch 
import neural_renderer as nr
import numpy as np
import os, sys, re
from tqdm import tqdm

def gen_cam_views(center, viewnum, dist, gl=False):
    def viewmatrix(z, up, translation):
        vec3 = z / np.linalg.norm(z)
        up = up / np.linalg.norm(up)
        vec1 = np.cross(up, vec3)
        vec2 = np.cross(vec3, vec1)
        view = np.stack([vec1, vec2, vec3, translation], axis=1)
        view = np.concatenate([view, np.array([[0,0,0,1]])], axis=0)
        return view

    cam_poses = []
    cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) if gl else np.eye(4)
    for theta in np.linspace(0, 2*np.pi, viewnum+1)[:-1]:
        z = np.array([np.cos(theta), 0, -np.sin(theta)]) * dist
        t = z + center
        up = np.array([0,1,0])
        view = viewmatrix(z, up, t)
        cam_poses.append(cv2gl @ np.linalg.inv(view))
    return cam_poses

def render_texture_mesh(file, imgsize=512, viewnum=8, white_bkgd=False, pose_only=False):
    vert, face, tex = nr.load_obj(file, normalization=False, load_texture=True)
    center = (torch.max(vert, 0)[0] + torch.min(vert, 0)[0]).detach().cpu().numpy() / 2.0
    bound = (torch.max(vert, 0)[0] - torch.min(vert, 0)[0]).detach().cpu().numpy()
    vert = vert.unsqueeze(0)
    tex = tex.unsqueeze(0)
    face = face.unsqueeze(0)

    
    K = np.array([[imgsize, 0, imgsize/2],[0, imgsize, imgsize/2], [0, 0, 1]])
    height = bound[1]
    # to render a human in image with 0.8 img height
    dist = height / 0.8
    poses = gen_cam_views(center, viewnum, dist, gl=True)
    Ks = [K for _ in range(viewnum)]
    if pose_only:
        return poses, Ks
    imgs, masks = [], []
    for K, pose in zip(Ks, poses):
        t = pose[:3, 3].reshape([1, 1, 3])
        R = pose[:3, :3].reshape([1, 3, 3])
        K = K.reshape([1, 3, 3])
        # print(f"rendering {i}th frame....")
        renderer = nr.Renderer(image_size=imgsize, fill_back=False, camera_mode='projection', K=K, R=R, t=t,
                            orig_size=imgsize, light_intensity_ambient=1.0, light_intensity_directional=0,
                            near=0, far=2*dist)
        images, depth, _ = renderer(vert.cuda(), face.cuda(), tex)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        image = (np.clip(image[:imgsize, :imgsize, :], 0, 1) * 255).astype(np.uint8)
        depth = depth.detach().cpu().numpy()[0]
        mask = (np.clip((depth[:imgsize, :imgsize] < 2*dist), 0, 1) * 255).astype(np.uint8)
        if white_bkgd:
            image = image + (255 - mask[...,None])
        imgs.append(image)
        masks.append(mask)
    return imgs, masks, poses, Ks
