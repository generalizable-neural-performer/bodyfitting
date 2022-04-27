import argparse
from distutils.log import debug
import os, sys

import torch
import imageio
import cv2
import neural_renderer as nr
import numpy as np
from models import Inpainter
from tqdm import tqdm, trange
from utils.renderer import gen_cam_views

def load_obj_uv(filename):
    vertices, uv = [], []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            uv.append([float(line.split()[1]), 1-float(line.split()[2])])
    uv = np.vstack(uv).astype(np.float32)

    # load faces for textures of deformed smpl
    faces = []
    faces_t = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[0])
                vt0 = int(vs[0].split('/')[1])
            else:
                v0, vt0 = 0, 0
            
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[0])
                    vt1 = int(vs[i + 1].split('/')[1])
                else:
                    v1, vt1 = 0, 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[0])
                    vt2 = int(vs[i + 2].split('/')[1])
                else:
                    v2, vt2 = 0, 0
                faces.append((v0, v1, v2))
                faces_t.append((vt0, vt1, vt2))
                material_names.append(material_name)
    faces = np.vstack(faces).astype(np.int32) - 1
    faces_t = np.vstack(faces_t).astype(np.int32) - 1
    face_uv = uv[faces_t]
    return face_uv


to8b = lambda x: (np.clip(np.flip(x,2),0,1)*255).astype(np.uint8)

def sphere2rot(rad, theta, phi, t=[0,0,0]):
    sin, cos, pi = np.sin, np.cos, np.pi
    normalize = lambda x: x / np.linalg.norm(x)
    def view_matrix(look, right):
        z = normalize(look)
        # x = normalize(right)
        y = normalize(np.cross(z, right))
        x = normalize(np.cross(y, z))
        rot = np.stack([x, y, z], axis=1)
        return rot
    sin, cos, pi = np.sin, np.cos, np.pi
    transl = np.array([rad*sin(theta)*sin(phi), rad*cos(theta), rad*sin(theta)*cos(phi)])
    look = -transl
    right = np.array([sin(phi+pi/2), 0, cos(phi+pi/2)])
    rot = view_matrix(look, right)
    R = np.eye(4)
    R[:3,:3] = rot
    R[:3, 3] = transl + np.array(t)
    return R

def create_smpld_uv(output_dir, smpld_dir, smpl_uv_dir, tex_img_size):
    """
    Create deformed smpl obj file with uv and default texture
    """
    new_lines = []
    with open(smpld_dir, 'r') as f_v:
        lines = f_v.readlines()
        for line in lines:
            if line.startswith("v "):
                new_lines.append(line)
    with open(smpl_uv_dir, 'r') as f_uv:
        lines = f_uv.readlines()
        for line in lines:
            if line.startswith('mtllib'):
                mtl = os.path.join(os.path.dirname(smpl_uv_dir), line.split()[1])
                new_mtl = os.path.join(os.path.dirname(output_dir), line.split()[1])
                print(f"cp {mtl} {new_mtl}")
                os.system(f"cp {mtl} {new_mtl}")
                with open(mtl, 'r') as f_mtl:
                    mtl_lines = f_mtl.readlines()
                    for m_line in mtl_lines:
                        if m_line.startswith("newmtl"):
                            mtl_name = m_line.split()[1]
                        elif m_line.startswith("map_Kd"):
                            tex_dir = m_line.split()[1]
                new_lines = new_lines + [f"mtllib {line.split()[1]}\n", f"usemtl {mtl_name}\n"]
                tex_img = np.ones([tex_img_size,tex_img_size,3], dtype=np.uint8) * 128
                imageio.imwrite(os.path.join(os.path.dirname(output_dir), tex_dir), tex_img)
            if line.startswith("f "):
                new_lines.append(line)
            elif line.startswith("vt "):
                new_lines.append(line)
    with open(output_dir, 'w') as f_smpl:
        f_smpl.writelines(new_lines)
        
def render_compare(renderer, smpl, scan, poses, outdir, logging=False):
    """
    Render comparison of smpl and scan in round views
    """
    scan_v, scan_f, scan_t = scan
    smpl_v, smpl_f, smpl_t = smpl
    # render_poses = gen_cam_traj(center, viewnum, dist)
    render_imgs = []
    
    with torch.no_grad():
        for i, pose in enumerate(poses):
            if logging:
                print(f"rendering {i}th view")
            t = pose[:3, 3].reshape([1, 1, 3])
            R = pose[:3, :3].reshape([1, 3, 3])
            R, t = torch.cuda.FloatTensor(R), torch.cuda.FloatTensor(t)

            scan_img = renderer.render_rgb(scan_v, scan_f, scan_t, R=R, t=t)
            smpl_img = renderer.render_rgb(smpl_v, smpl_f, smpl_t, R=R, t=t)
            
            smpl_img = smpl_img.detach().cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]
            smpl_img = to8b(smpl_img)
            
            scan_img = scan_img.detach().cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]
            scan_img = to8b(scan_img)
            img = np.hstack((scan_img, smpl_img))
            imageio.imwrite(os.path.join(outdir, f"{i:04d}.png"), img)
            render_imgs.append(img)
        imageio.mimwrite(os.path.join(outdir, 'video.mp4'), render_imgs, fps=10, quality=8)

def render_texture_map(renderer, objdir, textures, morph=False):
    tex, tex_depth = renderer.render_texture(objdir, textures)
    tex_img = tex.detach().cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]
    tex_img = to8b(tex_img)

    if morph:
        tex_depth = tex_depth.detach().cpu().numpy()[0]
        valid_mask = (tex_depth[:, :, None] < 2).astype(np.uint8)
        kernel = np.ones((3,3),np.uint8)
        valid_mask2 = cv2.dilate(valid_mask, kernel, iterations = 1)[:,:,None]
        tex_img2 = cv2.erode(tex_img, kernel, iterations = 1)

        tex_img = (valid_mask2 - valid_mask) * tex_img2 + valid_mask * tex_img + (1 - valid_mask2) * tex_img2
        # imageio.imwrite(os.path.join(outdir, f"mask_{i}.png"), valid_mask*255)
        # imageio.imwrite(os.path.join(outdir, f"mask2_{i}.png"), valid_mask2*255)
        # imageio.imwrite(os.path.join(debug_dir, f"smpl.png"), tex_img)
    return tex_img
    
def render_displacement_map(renderer, deformed_smpl_obj, smpl_obj):
    dis, _ = renderer.render_displacement(deformed_smpl_obj, smpl_obj)
    dis_img = dis.detach().cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]
    dis_img = to8b(dis_img)
    return dis_img

class TextureFitting:
    def __init__(self, smpl_uv_dir, tex_img_size=1024, render_img_size=512, lrate=1e-2, iter_num=200, 
                 debug=False, render=True, inpaint=False, logging=False):
        # prepare data
        self.debug = debug
        self.render = render
        self.iter_num = iter_num
        self.lrate = lrate
        self.tex_img_size = tex_img_size
        self.img_size = render_img_size
        self.is_inpaint = inpaint
        self.logging = logging

        self.smpl_uv_dir = smpl_uv_dir

        if self.is_inpaint:
            self.inpainter = Inpainter("external/LBAM_NoBN_ParisStreetView.pth")

    def inpaint(self, img):
        uv = load_obj_uv(self.smpl_uv_dir)
        inpaint_mask = np.zeros_like(img)
        imgsize = img.shape[0]
        uv = (uv * imgsize)

        sz = 4
        dims = [[x,y,z] for x in range(sz) for y in range(sz) for z in range(sz)]
        dims = np.array(dims[1:])
        dims = dims / np.sum(dims, axis=1, keepdims=True)
        for i, face in enumerate(uv):
            f = (dims @ face).astype(np.int32)
            m = (img[f[:,1], f[:,0], :] > np.ones(3) * 118) & (img[f[:,1], f[:,0], :] < np.ones(3) * 138)
            tri = np.sum(m, axis=-1) == 3
            if np.sum(np.array(tri).astype(np.int32)) > len(tri) / 6:
                triangle_cnt = np.array( face.astype(np.int32) )
                cv2.drawContours(inpaint_mask, [triangle_cnt], 0, (255,255,255), -1)

        network_out = self.inpainter(img, inpaint_mask)
        img = (network_out*255).astype(np.uint8)

        mask = (1- (img == 255).astype(np.uint8))
        img2 = cv2.erode(img, np.ones((7,7),np.uint8), iterations = 1)
        mask = cv2.erode(mask, np.ones((3,3),np.uint8), iterations = 1)
        mask_d = cv2.dilate(mask, np.ones((7,7),np.uint8), iterations = 1)
        mask2 = mask_d - mask
        out = mask * img + mask2 * img2 + (1 - mask_d) * img
        return out

    def __call__(self, output_dir, smpld_dir, scan_dir):
        # create smpl+d uv mesh
        os.makedirs(output_dir, exist_ok=True)
        smpld_uv_dir = os.path.join(output_dir, os.path.basename(smpld_dir))
        create_smpld_uv(smpld_uv_dir, smpld_dir, self.smpl_uv_dir, self.tex_img_size)

        # load vertices, faces and textures
        smpl_v, smpl_f, smpl_t = nr.load_obj(smpld_uv_dir, normalization=False, load_texture=True)
        scan_v, scan_f, scan_t = nr.load_obj(scan_dir, normalization=False, load_texture=True)
        smpl_v, smpl_f, smpl_t = smpl_v.unsqueeze(0), smpl_f.unsqueeze(0), smpl_t.unsqueeze(0)
        scan_v, scan_f, scan_t = scan_v.unsqueeze(0), scan_f.unsqueeze(0), scan_t.unsqueeze(0)

        # identify scene bound
        center = (torch.max(scan_v, 1)[0] + torch.min(scan_v, 1)[0]).cpu().numpy() / 2.0
        bound = (torch.max(scan_v, 1)[0] - torch.min(scan_v, 1)[0]).cpu().numpy()
        center, bound = center.squeeze(0), bound.squeeze(0)
        dist = bound[1] / 0.8

        # train params
        smpl_t.requires_grad = True
        opt_params = [smpl_t]
        optimizer = torch.optim.Adam(opt_params, lr=self.lrate)

        # # renderer
        round_view_iters = 5
        K = np.array([[self.img_size, 0.0, self.img_size//2], [0.0, self.img_size, self.img_size//2], [0.0, 0.0, 1.0]],
                        dtype=np.float32).reshape([1, 3, 3])
        renderer = nr.Renderer(image_size=self.img_size, fill_back=False, camera_mode='projection',
                                orig_size=self.img_size, light_intensity_ambient=1.0, light_intensity_directional=0,
                                near=0, far=2*dist, background_color=[1,1,1], K=K)

        round_poses = gen_cam_views(center, 18, dist, gl=True)
        training_imgs = []
        if self.debug:
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)

        for i in tqdm(range(self.iter_num)):
            # round view training frist
            if i < round_view_iters * len(round_poses):
                pose = round_poses[i%len(round_poses)]
            # random view 
            else: 
                pose = sphere2rot(dist, np.random.uniform(0, np.pi), np.random.uniform(0, np.pi*2), t=center)
                pose = np.linalg.inv(pose)
            t = pose[:3, 3].reshape([1, 1, 3])
            R = pose[:3, :3].reshape([1, 3, 3])
            R, t = torch.cuda.FloatTensor(R), torch.cuda.FloatTensor(t)

            scan_img = renderer.render_rgb(scan_v, scan_f, scan_t, R=R, t=t)
            smpl_img = renderer.render_rgb(smpl_v, smpl_f, smpl_t, R=R, t=t)
            
            # L1 loss between rendered images of deformed smpl and gt bodyscan
            loss = torch.sum(torch.abs(scan_img - smpl_img))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.logging:
                print(f"texture fitting iter {i}, loss {loss}")

            if self.debug:
                smpl_img = smpl_img.detach().cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]
                smpl_img = to8b(smpl_img[:self.img_size, :self.img_size, :])
                imageio.imwrite(os.path.join(debug_dir, f"{i}.png"), smpl_img)
                training_imgs.append(smpl_img)

        if self.debug:
            imageio.mimwrite(os.path.join(debug_dir, 'video.mp4'), training_imgs, fps=15, quality=8)

        # render optimized model and gt comparison
        if self.render:
            renderer_poses = gen_cam_views(center, 36, dist, gl=True)
            render_dir = os.path.join(output_dir, 'render')
            os.makedirs(render_dir, exist_ok=True)
            render_compare(renderer, [smpl_v, smpl_f, smpl_t], [scan_v, scan_f, scan_t], renderer_poses, render_dir, logging=self.logging)

        # generate smpl texture
        print("generating smpl texture...")
        tex_img = render_texture_map(renderer, self.smpl_uv_dir, smpl_t)
        if self.is_inpaint:
            tex_img = self.inpaint(tex_img)
        imageio.imwrite(os.path.join(output_dir, f"smpl.png"), tex_img)

        # # generate displacement map
        # if os.path.exists(smpld_dir):
        #     print("generating smpl displacment map...")
        #     dis_img = render_displacement_map(renderer, smpld_dir, self.smpl_uv_dir)
        #     imageio.imwrite(os.path.join(output_dir, f"smpl_dis.png"), dis_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str,
                        default="./data/bodyfitting/0127",
                        help='folder to input image')
    parser.add_argument('--smpl_uv_dir', type=str,
                        default="./data/smpl_uv",
                        help='folder to smpl uv')
    parser.add_argument('--lrate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument("--iter_num", type=int, default=200, 
                        help='total iteration number')
    parser.add_argument("--debug", action='store_true',
                        help='save debug info')
    parser.add_argument("--render", action='store_true',
                        help='render compare')

    args = parser.parse_args()
    texfitter = TextureFitting(**vars(args))
    texfitter.fit()