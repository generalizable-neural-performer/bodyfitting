from __future__ import division
import math

import torch
import torch.nn as nn
import numpy

import neural_renderer as nr
import numpy as np

class Renderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera_mode='projection',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0,0,1],
                 near=0.1, far=100,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0]):
        super(Renderer, self).__init__()
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode in ['projection', 'orthogonal']:
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.cuda.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.cuda.FloatTensor(self.t)
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')


        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction 

        # rasterization
        self.rasterizer_eps = 1e-3

    def forward(self, vertices, faces, textures=None, mode=None, K=None, R=None, t=None, dist_coeffs=None, orig_size=None, lightoff=False):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        
        if mode is None:
            return self.render(vertices, faces, textures, K, R, t, dist_coeffs, orig_size, lightoff=lightoff)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures, K, R, t, dist_coeffs, orig_size, lightoff=lightoff)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces, K, R, t, dist_coeffs, orig_size, lightoff=lightoff)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, K, R, t, dist_coeffs, orig_size, lightoff=lightoff)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'orthogonal':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.orthogonal(vertices, K, R, t, orig_size)
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'orthogonal':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.orthogonal(vertices, K, R, t, orig_size)
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None, lightoff=False):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        if not lightoff:
            # lighting
            faces_lighting = nr.vertices_to_faces(vertices, faces)
            textures = nr.lighting(
                faces_lighting,
                textures,
                self.light_intensity_ambient,
                self.light_intensity_directional,
                self.light_color_ambient,
                self.light_color_directional,
                self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'orthogonal':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.orthogonal(vertices, K, R, t, orig_size)
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images

    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None, lightoff=False):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        if not lightoff:
            faces_lighting = nr.vertices_to_faces(vertices, faces)
            textures = nr.lighting(
                faces_lighting,
                textures,
                self.light_intensity_ambient,
                self.light_intensity_directional,
                self.light_color_ambient,
                self.light_color_directional,
                self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'orthogonal':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.orthogonal(vertices, K, R, t, orig_size)
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        out = nr.rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth'], out['alpha']

    def render_texture(self, filename_obj, textures):
        # load vertices
        vertices = []
        with open(filename_obj) as f:
            lines = f.readlines()
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'vt':
                vertices.append([float(v)*2-1.0 for v in line.split()[1:3]] + [1.0]) # mapping uv from [0,1] to [-1,1]
        vertices = np.vstack(vertices).astype(np.float32)

        # load faces for textures
        faces = []
        material_names = []
        material_name = ''
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                if '/' in vs[0] and '//' not in vs[0]:
                    v0 = int(vs[0].split('/')[1])
                else:
                    v0 = 0
                
                for i in range(nv - 2):
                    if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                        v1 = int(vs[i + 1].split('/')[1])
                    else:
                        v1 = 0
                    if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                        v2 = int(vs[i + 2].split('/')[1])
                    else:
                        v2 = 0
                    faces.append((v0, v1, v2))
                    material_names.append(material_name)
            if line.split()[0] == 'usemtl':
                material_name = line.split()[1]
        faces = np.vstack(faces).astype(np.int32) - 1
        faces = vertices[faces]
        faces = torch.from_numpy(faces).cuda()
        faces = faces.unsqueeze(0)
        ## Fill back
        faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
        textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # print(faces.shape, textures.shape)
        out = nr.rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth']


    def render_displacement(self, deformed_smpl_obj, smpl_obj):
        # load vertices of deformed smpl
        vertices, uv = [], []
        with open(deformed_smpl_obj) as f:
            lines = f.readlines()
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'vt':
                uv.append([float(v)*2-1.0 for v in line.split()[1:3]] + [1.0]) # mapping uv from [0,1] to [-1,1]
            if line.split()[0] == 'v':
                vertices.append([float(v) for v in line.split()[1:4]])
        uv = np.vstack(uv).astype(np.float32)
        vertices = np.vstack(vertices).astype(np.float32)

        # load faces for textures of deformed smpl
        faces = []
        material_names = []
        material_name = ''
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                if '/' in vs[0] and '//' not in vs[0]:
                    v0 = int(vs[0].split('/')[1])
                else:
                    v0 = 0
                
                for i in range(nv - 2):
                    if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                        v1 = int(vs[i + 1].split('/')[1])
                    else:
                        v1 = 0
                    if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                        v2 = int(vs[i + 2].split('/')[1])
                    else:
                        v2 = 0
                    faces.append((v0, v1, v2))
                    material_names.append(material_name)
            if line.split()[0] == 'usemtl':
                material_name = line.split()[1]
        faces = np.vstack(faces).astype(np.int32) - 1
        faces = uv[faces]
        faces = torch.from_numpy(faces).cuda()
        faces = faces.unsqueeze(0)

        # load vertices of smpl
        smpl_vertices = []
        with open(smpl_obj) as f:
            lines = f.readlines()

        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'v':
                smpl_vertices.append([float(v) for v in line.split()[1:4]])
        smpl_vertices = np.vstack(smpl_vertices).astype(np.float32)

        # load faces of smpl
        smpl_faces = []
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                v0 = int(vs[0].split('/')[0])
                for i in range(nv - 2):
                    v1 = int(vs[i + 1].split('/')[0])
                    v2 = int(vs[i + 2].split('/')[0])
                    smpl_faces.append((v0, v1, v2))
        smpl_faces = np.vstack(smpl_faces).astype(np.int32) - 1
        displacement = vertices - smpl_vertices
        textures = displacement[smpl_faces]

        sz = 4
        new_tex = []
        dims = np.array([[x,y,z] for x in range(sz) for y in range(sz) for z in range(sz)])
        dims = dims / (np.sum(dims, axis=1, keepdims=True) + 1e-9)
        for tex in textures:
            tex_list = []
            for dim in dims:
                tex_list.append(np.sum(tex * dim[:, None], axis=-2))
            tex_array = np.array(tex_list).reshape([sz,sz,sz,3])
            new_tex.append(tex_array)
        textures = np.array(new_tex).astype(np.float32)
        textures = torch.from_numpy(textures).cuda()
        textures = textures.unsqueeze(0)

        std = torch.var(textures.reshape(-1, textures.shape[-1]), dim=0) ** 0.5
        textures = textures / torch.max(std) / 2 + .5

        ## Fill back
        faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
        textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        out = nr.rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth']