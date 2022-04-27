## This script is for SMPL fitting for RenderPeople dataset


from asyncio import tasks
from distutils.log import debug
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np 
import argparse
from smplify.body_fitting import BodyFitting
from smplify.texture_fitting import TextureFitting
from tqdm import tqdm
import torch
import csv, json
import cv2, imageio
from utils.io_utils import load_openpose, image_cropping
from utils.renderer import render_texture_mesh
from utils.cam_pose_vis import cam_pose_vis
# from camera_transform import transform, load_cameras


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str,
                        default="/data/ours_new",
                        help='target directory storing obj data')
    parser.add_argument("--output_dir", type=str,
                        default="./logs",
                        help='target directory storing obj data')
    parser.add_argument("--openpose_dir", type=str,
                        default="../openpose",
                        help='target directory storing obj data')
    parser.add_argument("--info_dir", type=str,
                        help='target directory storing obj data')
    parser.add_argument('--debug', default=True, action='store_true',
                        help='is output debug, false will speed up')
    parser.add_argument('--load_size', default=512, type=int,
                        help='load size of image data')
    parser.add_argument('--tasks', nargs='+', type=str,
                        default=['openpose', 'smplify', 'smpld', 'texfit', 'output'], 
                        help='tasks to perform')
    parser.add_argument('--use_mask', default=False, action='store_true',
                        help='smplify with human mask or not')
    parser.add_argument('--smpl_type', default="smpl", type=str,
                        help='use smpl or smplx')
    parser.add_argument('--age', default="adult", type=str,
                        help='use smpl or smil')
    parser.add_argument('--smplx_with_smpl_init', default=True, action='store_true',
                        help='if use smpl fitting result to initialize smplx fitting')
    parser.add_argument('--viewnum', type=int, default=8, 
                        help='use multiview data')
    parser.add_argument('--smpl_uv_dir', type=str, default="./data/smpl_uv",
                        help='folder to smpl uv')
    parser.add_argument('--white_bkgd', default=True, action='store_true', help='white bkgd')
    return parser


class runner():
    def __init__(self, args):
        self.options = args
        self.openpose_dir = args.openpose_dir
        self.use_mask = args.use_mask
        self.white_bkgd = args.white_bkgd
        self.smpl_type = args.smpl_type
        self.debug = args.debug
        self.viewnum = args.viewnum
        self.output_dir = args.output_dir
        self.use_hand_face = (self.smpl_type == 'smplx')
        self.load_size = args.load_size
        self.tasks = args.tasks
        self.subjects, self.meshfiles = self.get_subjects()
        if self.options.info_dir is not None and os.path.exists(self.options.info_dir):
            self.genders = []
            with open(self.options.info_dir, 'r') as info:
                reader = csv.reader(info)
                for row in reader: self.genders.append('female') if int(row[1]) == 0 else self.genders.append('male')
        else:
            self.genders = ['neutral' for _ in range(len(self.subjects))]
        self.bodyfitter = BodyFitting(self.options)
        self.disp = 'smpld' in self.tasks
        if 'texfit' in self.tasks:
            self.texturefitter = TextureFitting(args.smpl_uv_dir, render=True, debug=debug)


    def get_subjects(self):
        subjects, meshes = [], []
        for path, subdirs, files in os.walk(self.options.target_dir):
            for name in files:
                if '.obj' in name:
                    if name[-8:] != "_30k.obj":
                        meshfile = os.path.join(path, name)
                        subject = meshfile.split('/')[-2]
                        subjects.append(subject)
                        meshes.append(meshfile)
        return subjects, meshes
                        
    def render_data(self, subject, meshfile):
        imgdir = os.path.join(self.output_dir, subject, 'images')
        maskdir = os.path.join(self.output_dir, subject, 'masks')
        os.makedirs(imgdir, exist_ok=True)
        if self.use_mask:
            os.makedirs(maskdir, exist_ok=True)

        # load from path directly
        if os.path.exists(os.path.join(imgdir, '%02d.png'%0)):
            images, masks = [], []
            for i in range(self.viewnum):
                images.append(imageio.imread(os.path.join(imgdir, '%02d.png'%i)))
                if self.use_mask:
                    masks.append(imageio.imread(os.path.join(maskdir, '%02d.png'%i)))
            glRts, Ks = render_texture_mesh(meshfile, self.load_size, pose_only=True)
        # render use neural renderer
        else:
            images, masks, glRts, Ks = render_texture_mesh(meshfile, self.load_size, white_bkgd=self.white_bkgd)
            for i in range(self.viewnum):
                imageio.imwrite(os.path.join(imgdir, '%02d.png'%i), images[i])
                if self.use_mask:
                    imageio.imwrite(os.path.join(maskdir, '%02d.png'%i), masks[i])
        Rts = [np.linalg.inv(Rt).astype(np.float32) for Rt in glRts] # cv w2c
        Ks = [K.astype(np.float32) for K in Ks]
        use_frames = list(range(self.viewnum))
        mask_frames = list(range(self.viewnum))

        return images, masks, Ks, Rts, use_frames, mask_frames

    def run_openpose(self, subject, data):
        openpose_bin = "build/examples/openpose/openpose.bin"
        img_dir = os.path.abspath(os.path.join(self.output_dir, subject, 'images'))
        wrt_dir = os.path.abspath(os.path.join(self.output_dir, subject, 'openpose'))
        os.makedirs(wrt_dir, exist_ok=True)
        
        if len([dir_ for dir_ in os.listdir(wrt_dir) if '.json' in dir_]) < len(data[0]):
            torch.cuda.empty_cache()
            hand_face = "--hand --face" if self.use_hand_face else ""
            write_image = f"--write_images {wrt_dir}" if self.debug else ""
            os.system(f"cd {self.openpose_dir} && {openpose_bin} --image_dir {img_dir} \
                        --write_json {wrt_dir} {write_image} --display 0 {hand_face}")

    def read_openpose(self, subject):
        openpose_dir = os.path.join(self.output_dir, subject, 'openpose')
        views = sorted([dir_ for dir_ in os.listdir(openpose_dir) if '.json' in dir_])
        openpose_keypoints = []
        for view in views:
            openpose_keypoints.append(load_openpose(os.path.join(openpose_dir, view)))
        return openpose_keypoints

    def run_smplify(self, subject, data, keypoints, gender, meshfile):
        images, masks, Ks, Rts, use_frames, mask_frames = data
        keyframe = use_frames[0]
        output_dir = os.path.join(self.output_dir, subject, 'smplify')
        self.bodyfitter(images, Rts, Ks, keypoints, gender=gender, keyframe=keyframe, use_frames=use_frames,
                        use_mask=self.use_mask, masks=masks, mask_frames=mask_frames, output_folder=output_dir,
                        use_mesh=True, meshfile=meshfile, disp=self.disp)

    def run_texfit(self, subject, meshfile):
        output_dir = os.path.join(self.output_dir, subject, 'texfit')
        smpld_dir = os.path.join(self.output_dir, subject, 'smplify', f'{self.smpl_type}+d.obj')
        if os.path.exists(smpld_dir):
            self.texturefitter(output_dir, smpld_dir, meshfile)

    def run_output(self, subject):
        frame_dir = os.path.join(self.output_dir, subject)
        smpl_out_dir = os.path.join(frame_dir, 'debug', 'opt_smpl', f'{self.smpl_type}.obj')
        param_dir = os.path.join(frame_dir, 'debug', f'{self.smpl_type}_paramters.npy')
        smpl_folder = os.path.join(self.output_dir, 'SMPL')
        os.makedirs(smpl_folder, exist_ok=True)
        os.system(f'cp {smpl_out_dir} {smpl_folder}')
        os.system(f'cp {param_dir} {smpl_folder}')

    def run(self):
        # for frame in tqdm(self.seqs):
        for subject, meshfile, gender in tqdm(zip(self.subjects, self.meshfiles, self.genders)):
            data = self.render_data(subject, meshfile)
            if 'openpose' in self.tasks:
                self.run_openpose(subject, data)
            keypoints = self.read_openpose(subject)
            if 'smplify' in self.tasks:
                self.run_smplify(subject, data, keypoints, gender, meshfile)
            if 'texfit' in self.tasks:
                self.run_texfit(subject, meshfile)
            if 'output' in self.tasks:
                self.run_output(subject)



if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    myrunner = runner(args)
    myrunner.run()