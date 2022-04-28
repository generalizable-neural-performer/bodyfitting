## This script is for SMPL fitting for GHR dataset


import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np 
import argparse
from fnmatch import fnmatch
from smplify.body_fitting import BodyFitting
from tqdm import tqdm
import torch
import csv, json
import cv2, imageio
from utils.io_utils import load_openpose, image_cropping


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str,
                        default="/data/ours_new",
                        help='target directory storing obj data')
    parser.add_argument("--annot_dir", type=str,
                        help='annot file contains camera parameter')
    parser.add_argument("--output_dir", type=str,
                        default="./logs",
                        help='directory contains output smpl and parameter')
    parser.add_argument("--openpose_dir", type=str,
                        default="../openpose",
                        help='directory of built openpose binary file')
    parser.add_argument("--info_dir", type=str,
                        help='csv file which contains gender information')
    parser.add_argument('--debug', default=True, action='store_true',
                        help='is output debug, false will speed up')
    parser.add_argument('--subject', type=str, default='zhuna',
                        help='target subject to fit smpl')
    parser.add_argument('--load_size', default=512, type=int,
                        help='load size of image data')
    parser.add_argument('--tasks', nargs='+', type=str,
                        default=['openpose', 'smplify', 'output'], 
                        help='tasks to perform')
    parser.add_argument('--use_mask', default=False, action='store_true',
                        help='smplify with human mask or not')
    parser.add_argument('--smpl_type', default="smpl", type=str,
                        help='use smpl or smplx')
    parser.add_argument('--age', default="adult", type=str,
                        help='use smpl or smil')
    parser.add_argument('--smplx_with_smpl_init', default=True, action='store_true',
                        help='if use smpl fitting result to initialize smplx fitting')
    parser.add_argument('--use_bodyscan', default=False, action='store_true',
                        help='use body scan as 3D supervision or not')
    parser.add_argument('--viewnum', type=int, default=8, 
                        help='use multiview data')
    parser.add_argument('--smpl_uv_dir', type=str, default="./data/smpl_uv",
                        help='folder to smpl uv')
    parser.add_argument('--white_bkgd', default=True, action='store_true', help='white bkgd')
    return parser


class runner():
    def __init__(self, args):
        self.options = args
        self.subject = args.subject
        self.target_dir = os.path.join(args.target_dir, self.subject)
        self.output_dir = os.path.join(args.output_dir, self.subject)
        self.openpose_dir = args.openpose_dir
        self.use_mask = args.use_mask
        self.white_bkgd = args.white_bkgd
        self.annot_dir = os.path.join(args.annot_dir, self.subject+'.npy') if args.annot_dir is not None \
                         else os.path.join(args.target_dir, 'annots.npy')
        self.smpl_type = args.smpl_type
        self.debug = args.debug
        self.use_hand_face = (self.smpl_type == 'smplx')
        self.load_size = args.load_size
        self.annots = np.load(self.annot_dir, allow_pickle=True).item()['cams']
        self.views = self.get_views()
        self.tasks = args.tasks
        if self.options.info_dir is not None and os.path.exists(self.options.info_dir):
            self.genders = {}
            with open(self.options.info_dir, 'r') as info:
                reader = csv.reader(info)
                for row in reader: self.gender[row[0]] = 'female' if int(row[1]) == 0 else 'male'
        self.gender = 'neutral' if not hasattr(self, 'genders') else self.genders[self.subject]
        self.bodyfitter = BodyFitting(self.options)
        
        self.seqs = self.get_sequence()
        self.mask_frames = [1,7,13,19,25,31,37,43] # frames with groud truth masks

    def get_views(self):
        # In GeneBody data, there exist some view missing in several sequences
        all_cameras_raw = list(range(48))
        if self.subject == 'Tichinah_jervier' or self.subject == 'dannier':
            all_cameras = list(set(all_cameras_raw) - set([32]))
        elif self.subject == 'wuwenyan':
            all_cameras = list(set(all_cameras_raw)-set([34, 36]))
        elif self.subject == 'joseph_matanda':
            all_cameras = list(set(all_cameras_raw) - set([39, 40, 42, 43, 44, 45, 46, 47]))
        else:
            all_cameras = all_cameras_raw
        
        return all_cameras

    def get_sequence(self):
        sequence_list = []
        sequence_list = os.listdir(os.path.join(self.target_dir, 'image', '00'))
        sequence_list = [int(os.path.splitext(dir_)[0]) for dir_ in sequence_list]
        # sequence_list = [seq for seq in sequence_list if seq in range(self.start, self.end)]
        sequence_list = sorted(sequence_list)
        return sequence_list

    def get_data(self, frame):
        img_dir = os.path.join(self.output_dir, '%06d' % frame, 'images')
        os.makedirs(img_dir, exist_ok=True)

        imgnames = sorted(os.listdir(os.path.join(self.target_dir, 'image', '00')))
        msknames = sorted(os.listdir(os.path.join(self.target_dir, 'mask', '00')))
        
        Ks, Rts, use_frames, mask_frames, images, masks = [], [], [], [], [], []
        for i, view in enumerate(self.views):
            img = imageio.imread(os.path.join(self.target_dir, 'image', '{:02d}'.format(view), imgnames[frame]))
            msk = imageio.imread(os.path.join(self.target_dir, 'mask', '{:02d}'.format(view), msknames[frame]))
            top, left, bottom, right = image_cropping(msk)
            img = img * (msk > 128)[...,None]
            img = cv2.resize(img[top:bottom, left:right].copy(), (self.load_size, self.load_size), cv2.INTER_CUBIC)
            if np.mean(img) > 10: # if not a black frame
                use_frames.append(view)
                imageio.imwrite(os.path.join(img_dir, '%02d.png' % view), img)
                images.append(img)
                if view in self.mask_frames and self.use_mask:
                    msk = cv2.resize(msk[top:bottom, left:right].copy(), (self.load_size, self.load_size), cv2.INTER_NEAREST)
                    masks.append(msk)
                    mask_frames.append(view)

                K, Rt = self.annots['K'][i].copy(), self.annots['RT'][i].copy()
                K[0,2] -= left
                K[1,2] -= top
                K[0,:] *= self.load_size / float(right - left)
                K[1,:] *= self.load_size / float(bottom - top)
                Ks.append(K.astype(np.float32))
                Rts.append(Rt.astype(np.float32))

        return images, masks, Ks, Rts, use_frames, mask_frames

    def run_openpose(self, frame, data):
        openpose_bin = "build/examples/openpose/openpose.bin"
        img_dir = os.path.abspath(os.path.join(self.output_dir, '%06d' % frame, 'images'))
        wrt_dir = os.path.abspath(os.path.join(self.output_dir, '%06d' % frame, 'openpose'))
        os.makedirs(wrt_dir, exist_ok=True)
        
        if len([dir_ for dir_ in os.listdir(wrt_dir) if '.json' in dir_]) < len(data[0]):
            torch.cuda.empty_cache()
            hand_face = "--hand --face" if self.use_hand_face else ""
            write_image = f"--write_images {wrt_dir}" if self.debug else ""
            os.system(f"cd {self.openpose_dir} && {openpose_bin} --image_dir {img_dir} \
                        --write_json {wrt_dir} {write_image} --display 0 {hand_face}")

    def read_openpose(self, frame):
        openpose_dir = os.path.join(self.output_dir, '%06d' % frame, 'openpose')
        views = sorted([dir_ for dir_ in os.listdir(openpose_dir) if '.json' in dir_])
        openpose_keypoints = []
        for view in views:
            openpose_keypoints.append(load_openpose(os.path.join(openpose_dir, view)))
        return openpose_keypoints

    def run_smplify(self, frame, data, keypoints):
        images, masks, Ks, Rts, use_frames, mask_frames = data
        keyframe = 25 if 25 in use_frames else use_frames[0]
        output_dir = os.path.join(self.output_dir, '%06d' % frame, 'smplify')
        self.bodyfitter(images, Rts, Ks, keypoints, gender=self.gender, keyframe=keyframe, use_frames=use_frames,
                        use_mask=self.use_mask, masks=masks, mask_frames=mask_frames, output_folder=output_dir)

    def run_output(self, frame):
        frame_dir = os.path.join(self.output_dir, '%06d' % frame)
        smpl_out_dir = os.path.join(frame_dir, 'debug', 'opt_smpl', 'smplx.obj')
        param_dir = os.path.join(frame_dir, 'debug', 'smpl_paramters.npy')
        smpl_folder = os.path.join(self.output_dir, 'smpl')
        param_folder = os.path.join(self.output_dir, 'param')
        os.makedirs(smpl_folder, exist_ok=True)
        os.makedirs(param_folder, exist_ok=True)
        os.system(f'cp {smpl_out_dir} {smpl_folder}'+'/%04d.obj'%(frame))
        os.system(f'cp {param_dir} {param_folder}'+'/%04d.npy'%(frame))

    def run(self):
        for frame in tqdm(self.seqs):
            data = self.get_data(frame)
            if 'openpose' in self.tasks:
                self.run_openpose(frame, data)
            keypoints = self.read_openpose(frame)
            if 'smplify' in self.tasks:
                self.run_smplify(frame, data, keypoints)
            if 'output' in self.tasks:
                self.run_output(frame)


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    myrunner = runner(args)
    myrunner.run()