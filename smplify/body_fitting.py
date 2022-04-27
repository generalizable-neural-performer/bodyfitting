import argparse
import os, sys
sys.path.insert(1, os.getcwd())
import torch
import cv2, imageio
import numpy as np
from tqdm import tqdm

from models import hmr, SMPL
from utils.geometry import convert_hom_to_angle
from smplify.smplify import SMPLify
import config, constants
import trimesh
import torchvision.transforms as transforms
from utils.io_utils import save_obj_mesh

def HMR_forward(norm_img, args_checkpoint=config.HMR_CHECKPOINT):
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).cuda()
    if os.path.basename(args_checkpoint) == "model_checkpoint.pt":
        checkpoint = torch.load(args_checkpoint)['model']
        model.load_state_dict(checkpoint, strict=False)
    else:
        checkpoint = torch.load(args_checkpoint)['model']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
    model.eval()

    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.cuda())

    return pred_rotmat, pred_betas, pred_camera


def check_smpl_fitting(image, verts, c2w, K):
    w2c = np.linalg.inv(c2w)
    rvec, tvec = cv2.Rodrigues(w2c[:3,:3])[0], w2c[:3,3]
    vproj, _ = cv2.projectPoints(verts.astype(np.float32), rvec, tvec, K, np.zeros(5))
    for p in vproj.reshape(-1,2):
        if  p[0] >= 0 and p[0] < image.shape[1] \
        and p[1] >= 0 and p[1] < image.shape[0]:
            image = cv2.circle(image.copy(),(int(p[0]),int(p[1])),1,(0,255,0),-1)
    return image

class BodyFitting(object):
    def __init__(self, options):
        self.options = options
        self.debug = options.debug
        self.loadsize = options.load_size
        self.use_mask = options.use_mask
        self.hmr_checkpoint = "data/model_checkpoint.pt"
        self.smpl_type = options.smpl_type
        self.use_hand_face = (self.smpl_type == 'smplx')
        # self.smplify = SMPLify(smpl_type=self.smpl_type, age=self.options.age,
        #                         gender=gender, use_mask=self.use_mask, debug=False)

    def run_hmr(self, image, c2w, input_res=224):
        normalize = transforms.Normalize(constants.IMG_NORM_MEAN, constants.IMG_NORM_STD)

        # run hmr to get inital smpl estimation from keyframe image
        h, w = image.shape[0], image.shape[1]
        image = cv2.resize(image, (input_res, input_res), cv2.INTER_CUBIC)
        image = torch.from_numpy(image).float().cuda() / 255.
        image = image.permute((2,0,1))
        
        image = normalize(image)[None]
        pred_rotmat, pred_betas, pred_camera = HMR_forward(image)

        # transfer smpl to world coordinate via calibrated camera pose
        smpl_rot = pred_rotmat.squeeze(0)[0]
        c2w = torch.from_numpy(c2w).to(image.device)
        pred_rotmat[0, 0] = c2w[:3,:3] @ smpl_rot
        pred_poses = convert_hom_to_angle(pred_rotmat, 1, image.device) # from rotmat to rotvec

        return pred_betas, pred_poses


    def __call__(self, images, c2ws, Ks, keypoints, gender='male', keyframe=25, use_frames=list(range(48)),
                 use_mask=False, masks=None, mask_frames=None, render_skip=12, output_folder=None,
                 use_mesh=False, meshfile=None, disp=False):

        smplify = SMPLify(smpl_type=self.smpl_type, age=self.options.age,
                          gender=gender, use_mask=self.use_mask, debug=False)

        # predict smpl pose and shape via hmr, leaving global scale and translation for multiview smplify
        net_output = self.run_hmr(images[keyframe], c2ws[keyframe])

        # run mutliview smplify
        result = smplify(net_output, c2ws, Ks, keypoints, output_folder, use_mask=use_mask, masks=masks, 
                         use_frames=use_frames, mask_frames=mask_frames, keyframe=keyframe, imsize=images[0].shape[0],
                         use_mesh=use_mesh, meshfile=meshfile, displacement=disp)

        # output result if output_folder is given
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            np.save(os.path.join(output_folder, f'{self.smpl_type}_parameter.npy'), result)
            save_obj_mesh(os.path.join(output_folder, f'{self.smpl_type}.obj'), result["vertices"], result["faces"])
            if disp:
                save_obj_mesh(os.path.join(output_folder, f'{self.smpl_type}+d.obj'), result["vertices"]+result["displacement"], result["faces"])
            # debug info check smpl fitting
            if self.debug:
                fitting_path = os.path.join(output_folder, "smpl_fitting")
                os.makedirs(fitting_path, exist_ok=True)
                for frame in use_frames[::render_skip]:
                    idx = use_frames.index(frame)
                    outimg = check_smpl_fitting(images[idx], result["vertices"], c2ws[idx], Ks[idx])
                    imageio.imwrite(os.path.join(fitting_path, '%02d.png'%frame), outimg)
