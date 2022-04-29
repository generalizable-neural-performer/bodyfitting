import os
from matplotlib import dviread

import numpy as np
import cv2
import torch
import smplx

from models.smpl import SMPL
from models.utils import smpl_to_openpose, JointMapper
from .loss import *
# from utils.geometry import *
import config
from utils.io_utils import load_obj_mesh, compute_normal, compute_normal_torch
from utils.mesh_grid_searcher import MeshGridSearcher

from .prior import MaxMixturePrior

class SMPLify():
    """Implementation of multiview SMPLify."""
    def __init__(self,
                 smpl_type='smpl',
                 age='adult',
                 step_size=1e-2,
                 batch_size=1,
                 num_iters=600,
                 gender='male',
                 use_mask=False,
                 device=torch.device('cuda'), 
                 debug=True
                 ):

        # Store options
        self.device = device
        self.debug = debug
        self.gender = gender
        self.smpl_type = smpl_type
        self.use_hand_face = (smpl_type == 'smplx')
        self.age = age
        self.use_mask = use_mask
        self.batch_size = batch_size

        # Ignore the the following joints for the fitting process
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='data',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        if self.smpl_type == 'smpl':
            self.smpl = SMPL(config.SMPL_MODEL_DIR,
                            batch_size=batch_size,
                            kid_template_path=config.SMIL_MODEL_DIR,
                            gender=gender,
                            age=age,
                            create_transl=True).to(self.device)
        elif self.smpl_type == 'smplx':
            print('using smplx...')
            joint_mapper = JointMapper(smpl_to_openpose("smplx", use_hands=True,
                                    use_face=True,
                                    use_face_contour=True,
                                    openpose_format="coco25"))
            model_params = dict(model_path="data",
                                model_type='smplx',
                                joint_mapper=joint_mapper,
                                ext='npz',
                                gender=gender,
                                create_global_orient=True,
                                create_body_pose=True,
                                create_betas=True,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                create_expression=True,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=False,
                                use_face_contour=True,
                                dtype=torch.float32)
            self.smpl = smplx.create(**model_params).to(self.device)

        self.smpl_faces = self.smpl.faces.astype(np.int32).reshape(1, -1, 3)

    def __call__(self, net_output, c2ws, Ks, keypoints, output_folder, use_mask=False, 
                 masks=None, use_frames=[0], mask_frames=[0], keyframe=6, imsize=512,
                 use_mesh=False, meshfile=None, displacement=False):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        init_betas, init_poses = net_output
        device = init_poses.device

        # init skeleton poses with hmr estimated SMPL parameters
        body_pose = None
        if self.smpl_type == 'smpl':
            body_pose = init_poses[:, 3:].detach().clone()
        else:
            body_pose = init_poses[:, 3:63+3].detach().clone()
        if self.age == 'adult':
            betas = init_betas.detach().clone()
        elif self.age == 'kid':
            betas = torch.zeros([self.batch_size,11], dtype=torch.float32, device=device, requires_grad=True)

        # init hand face poses with zeros
        jaw_pose = torch.zeros([self.batch_size,1,3], dtype=torch.float32, device=device, requires_grad=True)
        leye_pose = torch.zeros([self.batch_size,1,3], dtype=torch.float32, device=device, requires_grad=True)
        reye_pose = torch.zeros([self.batch_size,1,3], dtype=torch.float32, device=device, requires_grad=True)
        left_hand_pose = torch.zeros([self.batch_size,6], dtype=torch.float32, device=device, requires_grad=True)
        right_hand_pose = torch.zeros([self.batch_size,6], dtype=torch.float32, device=device, requires_grad=True)

        # init global rotation with hmr estimation and body scale and global scale, translation with zero 
        # (because of depth ambiguity of monocular method)
        global_orient = init_poses[:, :3]
        global_transl = torch.zeros_like(global_orient, device=device)
        body_scale = torch.ones_like(global_transl[:, :1], device=device)

        # convert c2w to w2c 
        if isinstance(c2ws[0], np.ndarray):
            c2ws = torch.from_numpy(np.array(c2ws)).float().to(device)
        else:
            c2ws = torch.stack(c2ws, dim=0)
        w2cs = torch.inverse(c2ws)

        # load contours from masks if they are used
        if use_mask:
            masks = (np.array(masks) > 128).astype(np.float32)
            masks = torch.from_numpy(masks).to(device)
            mask_w2cs = [w2cs[use_frames.index(frame)] for frame in mask_frames]
            mask_Ks = [Ks[use_frames.index(frame)] for frame in mask_frames]
            # extract contours from mask, use mask contours for 2d closest point distance for mask loss
            contours = extract_countours(masks)
        
        if use_mesh:
            scan_verts, scan_faces = load_obj_mesh(meshfile)
            tris = scan_verts[scan_faces]
            face_norms = torch.from_numpy(np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])).float().to(device)
            scan_min, scan_max = scan_verts.min(0), scan_verts.max(0)
            scan_height = (scan_max - scan_min)[1]
            # set mesh for MeshGridSearcher
            pointsearcher = MeshGridSearcher(verts=scan_verts, faces=scan_faces)

            # constant scale is a scale prior knowledge of the scene, eg. for Renderpeople data the scale is relevant to scan's scale
            constant_scale = scan_height / 1.7
            
        else:
            # constant scale is a scale prior knowledge of the scene, eg. for GeneBody data the scene scale is 0.3
            constant_scale = 0.3
        # prepare for optimization
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        global_transl.requires_grad = True
        body_scale.requires_grad = True
        opt_params = [{"params": global_transl, "lr": 0.1},
                            {"params": body_scale, "lr": 0.1},
                            {"params": body_pose},{"params": betas},
                            {"params": global_orient},
                            {"params": leye_pose},{"params": reye_pose},
                            {"params": left_hand_pose},{"params": right_hand_pose},
                            ]
        optimizer = torch.optim.Adam(opt_params, lr=1e-2, betas=(0.9, 0.999))

        # optimization loop
        for i in range(self.num_iters):
            # smpl forward pass given input parameters
            smpl_output = self.smpl(global_orient=global_orient,
                            body_pose=body_pose,
                            betas=betas,
                            jaw_pose=jaw_pose,
                            leye_pose=leye_pose,
                            reye_pose=reye_pose,
                            left_hand_pose=left_hand_pose,
                            right_hand_pose=right_hand_pose,
                            return_full_pose=True)
            # estimated smpl vertices and 3d joints 
            model_joints = (smpl_output.joints + global_transl) * body_scale * constant_scale
            body_vertices = (smpl_output.vertices + global_transl) * body_scale * constant_scale

            # multi-view keypoint loss
            body_loss, loss_dict = multiview_keypoint_loss(w2cs, Ks, keypoints, model_joints, body_pose, betas, use_frames, 
                                                            self.pose_prior, imsize=imsize, use_hand_face=self.use_hand_face,
                                                            output_folder=output_folder, debug=False, verts=body_vertices)
            # multiview mask loss
            if use_mask and i > (self.num_iters // 3):
                mask_loss = multview_mask_loss(contours, masks, body_vertices, self.smpl_faces, mask_w2cs, mask_Ks, 
                                                mask_frames, imsize=imsize)
                # print("body_loss ", body_loss.item(), "mask_loss ", mask_loss.item())
            else:
                mask_loss = 0

            # point cloud loss 
            if use_mesh and i > (self.num_iters // 3):
                pc_loss = point_cloud_loss_mesh_grid(pointsearcher, body_vertices) / scan_height * imsize
            else:
                pc_loss = 0
            # total loss
            loss = body_loss + 5 * mask_loss + 5 * pc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # final output dict
        rtn_dict = {
            "vertices": self.cpu(body_vertices),
            "joints": self.cpu(model_joints),
            "pose": self.cpu(body_pose),
            "betas": self.cpu(betas),
            "global_orient": self.cpu(global_orient),
            "faces": self.smpl_faces[0],
            "global_transl": self.cpu(global_transl * body_scale),
            "scale": self.cpu(body_scale),
            "full_pose": self.cpu(smpl_output.full_pose),
        }

        if displacement and use_mesh:
            disp = torch.zeros_like(body_vertices)
            body_vertices = body_vertices.detach()
            disp.requires_grad = True
            opt_params = [disp]
            optimizer = torch.optim.Adam(opt_params, lr=5e-2, betas=(0.9, 0.999))
            smpl_faces = torch.from_numpy(self.smpl_faces).long().to(device)
            scan_verts = torch.from_numpy(scan_verts).float().to(device)
            for i in range(self.num_iters):
                deformed_verts = body_vertices+disp
                deformed_norms = compute_normal_torch(deformed_verts, smpl_faces)
                icp_loss = point_cloud_loss_mesh_grid(pointsearcher, deformed_verts)
                norm_loss = normal_loss_mesh_grid(pointsearcher, deformed_verts, face_norms, deformed_norms)
                smoothness = normal_laplacian_smoothness(deformed_norms, smpl_faces)
                loss = icp_loss + (norm_loss + smoothness) * constant_scale * 0.1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            rtn_dict.update({"displacement": self.cpu(disp)})


        return rtn_dict

    def cpu(self, tensor):
        cpu_array = tensor.detach().cpu().squeeze(0).numpy()
        return cpu_array