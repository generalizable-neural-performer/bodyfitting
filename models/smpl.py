import torch
import numpy as np
import smplx
from smplx import SMPL as _SMPL
# from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints
from dataclasses import dataclass
from typing import NewType, Union

import config
import constants

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


@dataclass
class ModelOutput:
    vertices: Tensor = None
    joints: Tensor = None
    full_pose: Tensor = None
    betas: Tensor = None
    expression: Tensor = None
    global_orient: Tensor = None
    body_pose: Tensor = None
    left_hand_pose: Tensor = None
    right_hand_pose: Tensor = None
    jaw_pose: Tensor = None
    joints_ori: Tensor = None

    def __getitem__(self, key):
        data_dict = asdict(self)
        return data_dict[key]

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        J_regressor_h36m = np.load(config.JOINT_REGRESSOR_H36M)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        self.joints = None

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        self.joints = smpl_output.joints
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             joints_ori=self.joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

    def get_joints_h36m(self, vertices):
        joints = vertices2joints(self.J_regressor_h36m, vertices)
        return joints

    def get_joints_ori(self):
        return self.joints
