"""
Functions for camera computation
"""
import os
import numpy as np
import cv2
import math

"""
Some camera parameters [fx, fy, cx, cy, k1, k2, p1, p2, k3, w, h]
                         0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10
                       [rvec, rvec, rvec, tvec, tvec, tvec]  
"""
CAMCONF_OPPOv1_TOF = [219.748291, 219.708481, 120.041603, 90.454445, 0.0, 0.0, 0.0, 0.0, 0.0, 240, 180]
CAMCONF_OPPOv1_RGB = [3127.59, 3127.03, 2013.29, 1498.57, 0.0, 0.0, 0.0, 0.0, 0.0, 4032, 3024]
CAMCONF_OPPOv1_RT = [0.003402, -0.002128, 0.0047195, 14.502988, 0.207659, 2.017257]

CAMCONF_OPPOv2_TOF = [219.584412, 219.65126, 120.479614, 90.125099, 0.169974, -0.574944, -0.001121, -0.001121, 0.431544,
                      240, 180]
CAMCONF_OPPOv2_RGB = [3116.318604, 3116.331787, 2023.120117, 1564.619629, 0.281918, -1.460955, 0.000182, 0.000182,
                      2.364736,
                      4032, 3024]
CAMCONF_OPPOv2_RT = [0.003215, -0.0041135, -0.0045655, 13.96171, -0.051691, 2.340252]

CAMCONF_H36M_RGB = [1149.68, 1147.59, 508.85, 508.06, 0, 0, 0, 0, 0, 1000, 1000]
CAMCONF_H36M_TOF = [1149.68, 1147.59, 508.85, 508.06, 0, 0, 0, 0, 0, 1000, 1000]
CAMCONF_H36M_RT = [0.003215, 0.003215, -0.0045655, 0.003215, -0.051691, 0.003215]
###############  OPPOv1  ########################
OPPOv1 = dict()
OPPOv1["dual"] = True
OPPOv1["height"] = 3024
OPPOv1["width"] = 4032
OPPOv1["rgb"] = np.array([[3127.59, 0.00000, 2013.29],
                          [0.00000, 3127.03, 1498.57],
                          [0.00000, 0.00000, 1.00000]])
OPPOv1["dist_rgb"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
OPPOv1["tof_height"] = 180
OPPOv1["tof_width"] = 240
OPPOv1["tof"] = np.array([[219.748291, 0.000000, 120.041603],
                          [0.000000, 219.708481, 90.454445],
                          [0.000000, 0.000000, 1.000000]])
OPPOv1["dist_tof"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
OPPOv1["R"] = np.array([[0.999987, -0.004723, -0.002120],
                        [0.004716, 0.999983, -0.003407],
                        [0.002136, 0.003397, 0.999992]])
OPPOv1["T"] = np.array([14.502988, 0.207659, 2.017257]).reshape(-1, 1)
##################################################
OPPO = dict()
OPPO["dual"] = True
OPPO["height"] = 3024
OPPO["width"] = 4032
OPPO["rgb"] = np.array([[3127.59, 0.00000, 2013.29],
                        [0.00000, 3127.03, 1498.57],
                        [0.00000, 0.00000, 1.00000]])
OPPO["dist_rgb"] = np.array([0.26464, -1.42012, -0.000822745, -0.000822745, 2.32516])
OPPO["tof_height"] = 180
OPPO["tof_width"] = 240
OPPO["tof"] = np.array([[221.929, 0.000000, 119.298],
                        [0.000000, 221.875, 89.7724],
                        [0.000000, 0.000000, 1.000000]])
OPPO["dist_tof"] = np.array([0.135271, 0.389894, -0.000371168, -0.000371168, 0.124161])
OPPO["R"] = np.array([[0.999997, -0.00246939, 0.000554154],
                      [0.00246996, 0.999996, -0.00102186],
                      [-0.000551629, 0.00102323, 0.999999]])
OPPO["T"] = np.array([14.8343, -0.406686, -1.0636]).reshape(-1, 1)
###############  OPPOv2  ##########################
OPPOv2 = dict()
OPPOv2["dual"] = True
OPPOv2["height"] = 3024
OPPOv2["width"] = 4032
OPPOv2["rgb"] = np.array([[3116.318604, 0.00000, 2023.120117],
                          [0.00000, 3116.331787, 1564.619629],
                          [0.00000, 0.00000, 1.00000]])
OPPOv2["dist_rgb"] = np.array([0.281918, -1.460955, 0.000182, 0.000182, 2.364736])
OPPOv2["tof_height"] = 180
OPPOv2["tof_width"] = 240
OPPOv2["tof"] = np.array([[219.584412, 0.000000, 120.479614],
                          [0.000000, 219.65126, 90.125099],
                          [0.000000, 0.000000, 1.000000]])
OPPOv2["dist_tof"] = np.array([0.169974, -0.574944, -0.001121, -0.001121, 0.431544])
OPPOv2["R"] = np.array([[0.999981, 0.004559, -0.004121],
                        [-0.004572, 0.999984, -0.003206],
                        [0.004106, 0.003224, 0.999986]])
OPPOv2["T"] = np.array([13.96171, -0.051691, 2.340252]).reshape(-1, 1)
###############  Kinect  ##########################
H36M = dict()
H36M["dual"] = True
H36M["height"] = 1000
H36M["width"] = 1000
H36M["rgb"] = np.array([[1149.68, 0.00000, 508.85],
                        [0, 00000, 1147.59, 508.06],
                        [0.00000, 0.00000, 1]])
H36M["dist_rgb"] = np.array([0, 0, 0, 0, 0])
H36M["tof_height"] = 1000
H36M["tof_width"] = 1000
H36M["tof"] = np.array([[1149.68, 0.00000, 508.85],
                        [0, 00000, 1147.59, 508.06],
                        [0.00000, 0.00000, 1]])
H36M["dist_tof"] = np.array([0, 0, 0, 0, 0])
H36M["R"] = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
H36M["T"] = np.array([0, 0, 0]).reshape(-1, 1)
CAMERA = dict()
CAMERA["OPPO"] = OPPO
CAMERA["OPPOv1"] = OPPOv1
CAMERA["OPPOv2"] = OPPOv2
CAMERA["H36M"] = H36M


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 2e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class Camera(object):
    def __init__(self, args):
        self.loadFlatArgs(args)
        self.updateArgs()

    def updateArgs(self):
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def resize(self, scale):
        self.K = self.K * scale
        self.K[2, 2] = 1.0
        self.height = self.height * scale
        self.width = self.width * scale
        self.updateArgs()

    def rotate(self, orient):
        """
        Rotate the camera intrinsic parameters
        :param orient: 0,1,2,3 => clockwise 0,90,180,270
        """
        if orient == 0:
            return
        elif orient == 1:
            self.K[0, 0] = self.fy
            self.K[1, 1] = self.fx
            self.K[0, 2] = self.height - self.cy
            self.K[1, 2] = self.cx
            self.dist[[2, 3]] = self.dist[[3, 2]]
            tmp = self.height
            self.height = self.width
            self.width = tmp
        elif orient == 2:
            self.K[0, 2] = self.width - self.cx
            self.K[1, 2] = self.height - self.cy
        elif orient == 3:
            self.K[0, 0] = self.fy
            self.K[1, 1] = self.fx
            self.K[0, 2] = self.cy
            self.K[1, 2] = self.width - self.cx
            self.dist[[2, 3]] = self.dist[[3, 2]]
            tmp = self.height
            self.height = self.width
            self.width = tmp
        self.updateArgs()

    def loadFlatArgs(self, args):
        """
        :param args: [fx, fy, cx, cy, k1, k2, p1, p2, k3, w, h]
                       0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10
        """
        self.K = np.zeros((3, 3), dtype=np.float64)
        self.K[0, 0] = args[0]
        self.K[1, 1] = args[1]
        self.K[0, 2] = args[2]
        self.K[1, 2] = args[3]
        self.K[2, 2] = 1
        self.dist = np.array(args[4:9], dtype=np.float64)
        self.width = args[9]
        self.height = args[10]
        self.updateArgs()

    def undistort(self, points):
        """
        Undistort the points, the points are focalized
        :param points: k x 2
        :return: k x 2
        """
        k1, k2, p1, p2, k3 = self.dist
        xx_ = points[:, 0] ** 2
        yy_ = points[:, 1] ** 2
        rr_ = xx_ + yy_
        xy2_ = 2 * points[:, 0] * points[:, 1]
        t = 1 / (1 + ((k3 * rr_ + k2) * rr_ + k1) * rr_)
        dx = p1 * xy2_ + p2 * (rr_ + 2 * xx_)
        dy = p1 * (rr_ + 2 * yy_) + p2 * xy2_
        x = (points[:, 0] - dx) * t
        y = (points[:, 1] - dy) * t
        return np.stack((x, y), axis=1)

    def distort(self, points):
        """
        distort the points, the points are focalized
        :param points: k x 2
        :return: k x 2
        """
        k1, k2, p1, p2, k3 = self.dist
        x_ = points[:, 0]
        y_ = points[:, 1]
        r = np.sqrt(x_ ** 2 + y_ ** 2)
        x__ = x_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + 2 * p1 * x_ * y_ + p2 * (r ** 2 + 2 * x_ ** 2)
        y__ = y_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + p1 * (r ** 2 + 2 * y_ ** 2) + 2 * p2 * x_ * y_
        return np.stack((x__, y__), axis=1)

    def reproject(self, points, depth):
        """
        reproject the points into camera coordinate
        :param points: k x 2
        :param depth: k dim
        :return: k x 3 points
        """
        assert points.shape[0] == depth.shape[0]
        assert len(points.shape) == 2
        cxy = [self.cx, self.cy]
        fxy = [self.fx, self.fy]
        points = (points - cxy) / fxy
        points = self.undistort(points)
        depth = depth.reshape(-1, 1)
        xy = points * depth
        points3d = np.concatenate((xy, depth), axis=1)
        return points3d

    def solve_coorz(self, points, depth):
        """
        solve the coordinate z and reproject points into camera coordinate
        :param points: k x 2
        :param depth: k dim
        :return: k x 3 points
        """
        assert points.shape[0] == depth.shape[0]
        assert len(points.shape) == 2
        cxy = [self.cx, self.cy]
        fxy = [self.fx, self.fy]
        points = (points - cxy) / fxy
        points = self.undistort(points)
        points = points * fxy + cxy
        depth = depth.reshape(-1, 1)
        depth = np.array(depth, dtype=np.float)

        z = []
        for i in range(points.shape[0]):
            # print(points[i][0], points[i][1])
            # print((((points[i][0] - self.cx) / self.fx) ** 2 + ((points[i][1] - self.cy) / self.fy) ** 2 + 1))
            item_z = depth[i] ** 2 / (((points[i][0] - self.cx) / self.fx) ** 2 + ((points[i][1] - self.cy) / self.fy) ** 2 + 1)
            item_z = np.sqrt(item_z)
            # print(item_z, " ", depth[i])
            z.append(item_z)
        z = np.array(z).reshape(-1, 1)
        points = (points - cxy) / fxy
        xy = points * z
        points3d = np.concatenate((xy, z), axis=1)
        return points3d


class DualCamera(object):
    def __init__(self, cam1, cam2, rvec, tvec):
        """
        :param cam1:
        :param cam2:
        :param R: dim 3 vector
        :param T: dim 3 vector
        """
        self.RGB = cam1
        self.TOF = cam2
        self.rvec_RGB2TOF = np.array(rvec)
        self.tvec_RGB2TOF = np.array(tvec)
        self.R = cv2.Rodrigues(self.rvec_RGB2TOF)[0]
        self.T = self.tvec_RGB2TOF.reshape(3, 1)

    def tof2rgb(self, points):
        """
        :param points_tof: k x 3 numpy
        :param CAM:
        :return: k x 3 numpy
        """
        assert points.shape[1] == 3
        uv_depth = points[:, :2]
        cxy = [self.TOF.cx, self.TOF.cy]
        fxy = [self.TOF.fx, self.TOF.fy]
        uv_depth = (uv_depth - cxy) / fxy
        uv_depth = undistort_points(uv_depth, self.TOF.dist)
        s3d_f = np.concatenate((uv_depth, np.ones((uv_depth.shape[0], 1), dtype=uv_depth.dtype)), axis=1)
        s3d_tof = s3d_f * points[:, 2, None]
        # s3d_rgb = (s3d_tof + self.tvec_RGB2TOF)
        # print("rot matrix")
        # angles = np.array([0 / 180 * np.pi, 0 / 180 * np.pi, 0 / 180 * np.pi], dtype=np.float32)
        # rot_matrix = cv2.Rodrigues(angles)[0]
        # print(rot_matrix)
        s3d_rgb = np.dot((s3d_tof + self.tvec_RGB2TOF), self.R)

        cxy = [self.RGB.cx, self.RGB.cy]
        fxy = [self.RGB.fx, self.RGB.fy]
        # print("rgb cxy ")
        # print(cxy)
        # print(fxy)
        uv_rgb = s3d_rgb[:, :2] / s3d_rgb[:, 2, None]
        uv_rgb = distort_points(uv_rgb, self.RGB.dist)
        uv_rgb = uv_rgb * fxy + cxy
        return uv_rgb

    def tof2undistort(self, points):
        """
        :param points_tof: k x 3 numpy
        :param CAM:
        :return: k x 3 numpy
        """
        assert points.shape[1] == 3
        uv_depth = points[:, :2]
        cxy = [self.TOF.cx, self.TOF.cy]
        fxy = [self.TOF.fx, self.TOF.fy]
        uv_depth = (uv_depth - cxy) / fxy
        uv_depth = undistort_points(uv_depth, self.TOF.dist)
        s3d_f = np.concatenate((uv_depth, np.ones((uv_depth.shape[0], 1), dtype=uv_depth.dtype)), axis=1)
        s3d_tof = s3d_f * points[:, 2, None]
        # print(cxy)
        # print(fxy)
        # print("rot matrix")
        angles = np.array([0 / 180 * np.pi, 0 / 180 * np.pi, 0 / 180 * np.pi], dtype=np.float32)
        rot_matrix = cv2.Rodrigues(angles)[0]
        # print(rot_matrix)
        s3d_rgb = np.dot((s3d_tof + self.tvec_RGB2TOF), rot_matrix)
        uv_rgb = s3d_rgb[:, :2] / s3d_rgb[:, 2, None]
        uv_rgb = uv_rgb * fxy + cxy
        return uv_rgb


def undistort_points(points, distcoeff):
    """
    Undistort the points, the points are focalized
    :param points: k*2
    :param distcoeff:
    :return:
    """
    k1 = distcoeff[0]
    k2 = distcoeff[1]
    k3 = distcoeff[4]
    p1 = distcoeff[2]
    p2 = distcoeff[3]

    xx_ = points[:, 0] ** 2
    yy_ = points[:, 1] ** 2
    rr_ = xx_ + yy_
    xy2_ = 2 * points[:, 0] * points[:, 1]
    t = 1 / (1 + ((k3 * rr_ + k2) * rr_ + k1) * rr_)
    dx = p1 * xy2_ + p2 * (rr_ + 2 * xx_)
    dy = p1 * (rr_ + 2 * yy_) + p2 * xy2_
    x = (points[:, 0] - dx) * t
    y = (points[:, 1] - dy) * t
    return np.stack((x, y), axis=1)


def distort_points(points, distcoeff):
    """
    distort the points, the points are focalized
    :param points: k*2
    :param distcoeff:
    :return:
    """
    k1 = distcoeff[0]
    k2 = distcoeff[1]
    k3 = distcoeff[4]
    p1 = distcoeff[2]
    p2 = distcoeff[3]

    x_ = points[:, 0]
    y_ = points[:, 1]
    r = np.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + 2 * p1 * x_ * y_ + p2 * (r ** 2 + 2 * x_ ** 2)
    y__ = y_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + p1 * (r ** 2 + 2 * y_ ** 2) + 2 * p2 * x_ * y_
    return np.stack((x__, y__), axis=1)


def align_coords(points_tof, CAM):
    """
    :param points_tof: k x 3 numpy
    :param CAM:
    :return: k x 3 numpy
    """
    assert points_tof.shape[1] == 3
    assert CAM["dual"]
    uv_depth = points_tof[:, :2]
    cxy = [CAM["tof"][0, 2], CAM["tof"][1, 2]]
    fxy = [CAM["tof"][0, 0], CAM["tof"][1, 1]]
    uv_depth = (uv_depth - cxy) / fxy
    uv_depth = undistort_points(uv_depth, CAM["dist_tof"])
    s3d_f = np.concatenate((uv_depth, np.ones((uv_depth.shape[0], 1), dtype=uv_depth.dtype)), axis=1)
    s3d_tof = s3d_f * points_tof[:, 2, None]

    # s3d_rgb = np.dot(s3d_tof, CAM["R"].T) + CAM["T"].T
    s3d_rgb = np.dot((s3d_tof + CAM["T"].T), CAM["R"])

    cxy = [CAM["rgb"][0, 2], CAM["rgb"][1, 2]]
    fxy = [CAM["rgb"][0, 0], CAM["rgb"][1, 1]]
    uv_rgb = s3d_rgb[:, :2] / s3d_rgb[:, 2, None]
    uv_rgb = distort_points(uv_rgb, CAM["dist_rgb"])
    uv_rgb = uv_rgb * fxy + cxy
    return uv_rgb


def reproject3d(s2d, depth, CAM, undistort=False):
    """
    reconstruct the pose3d through pose2d and depth
    :param s2d:   k x 2  pose2d
    :param depth: k x 1  depth
    :param CAM:
    :return:  k x 3 pose3d
    """
    assert s2d.shape[0] == depth.shape[0]
    assert len(s2d.shape) == 2
    assert len(depth.shape) == 1
    cxy = [CAM["rgb"][0, 2], CAM["rgb"][1, 2]]
    fxy = [CAM["rgb"][0, 0], CAM["rgb"][1, 1]]
    s2d = (s2d - cxy) / fxy
    if undistort:
        s2d = undistort_points(s2d, CAM["dist_rgb"])

    depth = depth.reshape(-1, 1)
    s3d_xy = s2d * depth
    s3d = np.concatenate((s3d_xy, depth), axis=1)
    return s3d

# if __name__ == "__main__":
#     s2ds, s3ds, depths, _ = dump.get_dumps(filedir="bugs/0412_leg", num_item=5, width=1440, height=1080)
#
#     s2d = s2ds[0]
#     print(s2d)
#     cxy = [CAMERA["OPPO"]["rgb"][0, 2], CAMERA["OPPO"]["rgb"][1, 2]]
#     fxy = [CAMERA["OPPO"]["rgb"][0, 0], CAMERA["OPPO"]["rgb"][1, 1]]
#     s2d_f = (s2d - cxy) / fxy
#     print(s2d_f)
#     s2d_f_undist = undistort_points(s2d_f, CAMERA["OPPO"]["dist_rgb"])
#     s2d_f_check = distort_points(s2d_f_undist, CAMERA["OPPO"]["dist_rgb"])
#
#     s2d_undist = s2d_f_undist * fxy + cxy
#     s2d_check = s2d_f_check * fxy + cxy
#     print(s2d_undist)
#     print(s2d_check)
#     img = viz.draw_s2d(s2d, size=1440, rule="SDK14")
#     img = viz.draw_s2d(s2d_check, image=img, rule="SDK14")
#     cv2.imshow("test", cv2.resize(img, (500, 500)))
#     cv2.waitKey()
