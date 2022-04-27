import argparse
import json
import copy
import os

import numpy as np
import cv2
import poseviz

import openpose.util
from openpose.body import Body


def get_pose(oriImg, body_estimation):
    candidate, subset = body_estimation(oriImg)
    np.set_printoptions(suppress=True)
    poses = []
    for idx, item in enumerate(subset):
        single_pose = np.zeros((18, 3))
        for i in range(18):
            kpt_idx = int(item[i])
            if kpt_idx == -1:
                single_pose[i] = [0, 0, 0]
            else:
                single_pose[i] = candidate[kpt_idx][0:3]
        mapping = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
        coco_pose = single_pose[mapping]
        poses.append(coco_pose)
        # cp = poseviz.draw_s2d(coco_pose, image=oriImg, rule="COCO17")
        # cv2.imshow("cp", cp)
        # cv2.waitKey()
    canvas = copy.deepcopy(oriImg)
    canvas = openpose.util.draw_bodypose(canvas, candidate, subset)
    cv2.imwrite("results/OpenPose_inf.png", canvas)
    cv2.waitKey()
    return poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="openpose image")
    parser.add_argument(
        "--image",
        help="path to image",
        default="./data/bodyfitting/0125/images/image.jpg",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        help="openpose output dir",
        default="./data/bodyfitting/0125/pred/",
        type=str
    )
    args = parser.parse_args()

    body_estimation = Body('models/body_pose_model.pth')
    img = cv2.imread(args.image)
    ori_img = img.copy()
    kpts = get_pose(img, body_estimation)
    kpts = np.array(kpts).reshape([-1, 17, 3])
    kpts = np.float32(kpts)
    print(kpts.shape)
    cp = poseviz.draw_s2d_as_points(kpts[0][:, :2], image=ori_img)
    # cv2.imshow("cp", cp)
    # cv2.waitKey()
    obj = {"version": 1.3}
    people = []
    for i in range(kpts.shape[0]):
        openpose2d = np.zeros((25, 3))
        mapping = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
        openpose2d[mapping] = kpts[i]
        cp = poseviz.draw_s2d_as_points(openpose2d[:, :2], image=ori_img)
        person_obj = {
            "person_id": [-1],
            "pose_keypoints_2d": openpose2d.flatten().tolist()
        }
        people.append(person_obj)
    obj["people"] = people
    obj_json = json.dumps(obj)
    output_dir = os.path.join(args.output_dir, "openpose2d")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "image_keypoints.json"), "w") as f:
        f.write(obj_json)
    cv2.imwrite(os.path.join(output_dir, "image_rendered.png"), cp)
