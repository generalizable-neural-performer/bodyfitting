import argparse
import json
import os

import cv2
import numpy as np
from FileDecoder import StreamerFileDecoder
from tqdm import tqdm


def read_rgbd(rgbd_file_path,
              save_path=None,
              read_color=True,
              read_depth=False,
              read_mask=False):
    decoder = StreamerFileDecoder(rgbd_file_path)
    color_frames = []
    depth_frames = []
    mask_frames = []

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if read_color:
            color_path = save_path
        if read_depth:
            depth_path = os.path.join(save_path, 'depth')
        if read_mask:
            mask_path = os.path.join(save_path, 'mask')

        if read_color and not os.path.exists(color_path):
            os.mkdir(color_path)
        if read_depth and not os.path.exists(depth_path):
            os.mkdir(depth_path)
        if read_mask and not os.path.exists(mask_path):
            os.mkdir(mask_path)

    for i in tqdm(range(decoder.frame_count)):
        frame_elems = decoder.getFrame(frame_index=i)
        if frame_elems is None:
            break
        color_mat, _, _, depth_16uc1, _, index_8uc1 = frame_elems
        if read_color:
            color_frames.append(color_mat)
            if save_path is not None:
                cv2.imwrite(
                    os.path.join(color_path, 'frame_%06d.jpg' % i), color_mat)
        if read_depth:
            depth_frames.append(depth_16uc1)
            if save_path is not None:
                cv2.imwrite(
                    os.path.join(depth_path, '%06d.png' % i), depth_16uc1,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if read_mask:
            mask_frames.append(index_8uc1)
            if save_path is not None:
                cv2.imwrite(
                    os.path.join(mask_path, '%06d.png' % i), index_8uc1,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

    rgbd = {}
    if read_color:
        rgbd['color'] = color_frames
    if read_depth:
        rgbd['depth'] = depth_frames
    if read_mask:
        rgbd['mask'] = mask_frames

    for key, val in decoder.color_camera_intrinsics_dict.items():
        if isinstance(val, np.ndarray):
            decoder.color_camera_intrinsics_dict[key] = val.tolist()
    for key, val in decoder.depth_camera_intrinsics_dict.items():
        if isinstance(val, np.ndarray):
            decoder.depth_camera_intrinsics_dict[key] = val.tolist()

    for key, val in decoder.extrinsics_dict.items():
        if isinstance(val, np.ndarray):
            decoder.extrinsics_dict[key] = val.tolist()

    rgbd['color_camera'] = decoder.color_camera_intrinsics_dict
    rgbd['depth_camera'] = decoder.depth_camera_intrinsics_dict
    rgbd['extrinsics'] = decoder.extrinsics_dict

    header_dict = {}
    header_dict['color_camera'] = rgbd['color_camera']
    header_dict['depth_camera'] = rgbd['depth_camera']
    header_dict['extrinsics'] = rgbd['extrinsics']

    # if save_path:
    #     json_path = os.path.join(save_path, "camera_header.json")
    #     with open(json_path, 'w') as f:
    #         json.dump(header_dict, f)

    return rgbd


def read_intrinsics(rgbd_file_path):
    decoder = StreamerFileDecoder(rgbd_file_path)
    for key, val in decoder.color_camera_intrinsics_dict.items():
        if isinstance(val, np.ndarray):
            decoder.color_camera_intrinsics_dict[key] = val.tolist()
    for key, val in decoder.depth_camera_intrinsics_dict.items():
        if isinstance(val, np.ndarray):
            decoder.depth_camera_intrinsics_dict[key] = val.tolist()
    return {
        'color_camera': decoder.color_camera_intrinsics_dict,
        'depth_camera': decoder.depth_camera_intrinsics_dict
    }


def read_extrinsics(rgbd_file_path):
    decoder = StreamerFileDecoder(rgbd_file_path)
    for key, val in decoder.extrinsics_dict.items():
        if isinstance(val, np.ndarray):
            decoder.extrinsics_dict[key] = val.tolist()
    return decoder.extrinsics_dict


def save_camera_header(rgbd_file_path,
                       save_path,
                       filename='camera_header.json'):
    header_dict = read_intrinsics(rgbd_file_path)
    header_dict['extrinsics'] = read_extrinsics(rgbd_file_path)
    with open(os.path.join(save_path, filename), 'w') as f:
        json.dump(header_dict, f)


def main():
    parser = argparse.ArgumentParser(description='Rgbd analyser')
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        dest='rgbd_file_path',
        help='RGBD file path',
        default='./rgbd/pc_test/test/k4ARecord_05_10_20_33_06_cam5.rgbd')
    parser.add_argument(
        '-d',
        '--depth',
        type=bool,
        dest='read_depth',
        help='save depth image',
        default=False)
    parser.add_argument(
        '-m',
        '--mask',
        type=bool,
        dest='read_mask',
        help='save mask image',
        default=False)
    args = parser.parse_args()
    rgbds = sorted([f for f in os.listdir(args.rgbd_file_path) if '.rgbd' in f])
    for rgbd in tqdm(rgbds):
        save_path = os.path.join(args.rgbd_file_path, rgbd[-9:-5])
        os.makedirs(save_path, exist_ok=True)
        # read_rgbd(
        #     os.path.join(args.rgbd_file_path, rgbd),
        #     save_path=save_path,
        #     read_color=True,
        #     read_depth=args.read_depth,
        #     read_mask=args.read_mask)
        save_camera_header(
            os.path.join(args.rgbd_file_path, rgbd),
            save_path=save_path
        )

if __name__ == '__main__':
    main()
