import json
import os

import cv2
import numpy as np
from scipy.misc import face
import torch
import neural_renderer as nr
from torchvision.transforms import Normalize

import constants
from utils.imutils import crop
from utils.geometry import rotation_matrix_to_angle_axis
import struct, re

def copy_obj(obj_dir, target_dir):
    base = os.path.dirname(obj_dir)
    target_base = os.path.dirname(target_dir)
    verts = []
    up_axis = 1
    with open(obj_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('mtllib'):
                mtlfile = line.split()[1]
            elif line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:]])
        verts = np.array(verts)
        up_axis = (verts.max(0)-verts.min(0)).argmax()

    if up_axis == 1:
        os.system(f"cp {obj_dir} {target_dir}")
    else:
        # print('rewrite obj')
        with open(obj_dir, 'r') as f1:
            lines1 = f1.readlines()
            with open(target_dir, 'w') as f2:
                for line1 in lines1:
                    if line1.startswith('v '):
                        v = np.array([float(i) for i in line1.split()[1:]])
                        new_v = v.copy()
                        new_v[1] = v[up_axis]
                        new_v[up_axis] = -v[1]
                        line2 = f'v {new_v[0]} {new_v[1]} {new_v[2]}\n'
                    else:
                        line2 = line1
                    f2.write(line2)
    
    mtl = os.path.join(base, mtlfile)
    target_mtl = os.path.join(target_base, mtlfile)
    os.system(f"cp {mtl} {target_mtl}")

    teximg_list = []
    with open(mtl, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # if line.startswith('map_Kd'):
            if 'map_Kd' in line.split():
                teximg_list.append(line.split()[-1])
    for teximg in teximg_list:
        tex = os.path.join(base, teximg)
        target_tex = os.path.join(target_base, teximg)
        os.system(f"mkdir -p {os.path.dirname(target_tex)} && cp {tex} {target_tex}")

# RenderPeople dataset has no mtl file, add mtlfile to load by neural renderer
def mtl_check(obj_dir):

    base = os.path.dirname(obj_dir)
    subject = os.path.basename(obj_dir)[:-8]
    has_mtl = False
    line_id = None
    with open(obj_dir, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("usemtl"):
                has_mtl = True
            if line.startswith("f ") and (line_id is None):
                line_id = i

    if not has_mtl:
        new_lines = [f"mtllib {subject}.mtl\n", "usemtl default\n"]
        lines = lines[:line_id] + new_lines + lines[line_id:]
        with open(obj_dir, 'w') as f:
            f.writelines(lines)
        with open(os.path.join(base, subject+".mtl"), 'w') as f_mtl:
            mtl_lines = ['newmtl default\n',
                        'Ka 0 0 0\n',
                        'Kd 0.588 0.588 0.588\n'
                        'Ks 0 0 0\n'
                        'Ke 0 0 0\n'
                        'Tf 1 1 1\n'
                        'illum 0\n'
                        'Ns 2\n'
                        f'map_Kd tex/{subject}_dif_2k.jpg']
            f_mtl.writelines(mtl_lines)

def image_cropping(mask):
    a = np.where(mask != 0)
    h, w = list(mask.shape[:2])

    top, left, bottom, right = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    bbox_h, bbox_w = bottom - top, right - left

    # padd bbox
    bottom = min(int(bbox_h*0.1+bottom), h)
    top = max(int(top-bbox_h*0.1), 0)
    right = min(int(bbox_w*0.1+right), w)
    left = max(int(left-bbox_h*0.1), 0)
    bbox_h, bbox_w = bottom - top, right - left

    if bbox_h >= bbox_w:
        w_c = (left+right) / 2
        size = bbox_h
        if w_c - size / 2 < 0:
            left = 0
            right = size
        elif w_c + size / 2 >= w:
            left = w - size
            right = w
        else:
            left = int(w_c - size / 2)
            right = left + size
    else:   # bbox_w >= bbox_h
        h_c = (top+bottom) / 2
        size = bbox_w
        if h_c - size / 2 < 0:
            top = 0
            bottom = size
        elif h_c + size / 2 >= h:
            top = h - size
            bottom = h
        else:
            top = int(h_c - size / 2)
            bottom = top + size
    
    return top, left, bottom, right

def load_openpose(json_name, only_one=True):
    with open(json_name, 'r') as fid:
        d = json.load(fid)
    if len(d.get('people',[])) == 0:
        return None
    data = []
    for label in d['people']:
        data_ = {}; ID = -1
        for k, p in label.items():
            if 'id' in k:
                ID = np.reshape(p,-1)[0]
            elif 'keypoints' in k:
                p = np.reshape(p, -1)
                if len(p) == 0: continue
                if (p - np.floor(p)).max() <= 0:
                    p = p.astype(np.int32)
                dim = re.findall('([2-9]d)', k)
                dim = 2 if len(dim) == 0 else int(dim[-1][0])
                if len(p) % (dim+1) == 0 and (p.dtype != np.int32 or p.max()==0):
                    p = p.reshape(-1,dim+1)
                    if abs(p[:,-1]).max()  <= 0:
                        continue
                elif len(p) % dim == 0:
                    p = p.reshape(-1,dim)
                else:
                    p = p[:(len(p)//dim)*dim].reshape(-1,dim)
                k = k.replace('_keypoints','').replace('_%dd'%dim,'')
                data_[k] = p
        if ID < 0 and isinstance(data, list):
            data.append(data_)
        elif ID > 0:
            if isinstance(data,list):
                data = {(-k-1):d for k, d in enumerate(data)}
            data[ID] = data_

    if len(data) == 0:
        return None
    elif only_one:
        j = 0; score = 0
        for i, d in (enumerate(data) if isinstance(data,list) else data.items()):
            s = sum([p[:,-1].sum() for k, p in d.items()])
            if s > score:
                j = i; score = s
        return data[j]
    else:
        return data

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

def get_mask_from_pose2seg(seg_image):
    seg_img = cv2.imread(seg_image, cv2.IMREAD_GRAYSCALE)
    return seg_img


def get_depthmap(imagefile):
    depth = cv2.imread(imagefile, cv2.IMREAD_ANYDEPTH)
    return depth


def keypoints_list_from_openpose(openpose_file):
    keypoints_list = []
    with open(openpose_file, 'r') as f:
        people = json.load(f)['people']
        if len(list(people)) > 0:
            for item in list(people):
                keypoints = np.array(item['pose_keypoints_2d']).reshape(25, 3)
                keypoints_list.append(keypoints)
        else:
            return None
    return keypoints_list


def keypoints_from_openpose_single(openpose_file):
    keypoints_list = keypoints_list_from_openpose(openpose_file)
    keypoints_2d = None
    if keypoints_list is None or len(keypoints_list) == 0:
        valid = False
    else:
        conf_max = -1
        for person_id, keypoints in enumerate(keypoints_list):
            keypoints_2d_arr = np.array(keypoints.reshape(1, 25, 3))
            sum = np.sum(keypoints[:, -1] > 0.1)
            if sum > 10 and sum > conf_max:
                conf_max = sum
                keypoints_2d = keypoints_2d_arr
        if conf_max == -1:
            valid = False
            keypoints_2d = None
        else:
            valid = True
    return keypoints_2d, valid


def keypoints_hand_face_list_from_openpose(openpose_file):
    keypoints_list = []
    keys = ["hand_left_keypoints_2d", "hand_right_keypoints_2d", "face_keypoints_2d"]
    # use full 68 points
    mapping = list(range(17, 17+51)) + list(range(0, 17))
    with open(openpose_file, 'r') as f:
        people = json.load(f)['people']
        if len(list(people)) > 0:
            for item in list(people):
                keypoints_ = None
                for key in keys:
                    keypoints = np.array(item[key]).reshape(-1, 3)
                    if key == "face_keypoints_2d":
                        keypoints = keypoints[mapping]
                    if keypoints_ is None:
                        keypoints_ = keypoints
                    else:
                        keypoints_ = np.concatenate((keypoints_, keypoints), axis=0)
                keypoints_list.append(keypoints_)
        else:
            return None
    return keypoints_list



def keypoints_from_sdk2d_video(pred_dir, num_keypoints=23):
    pose2d_file = os.path.join(pred_dir, "pose2d_hrnet.txt")
    conf2d_file = os.path.join(pred_dir, "confidence2d_hrnet.txt")
    index_file = os.path.join(pred_dir, "imglist_origin.txt")
    p2d = []
    conf2d = []
    indexs = []
    with open(pose2d_file, "r") as f:
        for line in f.readlines():
            p2d.append(list(map(float, line.split())))
    p2d = np.array(p2d).reshape([-1, num_keypoints, 2])
    with open(conf2d_file, "r") as f:
        for line in f.readlines():
            conf2d.append(list(map(float, line.split())))
    conf2d = np.array(conf2d).reshape([-1, num_keypoints, 1])
    with open(index_file, "r") as f:
        for line in f.readlines():
            indexs.append(list(map(int, line.split()))[0])
    # modify toe pos
    for i in range(p2d.shape[0]):
        p2d[i][17] = p2d[i][12] + (p2d[i][17] - p2d[i][12]) * 1.4
        p2d[i][18] = p2d[i][13] + (p2d[i][18] - p2d[i][13]) * 1.4
    keypoints = np.concatenate((p2d, conf2d), axis=2)
    return keypoints, indexs


def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.2):
    keypoints = np.reshape(keypoints, (-1, 3))
    valid = keypoints[:, -1] > detection_thresh
    valid_keypoints = keypoints[valid][:, :-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale


def make_dirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def make_hmr_vec(pose, betas, cam_t, smpl_type):
    if smpl_type == "smplx":
        pose = pose.reshape(1, 66)
    elif smpl_type == "smpl":
        pose = pose.reshape(1, 72)
    else:
        print("smpl_type error!")
    betas = betas.reshape(1, 10)
    cam_t = cam_t.reshape(1, 3)
    rtn = np.concatenate((pose, betas, cam_t), axis=1)
    return rtn


def render_smpl_eval(pred_vertices, pred_faces, img, image_size, camera_translation, K, pose=None, 
                    output_filename=None, dist=100, white_bkgd=False):
    pred_faces = pred_faces.astype(np.int32)
    vertices = torch.tensor(pred_vertices, dtype=torch.float32)
    faces = torch.tensor(pred_faces, dtype=torch.int32)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    texture_size = 2
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    if pose is None:
        t = camera_translation.reshape([1, 1, 3])
        R = np.eye(3).reshape([1, 3, 3])
    else:
        t = pose[:3, 3].reshape([1, 1, 3])
        R = pose[:3,:3].reshape([1, 3, 3])
    renderer = nr.Renderer(image_size=image_size, fill_back=False, camera_mode='projection', 
                            K=K, R=R, t=t, orig_size=image_size, near=0, far=dist*2)
    images, rend_depth, _ = renderer(vertices.cuda(), faces.cuda(), textures)
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    rend_depth = rend_depth.detach().cpu().numpy()[0]
    valid_mask = (rend_depth < dist*2)[:, :, None]
    # height = int(constants.HEIGHT)
    # width = int(constants.WIDTH)
    height, width = image_size, image_size
    valid_mask = valid_mask[:height, :width, :]
    image = image[:height, :width, :]
    w = 1.0
    # output_img = (image * valid_mask * 255 + (1 - valid_mask) * img)
    if not white_bkgd:
        output_img = (image * valid_mask * 255 * w + img * valid_mask * (1-w) + (1 - valid_mask) * img)
    else:
        output_img = (image * valid_mask * 255 * w + 255 * (1-valid_mask))
    fig = output_img.astype(np.uint8)
    cv2.imwrite(output_filename, fig)
    w = 0.7
    output_img = (image * valid_mask * 255 * w + img * valid_mask * (1 - w) + (1 - valid_mask) * img)
    fig = output_img.astype(np.uint8)
    return fig


def process_image(img, keypoints, input_res=224):
    """
    :param keypoints: numpy array (n, 3): x, y, conf
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    # img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if keypoints is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        center, scale = bbox_from_keypoints(keypoints)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img, center, scale


def render_smpl(pred_vertices, pred_faces, img, camera_translation, output_filename=None):
    pred_faces = pred_faces.astype(np.int32)
    vertices = torch.tensor(pred_vertices, dtype=torch.float32)
    faces = torch.tensor(pred_faces, dtype=torch.int32)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    texture_size = 2
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    t = camera_translation.reshape([1, 1, 3])
    R = np.eye(3).reshape([1, 3, 3])
    fx = constants.FOCAL_LENGTH
    fy = constants.FOCAL_LENGTH
    cx = constants.IMG_RES // 2
    cy = constants.IMG_RES // 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32).reshape([1, 3, 3])
    renderer = nr.Renderer(image_size=224, fill_back=False, camera_mode='projection', K=K, R=R, t=t, orig_size=224)
    images, rend_depth, _ = renderer(vertices.cuda(), faces.cuda(), textures)
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    rend_depth = rend_depth.detach().cpu().numpy()[0]
    valid_mask = (rend_depth < 100.0)[:, :, None]
    output_img = (image[:, :, :3] * valid_mask + (1 - valid_mask) * img)
    fig = (255 * output_img).astype(np.uint8)
    cv2.imwrite(output_filename, fig)
    # print("{} renderer created!".format(output_filename))


def render_smpl_fixed_size(pred_vertices, pred_faces, img, height, width, camera_translation, K, pose=None, dist=100):
    pred_faces = pred_faces.astype(np.int32)
    vertices = torch.tensor(pred_vertices, dtype=torch.float32)
    faces = torch.tensor(pred_faces, dtype=torch.int32)
    if len(vertices.shape) == 2:
        vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    if len(faces.shape) == 2:
        faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    texture_size = 2
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    if pose is None:
        t = camera_translation.reshape([1, 1, 3])
        R = np.eye(3).reshape([1, 3, 3])
    else:
        t = pose[:3, 3].reshape([1, 1, 3])
        R = pose[:3,:3].reshape([1, 3, 3])
    image_size = max(constants.FX, 1000)
    renderer = nr.Renderer(image_size=image_size, fill_back=False, camera_mode='projection', 
                            K=K, R=R, t=t, orig_size=image_size, near=0, far=dist*2)
    images, rend_depth, _ = renderer(vertices.cuda(), faces.cuda(), textures)
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    rend_depth = rend_depth.detach().cpu().numpy()[0]
    valid_mask = (rend_depth < dist*2)[:, :, None]
    valid_mask = valid_mask[:height, :width, :]
    image = image[:height, :width, :]
    output_img = (image * valid_mask * 255 + (1 - valid_mask) * img)
    fig = output_img.astype(np.uint8)
    return fig
    # cv2.imwrite(output_filename, fig)
    # print("{} renderer created!".format(output_filename))


def update_camera_info(target_dir, height, width):
    info_file = os.path.join(target_dir, "intrinsics")
    intrinsics = []
    if os.path.exists(info_file):
        with open (os.path.join(target_dir, "intrinsics"), "r") as f:
            for line in f:
                nums = line[:-1].split(" ")[-4:]
            for item in nums:
                intrinsics.append(float(item))
        constants.FX = width * intrinsics[0]
        constants.FY = height * intrinsics[1]
        constants.CX = width * intrinsics[2]
        constants.CY = height * intrinsics[3]
        constants.WIDTH = width
        constants.HEIGHT = height
    else:
        # print("no intrinsic file found!  use default!")
        constants.FX = max(1, width, height)
        constants.FY = max(1, width, height)
        constants.CX = width * 0.5
        constants.CY = height * 0.5
        constants.WIDTH = width
        constants.HEIGHT = height
    # calib_file = os.path.join(target_dir, "calib.bin")


def keypoints_from_sdk2d_face(pred_dir, num_keypoints=106):
    face2d_file = os.path.join(pred_dir, "face106.txt")
    p2d = []
    with open(face2d_file, "r") as f:
        for line in f.readlines():
            p2d.append(list(map(float, line.split())))
    p2d = np.array(p2d).reshape([-1, num_keypoints, 2])
    mapping = list(range(33, 64)) + list(range(84, 104)) + [x * 2 for x in range(17)]
    p2d = p2d[:, mapping]
    conf2d = np.ones((p2d.shape[0], p2d.shape[1], 1))
    keypoints = np.concatenate((p2d, conf2d), axis=2)
    return keypoints

def normalize_v3(arr: np.ndarray):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices: np.ndarray, faces: np.ndarray):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def normalize_v3_torch(norm: torch.Tensor):
    norm = norm / (torch.norm(norm, dim=-1, keepdim=True) + 1e-8)
    return norm

def compute_normal_torch(vertices: torch.Tensor, faces: torch.Tensor):
    vertices, faces = vertices.view(-1, 3), faces.view(-1,3)
    va = torch.index_select(vertices, 0, faces[:,0])
    vb = torch.index_select(vertices, 0, faces[:,1])
    vc = torch.index_select(vertices, 0, faces[:,2])
    n = torch.cross(vb - va, vc - va, dim=1)
    n = normalize_v3_torch(n)
    norm = 0
    n = n.reshape(len(faces),-1)

    # different with numpy implementation, we use sparse multiplication for gradient computation
    for j in range(3):
        i = torch.cat((faces[:,j].view(1,-1), \
                       torch.LongTensor(range(len(faces))).view(1,-1).to(faces.device)), 0)
        I = torch.sparse_coo_tensor(i, torch.ones(len(faces),device=faces.device, dtype=n.dtype), \
                                    torch.Size([int(vertices.shape[0]),len(faces)]))
        vnj = torch.sparse.mm(I, n)
        norm = norm + vnj.view(-1,3)
    return normalize_v3_torch(norm)

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False, with_texture_image=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
        elif 'mtllib' in line.split():
            mtlname = line.split()[-1]
            mtlfile = os.path.join(os.path.dirname(mesh_file), mtlname)
            with open(mtlfile, 'r') as fmtl:
                mtllines = fmtl.readlines()
                for mtlline in mtllines:
                    # if mtlline.startswith('map_Kd'):
                    if 'map_Kd' in mtlline.split():
                        texname = mtlline.split()[-1]
                        texfile = os.path.join(os.path.dirname(mesh_file), texname)
                        texture_image = cv2.imread(texfile)
                        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
                        break

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        if with_texture_image:
            return vertices, faces, norms, face_normals, uvs, face_uvs, texture_image
        else:
            return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        # norms = np.array(norm_data)
        # norms = normalize_v3(norms)
        # face_normals = np.array(face_norm_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

