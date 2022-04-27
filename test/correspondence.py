import numpy as np
import torch
import neural_renderer as nr
import scipy.spatial.distance, scipy.optimize

if __name__ == "__main__":
    source_dir = "./data/bodyfitting/0125/debug/opt_smpl/smpl.obj"
    target_dir = "./data/bodyfitting/0125/pred/bodyscan/bodyscan_6890.obj"

    vert_src, face_src = nr.load_obj(source_dir, normalization=False, load_texture=False)
    vert_dst, face_dst, text_dst = nr.load_obj(target_dir, normalization=False, load_texture=True)
    vert_dst, vert_src = vert_dst.cpu().numpy(), vert_src.cpu().numpy()
    print(vert_dst.shape, vert_src.shape)
    C = scipy.spatial.distance.cdist(vert_src, vert_dst)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)
    print(row_ind.shape, col_ind.shape)
    print(row_ind, col_ind)
    new_vert1 = torch.tensor(vert_dst[col_ind])
    new_vert2 = torch.tensor(vert_src[col_ind])
    nr.save_obj("./data/bodyfitting/0125/debug/opt_smpl/new_vert1.obj", new_vert1, face_src)
    nr.save_obj("./data/bodyfitting/0125/debug/opt_smpl/new_vert2.obj", new_vert2, face_dst)


