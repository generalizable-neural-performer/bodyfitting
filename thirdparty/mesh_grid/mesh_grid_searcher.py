import torch
from mesh_grid import insert_grid_surface, cumsum, search_nearest_point, search_inside_mesh, search_intersect, search_nearest_point_backward

class SurfaceNearest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, verts, faces, tri_num, tri_idx, num, minmax, step):
        nearest_faces = torch.zeros(points.shape[-2], dtype=torch.int32).to(verts.device)
        coeff = torch.zeros(points.shape, dtype=torch.float32).to(verts.device)
        nearest_pts = torch.zeros_like(coeff)
        search_nearest_point(points, verts, faces, tri_num,
                              tri_idx, num, minmax, step,
                              nearest_faces, nearest_pts, coeff)
        ctx.save_for_backward(points, verts, faces, nearest_faces, coeff)
        return nearest_pts, nearest_faces
    # @staticmethod
    # def backward(ctx, dpts, dfaces = None):
    #     grad_p = grad_v = None
    #     if any(ctx.needs_input_grad[:2]):
    #         points, verts, faces, nearest_faces, coeff = ctx.saved_tensors
    #         sz = [int(points.shape[0]), int(tri.shape[1]), int(verts.shape[-1])]
    #         grad = torch.zeros(sz+sz[-1:], dtype=torch.float32).to(verts.device)
    #         search_nearest_point_backward(points, verts, faces, nearest_faces, grad)
    #         tri = faces.index_select(0, torch.clamp(nearest_faces, min = 0)).reshape(-1)
    #         if ctx.needs_input_grad[0]:
    #             grad_p = torch.matmul(dpts.view(-1,1,sz[-1]),
    #                      torch.matmul((verts.index_select(0,tri).reshape(sz) -
    #                                    points.reshape(sz[0],1,-1)).transpose(1,2),
    #                                   -grad.sum(-1))).view(-1,sz[-1])
    #         if ctx.needs_input_grad[1]:
    #             grad_v = torch.matmul(dpts.view(n,1,-1), \
    #                      torch.matmul((verts.index_select(0,tri).reshape(sz) -
    #                                    points.reshape(sz[0],1,-1)).transpose(1,2),
    #                                    grad.view(sz[0],sz[-1],-1))).view(sz).transpose(1,2) +
    #                      torch.matmul(coeff.view(sz[0],-1,1),
    #                                   dpts.view(sz[0],1,-1))
    #             args = [torch.cat([tri.view(1,-1),
    #                     torch.tensor(range(sz[0]*sz[-1]),
    #                         dtype = torch.int64,
    #                         device = points.device).view(1,-1)], 0),
    #                     torch.ones(sz[0]*sz[-1],
    #                         dtype = points.dtype,
    #                         device = points.device),
    #                     torch.Size([len(verts), sz[0]*sz[-1]])]
    #             I = torch.sparse.FloatTensor(*args) if points.dtype == torch.float32 else
    #                 torch.sparse.DoubleTensor(*args)
    #             grad_v = torch.matmul(I, grad_v.view(sz[0]*sz[-1],-1))
    #     return tuple([grad_p,grad_v]+[None]*6)

class MeshGridSearcher:
    def __init__(self, verts=None, faces=None):
        if verts is not None and faces is not None:
            self.set_mesh(verts, faces)

    def set_mesh(self, verts, faces):
        self.verts = verts
        self.faces = faces
        _min, _ = torch.min(verts, 0)
        _max, _ = torch.max(verts, 0)
        self.step = (torch.cumprod(_max-_min, 0)[-1] / len(verts)) ** (1./3.)
        l = _max - _min
        c = (_max + _min) / 2
        l = torch.max(torch.floor(l/self.step), torch.zeros_like(l)) + 1
        _min_step = c - self.step * l / 2
        self.num = torch.cat([l, torch.cumprod(l,0)[-1:]]).int()
        self.minmax = torch.cat([_min_step, _max])

        self.tri_num = torch.zeros(self.num[-1], dtype=torch.int32).to(verts.device)
        self.tri_idx = torch.zeros(1, dtype=torch.int32).to(verts.device)

        insert_grid_surface(self.verts, 
                            self.faces, 
                            self.minmax, self.num, self.step,
                            self.tri_num, self.tri_idx)
    
    def nearest_points(self, points):
        points = points.to(self.verts.device)
        return SurfaceNearest.apply(points, self.verts, self.faces, self.tri_num,
                                    self.tri_idx, self.num, self.minmax, self.step)
    def inside_mesh(self, points):
        points = points.to(self.verts.device)
        inside = torch.zeros(points.shape[-2], dtype=torch.float32).to(self.verts.device)
        search_inside_mesh(points, self.verts, self.faces, self.tri_num,
                            self.tri_idx, self.num, self.minmax, self.step, inside)
        return inside

    def intersects_any(self, origins, directions):
        origins = origins.to(self.verts.device)
        directions = directions.to(self.verts.device)
        intersect = torch.zeros(origins.shape[-2], dtype=torch.bool).to(self.verts.device)
        search_intersect(origins, directions, self.verts, self.faces, self.tri_num,
                        self.tri_idx, self.num, self.minmax, self.step, intersect)
        return intersect
