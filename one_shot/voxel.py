import torch
import torch.nn.functional as F


class VOXEL:
    def __init__(self, xyz: torch.Tensor, mesh: torch.Tensor, voxel_size=0.004):

        self.xyz = xyz
        self.mesh = mesh
        self.voxel_size = voxel_size

        # ---------------------------
        # 1. bounding box (from mesh)
        # ---------------------------
        self.min_bound = mesh.min(dim=0)[0]
        self.max_bound = mesh.max(dim=0)[0]

        grid_size = ((self.max_bound - self.min_bound) / voxel_size).ceil().long()
        self.grid_size = grid_size

        # ---------------------------
        # 2. occupancy grid init
        # ---------------------------
        self.occ = torch.zeros(
            (grid_size[0].item(), grid_size[1].item(), grid_size[2].item()),
            device=xyz.device,
            dtype=torch.bool
        )

        # ---------------------------
        # 3. mesh -> voxel occupancy
        # ---------------------------
        mesh_idx = ((mesh - self.min_bound) / voxel_size).long()
        mesh_idx = torch.clamp(mesh_idx, min=0, max=grid_size - 1)

        self.occ[mesh_idx[:, 0], mesh_idx[:, 1], mesh_idx[:, 2]] = True


        for _ in range(10):
            self.occ = self._dilate(self.occ)

        # ---------------------------
        # 5. xyz -> voxel query
        # ---------------------------
        xyz_idx = ((xyz - self.min_bound) / voxel_size).long()
        xyz_idx = torch.clamp(xyz_idx, min=0, max=grid_size - 1)

        mask = self.occ[xyz_idx[:, 0], xyz_idx[:, 1], xyz_idx[:, 2]]

        # ---------------------------
        # 6. output indices
        # ---------------------------
        self.valid_idx = mask.nonzero(as_tuple=True)[0]
        self.invalid_idx = (~mask).nonzero(as_tuple=True)[0]

    # ----------------------------------
    # 3D dilation via max pooling
    # ----------------------------------
    def _dilate(self, x: torch.Tensor):
        x = x.float()[None, None]
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        return (x[0, 0] > 0.5)