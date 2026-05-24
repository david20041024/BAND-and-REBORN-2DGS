import torch


class VOXEL:
    def __init__(
        self,
        mesh: torch.Tensor,
        nonboundary: torch.Tensor,
        expand=0.1,
        size=0.004
    ):

        self.aim = None
        self.mesh = mesh
        self.nonboundary = nonboundary
        self.expand = expand
        self.size = size

        min_bound = mesh.min(dim=0)[0]
        max_bound = mesh.max(dim=0)[0]

        self.min_bound = min_bound - expand
        self.max_bound = max_bound + expand

        self.grid_dim = torch.ceil(
            (self.max_bound - self.min_bound) / size
        ).long()

    def world_to_voxel(self, pts, device):

        voxel = torch.floor(
            (pts - self.min_bound.to(device)) / self.size
        ).long()

        voxel = torch.clamp(
            voxel,
            min=0
        )

        voxel[:, 0] = torch.clamp(voxel[:, 0], max=self.grid_dim[0]-1)
        voxel[:, 1] = torch.clamp(voxel[:, 1], max=self.grid_dim[1]-1)
        voxel[:, 2] = torch.clamp(voxel[:, 2], max=self.grid_dim[2]-1)

        return voxel

    def find_boundary(self, gaussians):

        xyz = gaussians.get_xyz
        device = xyz.device

        # -------------------------
        # 1. filter inside AABB
        # -------------------------
        inside = (
            (xyz[:, 0] >= self.min_bound[0]) &
            (xyz[:, 0] <= self.max_bound[0]) &
            (xyz[:, 1] >= self.min_bound[1]) &
            (xyz[:, 1] <= self.max_bound[1]) &
            (xyz[:, 2] >= self.min_bound[2]) &
            (xyz[:, 2] <= self.max_bound[2])
        )

        valid_idx = torch.where(inside)[0]
        valid_xyz = xyz[valid_idx]

        # -------------------------
        # 2. initialize voxel mask
        # True = possible boundary
        # -------------------------
        voxel_mask = torch.ones(
            *self.grid_dim.tolist(),
            dtype=torch.bool,
            device=device
        )

        # -------------------------
        # 3. remove nonboundary voxels
        # -------------------------
        nb_voxel = self.world_to_voxel(
            self.nonboundary.to(device),
            device
        )

        voxel_mask[
            nb_voxel[:, 0],
            nb_voxel[:, 1],
            nb_voxel[:, 2]
        ] = False

        # -------------------------
        # 4. gaussian -> voxel
        # -------------------------
        g_voxel = self.world_to_voxel(
            valid_xyz,
            device
        )

        is_boundary = voxel_mask[
            g_voxel[:, 0],
            g_voxel[:, 1],
            g_voxel[:, 2]
        ]

        self.aim = valid_idx[is_boundary]
        print(self.aim.shape[0], "boundary gaussians found out of", xyz.shape[0])

        return self