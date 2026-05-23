import torch


class VOXEL:
    def __init__(
        self,
        xyz: torch.Tensor,
        mesh: torch.Tensor,
        expand=0.1
    ):
        """
        xyz   : (N,3) gaussian points
        mesh  : (M,3) mesh points

        expand:
            AABB boundary expansion size
        """

        assert xyz is not None and mesh is not None

        device = xyz.device

        self.xyz = xyz
        self.mesh = mesh

        # =====================================================
        # 1. mesh AABB
        # =====================================================
        min_bound = mesh.min(dim=0)[0]
        max_bound = mesh.max(dim=0)[0]

        # =====================================================
        # 2. expand AABB
        # =====================================================
        min_bound = min_bound - expand
        max_bound = max_bound + expand

        self.min_bound = min_bound
        self.max_bound = max_bound

        # =====================================================
        # 3. xyz inside test
        # =====================================================
        inside = (
            (xyz[:, 0] >= min_bound[0]) &
            (xyz[:, 0] <= max_bound[0]) &

            (xyz[:, 1] >= min_bound[1]) &
            (xyz[:, 1] <= max_bound[1]) &

            (xyz[:, 2] >= min_bound[2]) &
            (xyz[:, 2] <= max_bound[2])
        )

        # =====================================================
        # 4. output
        # =====================================================
        self.valid_idx = torch.where(inside)[0]
        self.invalid_idx = torch.where(~inside)[0]

        print("===================================")
        print(f"expand    : {expand}")
        print(f"min_bound : {min_bound.tolist()}")
        print(f"max_bound : {max_bound.tolist()}")
        print(f"valid     : {len(self.valid_idx)}")
        print(f"invalid   : {len(self.invalid_idx)}")
        print("===================================")

    def update(
        self,
        xyz: torch.Tensor,
        mesh: torch.Tensor,
        expand=0.1
    ):
        assert xyz is not None and mesh is not None

        device = xyz.device

        self.xyz = xyz
        self.mesh = mesh

        # =====================================================
        # 1. mesh AABB
        # =====================================================
        min_bound = mesh.min(dim=0)[0]
        max_bound = mesh.max(dim=0)[0]

        # =====================================================
        # 2. expand AABB
        # =====================================================
        min_bound = min_bound - expand
        max_bound = max_bound + expand

        self.min_bound = min_bound
        self.max_bound = max_bound

        # =====================================================
        # 3. xyz inside test
        # =====================================================
        inside = (
            (xyz[:, 0] >= min_bound[0]) &
            (xyz[:, 0] <= max_bound[0]) &

            (xyz[:, 1] >= min_bound[1]) &
            (xyz[:, 1] <= max_bound[1]) &

            (xyz[:, 2] >= min_bound[2]) &
            (xyz[:, 2] <= max_bound[2])
        )

        # =====================================================
        # 4. output
        # =====================================================
        self.valid_idx = torch.where(inside)[0]
        self.invalid_idx = torch.where(~inside)[0]

        print("===================================")
        print(f"expand    : {expand}")
        print(f"min_bound : {min_bound.tolist()}")
        print(f"max_bound : {max_bound.tolist()}")
        print(f"valid     : {len(self.valid_idx)}")
        print(f"invalid   : {len(self.invalid_idx)}")
        print("===================================")