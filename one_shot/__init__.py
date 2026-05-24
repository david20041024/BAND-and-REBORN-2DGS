import torch
import numpy as np
from scene import Scene, GaussianModel
from utils.general_utils import build_rotation
from one_shot.voxel import VOXEL
class GaussianModelProcessor:
    def __init__(self, gaussian_model, xyz_mesh=None, xyz_nonboundary=None, expand=0.2, size=0.02):
        """
        gaussian_model: GaussianModel instance
        xyz_file: path to .xyz file (x y z nx ny nz)
        """
        self.gaussians = gaussian_model
        self.xyz_mesh = None
        self.xyz_nonboundary = None
        self.normal = None
        self.mask = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if xyz_mesh is not None:
            self.load_xyz(xyz_mesh, mesh=True)

        if xyz_nonboundary is not None:
            self.load_xyz(xyz_nonboundary, mesh=False)

        self.voxel = VOXEL(self.xyz_mesh, self.xyz_nonboundary, expand=expand, size=size)
        

    # --------------------------------------------------
    # 1. Load xyz file -> torch tensor
    # --------------------------------------------------
    def load_xyz(self, path, mesh=False):
        """
        Expect format:
        x y z nx ny nz
        """
        data = np.loadtxt(path)  # (N, 6)

        xyz = data[:, 0:3]
        normal = data[:, 3:6]

        if mesh == True:
            self.xyz_mesh = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        else:
            self.xyz_nonboundary = torch.tensor(xyz, dtype=torch.float32, device=self.device)

        return self
    
    def find_boundary_mask(self):
        self.voxel.find_boundary(self.gaussians)
        aim = self.voxel.aim
        num_gaussians = len(self.gaussians.get_xyz)
        bg_mask = torch.zeros(num_gaussians, dtype=torch.bool, device="cuda")
        bg_mask[aim] = True
        self.mask = bg_mask
        return self

    