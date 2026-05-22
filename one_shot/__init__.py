import torch
import numpy as np
from scene import Scene, GaussianModel
from utils.general_utils import build_rotation
from one_shot.voxel import VOXEL
class GaussianModelProcessor:
    def __init__(self, gaussian_model, xyz_file=None):
        """
        gaussian_model: GaussianModel instance
        xyz_file: path to .xyz file (x y z nx ny nz)
        """
        self.gaussians = gaussian_model
        self.mesh_xyz = None
        self.xyz = None
        self.normal = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if xyz_file is not None:
            self.load_xyz(xyz_file)

        voxel = VOXEL(self.xyz, self.mesh_xyz, voxel_size=0.004)
        self.prune_list = voxel.invalid_idx
        

    # --------------------------------------------------
    # 1. Load xyz file -> torch tensor
    # --------------------------------------------------
    def load_xyz(self, path):
        """
        Expect format:
        x y z nx ny nz
        """
        data = np.loadtxt(path)  # (N, 6)

        xyz = data[:, 0:3]
        normal = data[:, 3:6]

        self.mesh_xyz = torch.tensor(xyz, dtype=torch.float32, device=self.device)

        return self