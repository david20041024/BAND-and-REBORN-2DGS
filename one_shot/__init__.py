import torch
import numpy as np
from scene import Scene, GaussianModel
from utils.general_utils import build_rotation
from one_shot.kd import KD
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

        if gaussian_model is not None:
            self.calculate_normal()

        kd = KD(self.xyz, self.mesh_xyz, self.normal, self.gaussians.get_opacity ,max_size=512)
        self.count = kd.clone_count
        self.center = kd.center
        

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
    
    def calculate_normal(self):

        if self.gaussians is None:
            raise ValueError("GaussianModel is None, cannot calculate normal")
        
        R = build_rotation(self.gaussians._rotation)
        
        self.normal = R[:, :, 2]
        xyz = self.gaussians.get_xyz
        indices = torch.arange(xyz.size(0), device=self.device).unsqueeze(1)
        self.xyz = torch.cat([xyz, indices], dim=1)
        return self