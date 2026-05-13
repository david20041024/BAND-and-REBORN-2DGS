import torch
import torch.nn.functional as F
import sys
sys.path.append('/content/CARD')
from siren import modules


class SDFManager:

    def __init__(self, checkpoint_path):
        self.model = modules.SingleBVPNet(
            type='sine',
            mode='mlp',
            in_features=3,
            out_features=1
        )

        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.cuda()
        self.model.eval()

    # ---------------------------
    # forward SDF
    # ---------------------------
    def sdf(self, coords):
        return self.model({'coords': coords})['model_out']

    # ---------------------------
    # gradient
    # ---------------------------
    def gradient(self, coords):
        coords = coords.clone().detach().requires_grad_(True)

        sdf = self.sdf(coords)

        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]

        return grad

    # ---------------------------
    # normal
    # ---------------------------
    def normal(self, coords):
        grad = self.gradient(coords)
        return F.normalize(grad, dim=-1)

    # ---------------------------
    # curvature (simple mean curvature approx)
    # ---------------------------
    def curvature(self, coords, chunk_size=50000):
        all_H = []
        
        with torch.enable_grad():
            for i in range(0, coords.shape[0], chunk_size):
                chunk_coords = coords[i:i + chunk_size].detach().requires_grad_(True)
                
                sdf_val = self.sdf(chunk_coords)
                
                grad = torch.autograd.grad(
                    outputs=sdf_val,
                    inputs=chunk_coords,
                    grad_outputs=torch.ones_like(sdf_val),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                if grad is None:
                    all_H.append(torch.zeros(chunk_coords.shape[0], device=coords.device))
                    continue

                normal = F.normalize(grad, dim=-1)

                H_chunk = torch.zeros(chunk_coords.shape[0], device=coords.device)
                for j in range(3):
                    dn = torch.autograd.grad(
                        outputs=normal[..., j],
                        inputs=chunk_coords,
                        grad_outputs=torch.ones_like(normal[..., j]),
                        create_graph=False,
                        allow_unused=True 
                    )[0]
                    
                    if dn is not None:
                        H_chunk += dn[..., j]
                
                all_H.append(H_chunk.detach())
                
                del grad, normal, sdf_val, chunk_coords
                
        return torch.cat(all_H, dim=0)