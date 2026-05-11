import torch
import torch.nn.functional as F
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
    def curvature(self, coords):

        coords = coords.clone().detach().requires_grad_(True)

        sdf = self.sdf(coords)
        grad = torch.autograd.grad(
            sdf, coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True
        )[0]

        normal = F.normalize(grad, dim=-1)

        H = 0.0
        for i in range(3):
            dn = torch.autograd.grad(
                normal[..., i],
                coords,
                grad_outputs=torch.ones_like(normal[..., i]),
                create_graph=True
            )[0][..., i]
            H += dn

        return H