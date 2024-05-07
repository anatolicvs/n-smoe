import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from scico.functional import Functional
from scico.numpy import Array

class PnP_Denoiser(Functional):

    has_eval = False
    has_prox = True

    def __init__(self, model):
        super().__init__()
        self.denoiser = model
        
        def denoise(x,shape):
            xpinv_t= self.denoiser(x, shape)
            xden = xpinv_t.cpu().squeeze().squeeze().detach().numpy()
            xden = jnp.array(xden)
            return xden
        
        self._denoise = denoise

    def prox(self, x: Array, lam: float = 1.0, phw = 8, stride=1 **kwargs) -> Array:  # type: ignore
      
        x = torch.from_numpy(jax.device_get(x)).unsqueeze(0)
        x_t_p = x.unfold(1, phw, stride).unfold(2, phw, stride)
        x_t_p = F.max_pool3d(x_t_p, kernel_size=1, stride=1)
        x_t_p = x_t_p.view(
                        x_t_p.shape[1] * x_t_p.shape[2], x_t_p.shape[0], x_t_p.shape[3], x_t_p.shape[4]
                    )

        x_t_p = x_t_p.unsqueeze(0)
        x = x.unsqueeze(0)
        x_t_p = x_t_p.cuda()
        with torch.no_grad():
            x = self._denoise(x_t_p, x.shape)
       
        return x