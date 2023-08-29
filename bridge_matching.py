from tqdm import tqdm as tqdm
import torch
import torch.nn as nn

class BridgeNoiseScheduler:
    def __init__(self) -> None:
        pass

    def beta(self, t) -> torch.Tensor:
        raise NotImplementedError('Abstract class')

    def sigma(self, t) -> torch.Tensor:
        raise NotImplementedError('Abstract class')
    
    def sigma_overlined(self, t) -> torch.Tensor:
        raise NotImplementedError('Abstract class')
    
class LinearBridgeNoiseScheduler(BridgeNoiseScheduler):
    def __init__(self, max_beta=1.5e-4) -> None:
        super().__init__()
        self.max_beta = 2 * max_beta

    def beta(self, t) -> torch.Tensor:
        return (0.5 - torch.abs(t - 0.5)) * self.max_beta

    def _intergrate_beta(self, t_1, t_2):
        ret_val = torch.min(t_2, torch.tensor(0.5)) ** 2 / 2 - torch.min(t_1, torch.tensor(0.5)) ** 2 / 2
        ret_val -= (1 - torch.min(t_2, torch.tensor(1))) ** 2 / 2 - (1 - torch.min(t_1, torch.tensor(1))) ** 2 / 2
        return ret_val * self.max_beta
    
    def sigma(self, t) -> torch.Tensor:
        return torch.sqrt(self._intergrate_beta(torch.zeros_like(t), t))
    
    def sigma_overlined(self, t) -> torch.Tensor:
        return torch.sqrt(self._intergrate_beta(t, torch.ones_like(t)))

class BridgeMathcing(nn.Module):
    def __init__(self, unet, scheduler: BridgeNoiseScheduler):
        super().__init__()
        self.vector_net = unet
        self.euler_dt = 0.01
        self.sch = scheduler
        
    def forward(self, x_0):
        # solve forward ODE via Euler or torchdiffeq solver
        x_t = x_0
        
        t_range = tqdm(torch.arange(0, 1, step=self.euler_dt))
        
        for t in t_range:
            eps_noise = torch.randn_like(x_t)
            beta_t = self.sch.beta(t)
            x_t = x_t + self.vector_net(x_t, t) * self.euler_dt + torch.sqrt(self.euler_dt * beta_t) * eps_noise
            
        return x_t

    @torch.no_grad()
    def sample(self, x_0, pbar=True):
        x_t = x_0
        
        if pbar:
            t_range = tqdm(torch.arange(0, 1, step=self.euler_dt))
        else:
            t_range = torch.arange(0, 1, step=self.euler_dt)
        
        for t in t_range:
            eps_noise = torch.randn_like(x_t)
            beta_t = self.sch.beta(t)

            sigma_overlined_t = self.sch.sigma_overlined(t)
            x_t = x_t + (beta_t  / (sigma_overlined_t ** 2)) * self.vector_net(x_t, t) * self.euler_dt + torch.sqrt(self.euler_dt * beta_t) * eps_noise
            
        return x_t

    def sample_x_t(self, x_0, x_1, t):
        sigma_t_sq = self.sch.sigma(t) ** 2
        sigma_overlined_t_sq = self.sch.sigma_overlined(t) ** 2

        coef_0, coef_1 = sigma_overlined_t_sq / (sigma_t_sq + sigma_overlined_t_sq), sigma_t_sq / (sigma_t_sq + sigma_overlined_t_sq)

        std_t = torch.sqrt(sigma_t_sq * sigma_overlined_t_sq / (sigma_overlined_t_sq + sigma_t_sq))

        x_t = coef_1.reshape([-1, 1, 1, 1]) * x_1 + coef_0.reshape([-1, 1, 1, 1]) * x_0 + torch.randn_like(x_0) * std_t.reshape([-1, 1, 1, 1])
        return x_t
    
    def step(self, x_0, x_1, t):
        t = t.reshape([-1, 1, 1, 1])
        x_t = self.sample_x_t(x_0, x_1, t)
        x_t_hat = self.vector_net(x_t, t)
        return self.loss(x_t_hat, x_1, x_t, t).mean()
    
    def loss(self, x_t_hat, x_1, x_t, t):
        beta_t = self.sch.beta(t)
        sigma_overlined_t = self.sch.sigma_overlined(t)
        return torch.norm((x_t_hat - (x_1 - x_t)).reshape([x_1.shape[0], -1]), dim=-1)
