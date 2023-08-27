from tqdm import tqdm as tqdm
import torch
import torch.nn as nn

class FlowMathcing(nn.Module):
    def __init__(self, unet):
        super().__init__()
        # CNN + positional embeding time conditioning
        # can use NN from MNIST DDPM
        self.vector_net = unet
        self.euler_dt = 0.01
        
    
    def forward(self, x_0):
        # solve forward ODE via Euler or torchdiffeq solver
        x_t = x_0
        
        t_range = tqdm(torch.arange(0, 1, step=self.euler_dt))
        
        for t in t_range:
            x_t = x_t + self.vector_net(x_t, t) * self.euler_dt
            
        return x_t

    @torch.no_grad()
    def sample(self, x_0, pbar=True):
        x_t = x_0
        
        if pbar:
            t_range = tqdm(torch.arange(0, 1, step=self.euler_dt))
        else:
            t_range = torch.arange(0, 1, step=self.euler_dt)
        
        for t in t_range:
            x_t = x_t + self.vector_net(x_t, t) * self.euler_dt
            
        return x_t
    
    def step(self, x_0, x_1, t):
        t = t.reshape([-1, 1, 1, 1])
        x_t = t * x_1 + (1 - t) * x_0
        x_t_hat = self.vector_net(x_t, t)
        return self.loss(x_t_hat, x_0, x_1, t).mean()
    
    def loss(self, x_t_hat, x_0, x_1, t):
        return torch.norm((x_t_hat - (x_1 - x_0)).reshape([x_0.shape[0], -1]), dim=-1)