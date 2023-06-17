import torch
import numpy as np

# class BaseModule(torch.nn.Module):
#     def __init__(self):
#         super(BaseModule, self).__init__()

#     @property
#     def nparams(self):
#         num_params = 0
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 num_params += np.prod(param.detach().cpu().numpy().shape)
#         return num_params


#     def relocate_input(self, x: list):
#         device = next(self.parameters()).device
#         for i in range(len(x)):
#             if isinstance(x[i], torch.Tensor) and x[i].device != device:
#                 x[i] = x[i].to(device)
#         return x


class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class Block(torch.nn.Module):
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim, 7, 
                     padding=3), torch.nn.GroupNorm(groups, dim), Mish())

    def forward(self, x):
        output = self.block(x)
        return output


class ResnetBlock(torch.nn.Module):
    def __init__(self, dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.block1 = Block(dim, groups=groups)
        self.block2 = Block(dim, groups=groups)
        self.res = torch.nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        output = self.res(x) + h
        return output


class PostNet(torch.nn.Module):
    def __init__(self, dim, groups=8):
        super(PostNet, self).__init__()
        self.init_conv = torch.nn.Conv2d(1, dim, 1)
        self.res_block = ResnetBlock(dim, groups=groups)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.init_conv(x)
        x = self.res_block(x)
        output = self.final_conv(x)
        return output.squeeze(1)