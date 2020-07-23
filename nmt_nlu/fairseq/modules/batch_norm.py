import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import init
import math
import pdb


class RectifyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, betas=(0.9, 0.99)):
        super(RectifyBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.betas = betas
        # whether to detach the normalization factors from the graph during training

        if self.affine:
            self.weight = Parameter(torch.Tensor(1, 1, num_features))
            self.bias = Parameter(torch.Tensor(1, 1, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(1, 1, num_features))
        self.register_buffer('running_var', torch.ones(1, 1, num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
        self.reset_graph()

    def reset_graph(self):
        self.diff_mean = torch.zeros_like(self.running_mean).detach().clone()
        self.diff_var = torch.zeros_like(self.running_var).detach().clone()

    def forward(self, x, mask=None):
        # the size of x should be batch_size x seq_len x dim
        # mask is used to exclude zero-padding

        if isinstance(x, torch.cuda.HalfTensor):
            x = x.float()
            converted = True
        else:
            converted = False

        x = x.contiguous()
        dim = x.size(2)

        if self.training:
            if mask is not None:
                x_effective = torch.masked_select(x, mask).contiguous().view(-1, dim)
            else:
                x_effective = x.contiguous().view(-1, dim)
            x_batch_mean = torch.mean(x_effective, dim=0)
            x_batch_var = torch.mean((x_effective - x_batch_mean)**2, dim=0)

            self.num_batches_tracked.add_(1)

            beta1, beta2 = self.betas
            bcc1, bcc2 = 1 - beta1 ** self.num_batches_tracked, 1 - beta2 ** self.num_batches_tracked

            self.diff_mean = self.diff_mean.detach().to(x_batch_mean) * beta1 + x_batch_mean.view(1, 1, -1) * (1 - beta1)
            self.diff_var = self.diff_var.detach().to(x_batch_var) * beta2 + x_batch_var.view(1, 1, -1) * (1 - beta2)

            beta2_t = beta2 ** self.num_batches_tracked.item()
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * self.num_batches_tracked.item() * beta2_t / (1 - beta2_t)

            if N_sma >= 5:
                rect = math.sqrt((N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) * (N_sma_max) / N_sma / (N_sma_max - 2))
                rect *= math.sqrt(bcc2)
            else:
                rect = math.sqrt(bcc2)

            x = rect * (x - self.diff_mean / bcc1) / torch.sqrt(self.diff_var + self.eps)

            self.running_mean.copy_(self.diff_mean.data)
            self.running_var.copy_(self.diff_var.data)

        else:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        if self.affine:
            x = x * self.weight + self.bias

        if converted:
            return x.half()
        else:
            return x


