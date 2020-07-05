import logging
import math
import types

import torch
import torch.optim
import torch.distributed as dist

from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class

import distutils.util
import pdb

logger = logging.getLogger(__name__)


@register_optimizer('bender')
class FairseqBender(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = Bender(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--amsgrad', default=False, type=distutils.util.strtobool)
        parser.add_argument('--ams-warmup', default=1, type=int, metavar='WD',
                            help='adam warmup iterations')
        parser.add_argument('--moment-warmup', default=0, type=int)
        # Maintain backward compatibility with old checkpoints that have stored
        # optimizer state as fairseq.optim.adam.Adam.
        parser.add_argument(
            "--use-old-adam",
            action='store_true',
            default=False,
            help="Use fairseq.optim.adam.Adam",
        )
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'amsgrad': self.args.amsgrad,
            'ams_warmup': self.args.ams_warmup,
            'moment_warmup': self.args.moment_warmup
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


class Bender(torch.optim.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, beta_lip=0.9, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, ams_warmup=1, moment_warmup=0):
        defaults = dict(lr=lr, beta_lip=beta_lip, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ams_warmup=ams_warmup, moment_warmup=moment_warmup)
        super(Bender, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_step'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_lip'] = 0 #torch.zeros_like(p_data_fp32)
                    state['last_x'] = torch.zeros_like(p_data_fp32)
                    state['last_gx'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg_step'] = state['exp_avg_step'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    # state['exp_avg_lip'] = state['exp_avg_lip'].type_as(p_data_fp32)
                    state['last_x'] = state['last_x'].type_as(p_data_fp32)
                    state['last_gx'] = state['last_gx'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg_step, exp_avg_lip = state['exp_avg_step'], state['exp_avg_lip']
                exp_avg_sq = state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                beta_lip = group['beta_lip']

                state['step'] += 1
                self.last_lr = group['lr']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg_step.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if state['step'] <= group['moment_warmup']:
                    state['last_x'].copy_(p_data_fp32)
                    state['last_gx'].copy_(grad)
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    step_size = group['lr'] / bias_correction1
                    p_data_fp32.add_(-step_size, exp_avg_step)
                else:
                    this_lip = torch.norm(grad - state['last_gx']).item()
                    this_lip = this_lip / torch.norm(p_data_fp32 - state['last_x']).add(group['eps']).item() * 2

                    exp_avg_lip = exp_avg_lip * beta_lip + (1 - beta_lip) * this_lip
                    state['last_x'].copy_(p_data_fp32)
                    state['last_gx'].copy_(grad)

                    denom = exp_avg_sq.sqrt() + group['eps']

                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    lip_correction = 1 - beta_lip ** (state['step'] - group['moment_warmup'])
                    step_size = group['lr'] * (math.sqrt(bias_correction2) / bias_correction1) * lip_correction
                    # p_data_fp32.addcdiv_(-step_size, exp_avg_step, denom)
                    p_data_fp32.addcdiv_(-step_size / exp_avg_lip, exp_avg_step, denom)
                    # print("step size: {:.2e}".format(step_size  / exp_avg_lip))
                    # pdb.set_trace()

                # TODO: remove check once pyTorch avoids a copy for this case
                if p.data_ptr() != p_data_fp32.data_ptr():
                    p.data.copy_(p_data_fp32)

        return loss


class Bender_v2(torch.optim.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, ams_warmup=1, moment_warmup=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ams_warmup=ams_warmup, moment_warmup=moment_warmup)
        super(Bender_v2, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_step'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_lip2'] = torch.zeros_like(p_data_fp32)
                    state['last_x'] = torch.zeros_like(p_data_fp32)
                    state['last_gx'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg_step'] = state['exp_avg_step'].type_as(p_data_fp32)
                    state['exp_avg_lip2'] = state['exp_avg_lip2'].type_as(p_data_fp32)
                    state['last_x'] = state['last_x'].type_as(p_data_fp32)
                    state['last_gx'] = state['last_gx'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg_step, exp_avg_lip2 = state['exp_avg_step'], state['exp_avg_lip2']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']

                exp_avg_step.mul_(beta1).add_(1 - beta1, grad)

                if state['step'] > 1:
                    this_lip = (grad - state['last_gx']) ** 2
                    this_lip = this_lip / torch.sum((p_data_fp32 - state['last_x']) ** 2).add(group['eps'])
                    exp_avg_lip2.mul_(beta2).add_(1 - beta2, this_lip)

                state['last_x'].copy_(p_data_fp32)
                state['last_gx'].copy_(grad)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if state['step'] <= group['moment_warmup']:
                    step_size = group['lr'] / bias_correction1
                    p_data_fp32.add_(-step_size, exp_avg_step)
                else:
                    denom = exp_avg_lip2.sqrt()
                    denom.add_(group['eps'])

                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    step_size = group['lr'] * (math.sqrt(bias_correction2) / bias_correction1)
                    p_data_fp32.addcdiv_(-step_size, exp_avg_step, denom)

                    print(torch.max(exp_avg_lip2).item())
                    # pdb.set_trace()

                # TODO: remove check once pyTorch avoids a copy for this case
                if p.data_ptr() != p_data_fp32.data_ptr():
                    p.data.copy_(p_data_fp32)

        return loss

