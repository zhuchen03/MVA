# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


@register_optimizer('adam')
class FairseqAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            not getattr(args, 'use_old_adam', False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        if use_fused_adam:
            logger.info('using FusedAdam')
            self._optimizer = fused_adam_cls(params, **self.optimizer_config)
        else:
            self._optimizer = Adam(params, **self.optimizer_config)

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
        parser.add_argument('--no-adamw', default=False, type=distutils.util.strtobool)
        parser.add_argument('--adabound', default=False, type=distutils.util.strtobool)
        parser.add_argument('--gamma', default=1e-3, type=float)
        parser.add_argument('--final-lr', default=0.1, type=float)
        parser.add_argument('--base-lr', default=1e-3, type=float)
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
            'moment_warmup': self.args.moment_warmup,
            'no_adamw': self.args.no_adamw,
            'adabound': self.args.adabound,
            'gamma': self.args.gamma,
            'final_lr': self.args.final_lr,
            'base_lr': self.args.base_lr
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


class Adam(torch.optim.Optimizer):
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
                 weight_decay=0, amsgrad=False, ams_warmup=1, moment_warmup=0, no_adamw=False,
                 adabound=False, final_lr=0.1, gamma=1e-3, base_lr=None):

        if base_lr is None:
            base_lr = lr
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ams_warmup=ams_warmup, moment_warmup=moment_warmup,
                        no_adamw=no_adamw, adabound=adabound, final_lr=final_lr, gamma=gamma, base_lr=base_lr)
        super(Adam, self).__init__(params, defaults)

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

                if group['no_adamw'] and group['weight_decay'] > 0:
                    grad.add_(group['weight_decay'], p_data_fp32)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if state['step'] <= group['moment_warmup']:
                    continue

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                if amsgrad and state['step'] > group['ams_warmup']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** (state['step'] - group['moment_warmup'])
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if not group['no_adamw'] and group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if group['adabound']:
                    final_lr = group['final_lr'] * group['lr'] / group['base_lr']
                    lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                    upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                    step_size = torch.full_like(denom, step_size)
                    # print(torch.max(denom))
                    # pdb.set_trace()
                    step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                    p_data_fp32.add_(-step_size)
                else:
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                # TODO: remove check once pyTorch avoids a copy for this case
                if p.data_ptr() != p_data_fp32.data_ptr():
                    p.data.copy_(p_data_fp32)

        return loss


@register_optimizer('madam')
class FairseqMAdam(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = MAdam(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='betas for Adam optimizer')
        parser.add_argument('--beta2-range', default='(0.5, 0.99)')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--adam-beta0', default=-1, type=float)
        parser.add_argument('--no-adamw', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-lr', default=False, type=distutils.util.strtobool)
        parser.add_argument('--nesterov', default=False, type=distutils.util.strtobool)
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
            'beta1': self.args.beta1,
            'beta2_range': eval(self.args.beta2_range),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'adamw': not self.args.no_adamw,
            'nesterov': self.args.nesterov,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        print("Entered an unhandled average_param function in adam.py")
        pdb.set_trace() # haven't been handled
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


class MAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2_range=(0.9, 0.98), eps=1e-8,
                 weight_decay=0,
                 nesterov=False, adamw=False):
        defaults = dict(lr=lr, beta1=beta1, beta2_range=beta2_range, eps=eps,
                        weight_decay=weight_decay,
                        nesterov=nesterov, adamw=adamw)
        super(MAdam, self).__init__(params, defaults)

    def get_moments(self):
        num_el, mom2, var = 0, 0, 0
        momdiff = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                mom2_all = state['exp_avg_sq'] / state['total_w']
                var_all = (mom2_all - state['exp_avg']**2 / state['total_w']**2).clamp(min=0)
                # Exponential moving average of squared gradient values
                mom2 += torch.sum(mom2_all).item()
                var += torch.sum(var_all).item()
                momdiff += torch.sum(mom2_all - var_all)
                num_el += state['exp_avg_sq'].numel()
        return mom2 / num_el, var / num_el, momdiff/num_el

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    @property
    def update_size(self):
        if getattr(self, "update_size_", None) is not None:
            return None, None, self.update_size_
        else:
            return None, None, None

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.adaptive_beta_ = []
        self.update_size_ = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                if not group['adamw'] and group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p_data_fp32)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_grad'] = torch.zeros_like(p_data_fp32)
                    state['total_w'] = torch.zeros_like(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1 = group['beta1']
                beta2_min, beta2_max = group['beta2_range']

                state['step'] += 1
                total_w = state['total_w']

                if state['step'] <= 1 or beta2_max == beta2_min:
                    # exp_avg.copy_(grad)
                    # exp_avg_sq.copy_(grad*grad)
                    exp_avg.mul_(beta2_max).add_(1-beta2_max, grad)
                    exp_avg_sq.mul_(beta2_max).addcmul_(1-beta2_max, grad, grad)
                    state['total_w'][:] = 1 - beta2_max ** state['step']
                else:
                    # find the beta that maximize the variance
                    # beta is the multiplier for the new grad
                    exp_avg_sq_unbiased = exp_avg_sq / total_w
                    exp_avg_unbiased = exp_avg / total_w
                    moment_diff = exp_avg_sq_unbiased - exp_avg_unbiased ** 2
                    mean_diff_sq = (grad - exp_avg_unbiased) ** 2
                    w_diff_diff = mean_diff_sq.add(-moment_diff).mul_(total_w)
                    denominator = w_diff_diff.add(mean_diff_sq).add_(moment_diff).add_(1e-16)

                    adv_beta = w_diff_diff.div_(denominator)
                    # clamp the range
                    # self.adaptive_beta.append(torch.max(adv_beta).item())

                    adv_beta.clamp_(min=1-beta2_max, max=1-beta2_min)

                    adv_beta_comp = 1 - adv_beta
                    exp_avg.mul_(adv_beta_comp).add_(adv_beta * grad)
                    exp_avg_sq.mul_(adv_beta_comp).add_(adv_beta * grad * grad)

                    state['total_w'] = state['total_w'] * adv_beta_comp + adv_beta

                denom = (exp_avg_sq / state['total_w']).sqrt_().add_(group['eps'] )

                state['exp_avg_grad'].mul_(beta1).add_(grad, alpha=(1 - beta1))
                bias_correction0 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction0

                if group['nesterov']:
                    exp_avg_grad = state['exp_avg_grad'] * beta1 + (1-beta1) * grad
                else:
                    exp_avg_grad = state['exp_avg_grad']

                if group['adamw'] and group['weight_decay'] > 0:
                    p_data_fp32.add_(- group['lr'] * group['weight_decay'], p_data_fp32)

                if True:
                    # need to return update size
                    update = -step_size * exp_avg_grad.div(denom)
                    self.update_size_ = update.abs().mean().item()
                    p_data_fp32.add_(update)
                else:
                    p_data_fp32.addcdiv_(-step_size, exp_avg_grad, denom)

                p.data.copy_(p_data_fp32)

        return loss

