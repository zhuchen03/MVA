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


@register_optimizer('adam_gt')
class FairseqAdamGT(FairseqOptimizer):
    """Adam optimizer with gradient transportation.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = AdamGT(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--gt-beta', default=0.9, type=float)
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
            'moment_warmup': self.args.moment_warmup,
            'gt_beta': self.args.gt_beta
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


class AdamGT(torch.optim.Optimizer):
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

    def __init__(self, params, lr=1e-3, gt_beta=0.9, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, ams_warmup=1, moment_warmup=0):
        defaults = dict(lr=lr, gt_beta=gt_beta, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ams_warmup=ams_warmup, moment_warmup=moment_warmup)
        super(AdamGT, self).__init__(params, defaults)

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
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['pipe'] = p_data_fp32.clone()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    state['pipe'] = state['pipe'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                pipe = state['pipe']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                gt_beta = group['gt_beta']
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
                step_size = - group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                update_step = step_size * exp_avg / denom
                p_data_fp32 = pipe + update_step / (1 - gt_beta)
                pipe.add_(update_step)

                # if group['weight_decay'] != 0:
                #     p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # TODO: remove check once pyTorch avoids a copy for this case
                if p.data_ptr() != p_data_fp32.data_ptr():
                    p.data.copy_(p_data_fp32)

        return loss


@register_optimizer('adam_bb')
class FairseqAdamBB(FairseqOptimizer):
    """Adam optimizer with gradient transportation.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = AdamBB(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--gt-beta', default=0.9, type=float)
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
            'moment_warmup': self.args.moment_warmup,
            'gt_beta': self.args.gt_beta
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


class AdamBB(torch.optim.Optimizer):
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

    def __init__(self, params, lr=1e-3, gt_beta=0.9, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, ams_warmup=1, moment_warmup=0):
        defaults = dict(lr=lr, gt_beta=gt_beta, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ams_warmup=ams_warmup, moment_warmup=moment_warmup)
        super(AdamBB, self).__init__(params, defaults)

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
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['pipe'] = p_data_fp32.clone()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    state['pipe'] = state['pipe'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                pipe = state['pipe']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                gt_beta = group['gt_beta']
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

                if group['lr'] > 0:
                    update_step = exp_avg / denom
                    beta_ = (math.sqrt(bias_correction2) * bias_correction1 / group['lr']) * (grad * update_step) / (exp_avg * exp_avg).add(group['eps'])
                    beta = beta_.clamp(min=-1, max=1)
                    print(beta_, beta)
                    pdb.set_trace()

                    p_data_fp32 = pipe - update_step * (beta * step_size)
                    pipe.add_(-step_size, update_step)

                # if group['weight_decay'] != 0:
                #     p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # TODO: remove check once pyTorch avoids a copy for this case
                if p.data_ptr() != p_data_fp32.data_ptr():
                    p.data.copy_(p_data_fp32)

        return loss


@register_optimizer('adam_igt')
class FairseqAdamIGT(FairseqOptimizer):
    """Adam optimizer with gradient transportation.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = AdamIGT(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--gt-delta', default=1, type=float)
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
    def supports_flat_params(self):
        return True

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
            'delta': self.args.gt_delta
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


class AdamIGT(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 delta=1.0,
                 amsgrad=False):
        defaults = {
            'delta': delta,
            'num_steps': 0,
            'train': True,
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super(AdamIGT, self).__init__(params, defaults)


    def compute_update(self, p, param_state, group):
        exp_avg = param_state['exp_avg']
        exp_avg_sq = param_state['exp_avg_sq']
        beta1, beta2 = group['betas']
        lr = group['lr']
        grad = p.grad.data

        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        denom = exp_avg_sq.sqrt().add_(group['eps'])

        # NOTE: the + 1 is because IGT and Adam don't count steps the same way.
        bias_correction1 = 1 - beta1 ** (group['num_steps'] + 1)
        bias_correction2 = 1 - beta2 ** (group['num_steps'] + 1)
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        update = -step_size * (exp_avg / denom)
        return update

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            delta = group['delta']
            num_steps = group['num_steps']
            gamma = (num_steps) / (num_steps + delta)
            future_gamma = (num_steps + 1) / (num_steps + 1 + delta)
            future_transport = future_gamma / (1.0 - future_gamma)
            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data
                param_state = self.state[p]

                # Compute the IGT estimate
                if num_steps == 0:
                    param_state['igt_estimate'] = torch.zeros_like(d_p)
                    param_state['igt_estimate'].add_(d_p)
                    param_state['true_p'] = torch.zeros_like(p.data)
                    param_state['true_p'].add_(p.data)
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    true_p = param_state['true_p']
                else:
                    igt_estimate = param_state['igt_estimate']
                    true_p = param_state['true_p']
                    igt_estimate.mul_(gamma).add_((1.0 - gamma), d_p)
                    # Sets gradients to the IGT estimate
                    d_p.copy_(igt_estimate)
                    p.data.copy_(true_p)  # Revert to true params

                # Take the step according to opt
                update = self.compute_update(p, param_state, group)

                # Transport to the next parameter point
                true_p.copy_(p.data).add_(update)
                p.data.add_(1.0 + future_transport, update)
            group['num_steps'] += 1
        return loss
