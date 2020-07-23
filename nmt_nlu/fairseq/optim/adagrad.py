# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim

from . import FairseqOptimizer, register_optimizer
import math
import pdb


@register_optimizer('adagrad')
class FairseqAdagrad(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        # self._optimizer = torch.optim.Adagrad(params, **self.optimizer_config)
        self._optimizer = Adagrad(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--no-adamw', default=False, action="store_true")
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
            'weight_decay': self.args.weight_decay,
            'no_adamw': self.args.no_adamw
        }

    @property
    def supports_flat_params(self):
        return True


class Adagrad(torch.optim.Optimizer):
    """Implements Adagrad algorithm.
    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, no_adamw=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value, no_adamw=no_adamw)
        super(Adagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p, initial_accumulator_value, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                if group['no_adamw'] and group['weight_decay'] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(p, alpha=group['weight_decay'])

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if not group['no_adamw'] and group['weight_decay'] != 0:
                    p.data.add_(p, alpha=-group['lr'] * group['weight_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(group['eps'])
                    p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                else:
                    state['sum'].addcmul_(grad, grad, value=1)
                    std = state['sum'].sqrt().add_(group['eps'])
                    p.addcdiv_(grad, std, value=-clr)

        return loss


@register_optimizer('adagrad_m')
class FairseqAdagradM(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        # self._optimizer = torch.optim.Adagrad(params, **self.optimizer_config)
        self._optimizer = AdagradM(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--no-adamw', default=False, action="store_true")
        parser.add_argument('--beta1', default=0.9, type=float)
        parser.add_argument('--beta2', default=0.997, type=float)
        parser.add_argument('--reset-freq', default=4000, type=int)
        parser.add_argument('--adam-eps', default=1e-8, type=float)
        parser.add_argument('--need-lr', default=False, action="store_true")
        parser.add_argument('--need-update-size', default=False, action="store_true")
        parser.add_argument('--use-var', default=False, action="store_true")
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
            'weight_decay': self.args.weight_decay,
            'no_adamw': self.args.no_adamw,
            'beta1': self.args.beta1,
            'beta2': self.args.beta2,
            'reset_freq': self.args.reset_freq,
            'eps': self.args.adam_eps,
            'need_lr': self.args.need_lr,
            'need_update_size': self.args.need_update_size,
            'use_var': self.args.use_var
        }

    @property
    def supports_flat_params(self):
        return True


class AdagradM(torch.optim.Optimizer):
    """Implements Adagrad algorithm.
    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, weight_decay=0, beta1=0.9, beta2=0.997, eps=1e-8, reset_freq=4000, no_adamw=False,
                 need_lr=False, need_update_size=False, use_var=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                        reset_freq=reset_freq, no_adamw=no_adamw, use_var=use_var)
        super(AdagradM, self).__init__(params, defaults)
        self.need_update_size = need_update_size
        self.need_lr = need_lr

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['avg_counter'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['uniform_avg_sq'] = torch.zeros_like(p)

                if group['use_var']:
                    state['uniform_avg'] = torch.zeros_like(p)


    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @property
    def adaptive_lrs(self):
        if self.need_lr:
            if isinstance(self.adaptive_lrs_, float):
                return None, None, self.adaptive_lrs_
            else:
                return None, None, torch.median(self.adaptive_lrs_).item()
        else:
            return None, None, None

    @property
    def update_size(self):
        if self.need_update_size:
            return None, None, torch.median(
                self.update_size_).item()
        else:
            return None, None, None

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                beta1, beta2 = group['beta1'], group['beta2']

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                if group['no_adamw'] and group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if not group['no_adamw'] and group['weight_decay'] != 0:
                    p.data.add_(p, alpha=-group['lr'] * group['weight_decay'])

                state['exp_avg'].mul_(beta1).add_(1 - beta1, grad)
                grad_sq = grad * grad
                state['exp_avg_sq'].mul_(beta2).add_(1 - beta2, grad_sq)

                exp_avg = state['exp_avg']

                # updates for the uniform mean
                if state['step'] == 1:
                    state['uniform_avg_sq'].copy_(grad_sq)
                    if group['use_var']:
                        state['uniform_avg'].copy_(grad)
                state['avg_counter'] += 1
                beta3 = 1 - 1. / (state['avg_counter'] + 1)
                state['uniform_avg_sq'].mul_(beta3).add_(1 - beta3, grad_sq)
                if group['use_var']:
                    state['uniform_avg'].mul_(beta3).add_(1 - beta3, grad)
                    uniform_denom = state['uniform_avg_sq'] - state['uniform_avg'] ** 2
                else:
                    uniform_denom = state['uniform_avg_sq']

                if state['avg_counter'] % group['reset_freq'] == 0:
                    state['avg_counter'] = 0

                # get the denom here
                denom = torch.max(state['exp_avg_sq']/(1 - beta2 ** state['step']), uniform_denom)
                # denom = state['exp_avg_sq']/(1 - beta2 ** state['step'])
                denom.sqrt_().add_(group['eps'])

                step_size = group['lr'] / (1 - beta1 ** state['step'])

                if self.need_lr:
                    self.adaptive_lrs_ = step_size / denom

                if self.need_update_size:
                    self.update_size_ = -step_size * exp_avg / denom
                    p.add_(self.update_size_)
                    self.update_size_ = self.update_size_.abs()
                else:
                    p.addcdiv_(-step_size, exp_avg, denom)

        return loss

@register_optimizer('r_adagrad_m')
class FairseqRAdagradM(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        # self._optimizer = torch.optim.Adagrad(params, **self.optimizer_config)
        self._optimizer = RAdagradM(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--no-adamw', default=False, action="store_true")
        parser.add_argument('--beta1', default=0.9, type=float)
        parser.add_argument('--beta2', default=0.997, type=float)
        parser.add_argument('--reset-freq', default=4000, type=int)
        parser.add_argument('--adam-eps', default=1e-8, type=float)
        parser.add_argument('--need-lr', default=False, action="store_true")
        parser.add_argument('--need-update-size', default=False, action="store_true")
        parser.add_argument('--use-var', default=False, action="store_true")
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
            'weight_decay': self.args.weight_decay,
            'no_adamw': self.args.no_adamw,
            'beta1': self.args.beta1,
            'beta2': self.args.beta2,
            'reset_freq': self.args.reset_freq,
            'eps': self.args.adam_eps,
            'need_lr': self.args.need_lr,
            'need_update_size': self.args.need_update_size,
            'use_var': self.args.use_var
        }

    @property
    def supports_flat_params(self):
        return True


class RAdagradM(torch.optim.Optimizer):
    """Implements Adagrad algorithm.
    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, weight_decay=0, beta1=0.9, beta2=0.997, eps=1e-8, reset_freq=4000, no_adamw=False,
                 need_lr=False, need_update_size=False, use_var=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                        reset_freq=reset_freq, no_adamw=no_adamw, use_var=use_var)
        super(RAdagradM, self).__init__(params, defaults)
        self.need_update_size = need_update_size
        self.need_lr = need_lr

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['avg_counter'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['uniform_avg_sq'] = torch.zeros_like(p)

                if group['use_var']:
                    state['uniform_avg'] = torch.zeros_like(p)


    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @property
    def adaptive_lrs(self):
        if self.need_lr:
            if isinstance(self.adaptive_lrs_, float):
                return None, None, self.adaptive_lrs_
            else:
                return None, None, torch.median(self.adaptive_lrs_).item()
        else:
            return None, None, None

    @property
    def update_size(self):
        if self.need_update_size:
            return None, None, torch.median(
                self.update_size_).item()
        else:
            return None, None, None

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                beta1, beta2 = group['beta1'], group['beta2']

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                if group['no_adamw'] and group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if not group['no_adamw'] and group['weight_decay'] != 0:
                    p.data.add_(p, alpha=-group['lr'] * group['weight_decay'])

                state['exp_avg'].mul_(beta1).add_(1 - beta1, grad)
                grad_sq = grad * grad
                state['exp_avg_sq'].mul_(beta2).add_(1 - beta2, grad_sq)

                exp_avg = state['exp_avg']

                # updates for the uniform mean
                if state['step'] == 1:
                    state['uniform_avg_sq'].copy_(grad_sq)
                    if group['use_var']:
                        state['uniform_avg'].copy_(grad)
                state['avg_counter'] += 1
                beta3 = 1 - 1. / (state['avg_counter'] + 1)
                state['uniform_avg_sq'].mul_(beta3).add_(1 - beta3, grad_sq)
                if group['use_var']:
                    state['uniform_avg'].mul_(beta3).add_(1 - beta3, grad)
                    uniform_denom = state['uniform_avg_sq'] - state['uniform_avg'] ** 2
                else:
                    uniform_denom = state['uniform_avg_sq']

                if state['avg_counter'] % group['reset_freq'] == 0:
                    state['avg_counter'] = 0

                beta2_pesudo = 0.9995
                beta2_t = beta2_pesudo ** state['step']
                N_sma_max = 2 / (1 - beta2_pesudo) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt((N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) * (N_sma_max) / N_sma / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    # get the denom here
                    denom = torch.max(state['exp_avg_sq']/(1 - beta2 ** state['step']), uniform_denom)
                    # denom = state['exp_avg_sq']/(1 - beta2 ** state['step'])
                    denom.sqrt_().add_(group['eps'])

                    if self.need_lr:
                        self.adaptive_lrs_ = step_size / denom

                    if self.need_update_size:
                        self.update_size_ = -step_size * exp_avg / denom
                        p.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    if self.need_update_size:
                        self.update_size_ = -step_size * exp_avg
                        p.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p.add_(-step_size, exp_avg)
                    if self.need_lr:
                        self.adaptive_lrs_ = step_size

        return loss
