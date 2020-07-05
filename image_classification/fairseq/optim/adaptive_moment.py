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


@register_optimizer('radam_adv')
class FairseqRAdamAdv(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = RAdam_adv_v2(params, **self.optimizer_config)

    @property
    def supports_flat_params(self):
        return True

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='betas for Adam optimizer')
        parser.add_argument('--beta2-range', default='(0.5, 0.99)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--amsgrad', default=False, type=distutils.util.strtobool)
        parser.add_argument('--use-sgd', default=False, type=distutils.util.strtobool)
        parser.add_argument('--adam-step', default=False, type=distutils.util.strtobool)
        parser.add_argument('--total-var', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-lr', default=False, type=distutils.util.strtobool)
        parser.add_argument('--nesterov', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-update-size', default=False, type=distutils.util.strtobool)
        parser.add_argument('--no-adamw', default=False, type=distutils.util.strtobool)
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
            'amsgrad': self.args.amsgrad,
            'use_sgd': self.args.use_sgd,
            'adam_step': self.args.adam_step,
            'total_var': self.args.total_var,
            'need_lr': self.args.need_lr,
            'nesterov': self.args.nesterov,
            "need_update_size": self.args.need_update_size,
            "no_adamw": self.args.no_adamw
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


class RAdam_adv_v2(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2_range=(0.5, 0.99), eps=1e-8,
                 weight_decay=0, amsgrad=False, use_sgd=False,
                 adam_step=False, total_var=False, need_lr=False, nesterov=False, need_update_size=False, no_adamw=False):
        defaults = dict(lr=lr, beta1=beta1, beta2_range=beta2_range, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, use_sgd=use_sgd,
                        adam_step=adam_step, total_var=total_var, nesterov=nesterov, no_adamw=no_adamw)

        self.this_ratio = 1
        self.need_lr = need_lr
        self.need_update_size = need_update_size
        # self.cache_len = cache_len
        super(RAdam_adv_v2, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def adaptive_beta(self):
        if len(self.adaptive_beta_) > 0:
            return sum(self.adaptive_beta_) / len(self.adaptive_beta_)
        else:
            return None

    @property
    def adaptive_lrs(self):
        if self.need_lr:
            if isinstance(self.adaptive_lrs_, float):
                return self.adaptive_lrs_, self.adaptive_lrs_, self.adaptive_lrs_
            else:
                return torch.min(self.adaptive_lrs_[self.adaptive_lrs_ > 0]).item(), torch.max(self.adaptive_lrs_).item(), torch.median(
                    self.adaptive_lrs_).item()
        else:
            return None, None, None

    @property
    def update_size(self):
        if self.need_update_size:
            return torch.min(self.update_size_).item(), torch.max(
                self.update_size_).item(), torch.median(
                self.update_size_).item()
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

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                if group['no_adamw'] and group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p_data_fp32)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = torch.zeros_like(p_data_fp32)
                    state['total_w'] = torch.zeros_like(p_data_fp32) if not group['total_var'] else torch.zeros([]).to(
                        p_data_fp32)
                else:
                    state['ema_grad'] = state['ema_grad'].type_as(p_data_fp32)
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = state['max_var'].type_as(p_data_fp32)

                    state['total_w'] = state['total_w'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                ema_grad = state['ema_grad']
                beta1 = group['beta1']
                beta2_min, beta2_max = group['beta2_range']
                state['step'] += 1

                if state['step'] == 1:
                    exp_avg.mul_(beta2_max).add_(1 - beta2_max, grad)
                    exp_avg_sq.mul_(beta2_max).addcmul_(1 - beta2_max, grad, grad)
                    state['total_w'] += 1 - beta2_max
                else:
                    # find the beta that maximize the variance
                    # beta is the multiplier for the new grad
                    moment_diff = exp_avg_sq - exp_avg ** 2
                    mean_diff_sq = (grad - exp_avg) ** 2
                    w_diff_diff = state['total_w'] * (mean_diff_sq - moment_diff)
                    denominator = w_diff_diff + mean_diff_sq + moment_diff
                    if group['total_var']:
                        adv_beta = torch.sum(w_diff_diff) / torch.sum(denominator)

                        self.adaptive_beta_.append(torch.mean(adv_beta).item())
                        adv_beta = adv_beta.clamp(min=1 - beta2_max, max=1 - beta2_min).item()

                        exp_avg.mul_(1 - adv_beta).add_(adv_beta * grad)
                        exp_avg_sq.mul_(1 - adv_beta).add_(adv_beta * grad * grad)

                        state['total_w'] = state['total_w'] * (1 - adv_beta) + adv_beta
                    else:
                        neq_mask = (
                                    denominator != 0)  # torch.abs(grad - exp_avg) > (torch.abs(exp_avg) * group['neq_thresh'])
                        if not torch.all(neq_mask):
                            denominator = denominator[neq_mask]
                            w_diff_diff = w_diff_diff[neq_mask]
                            adv_beta = w_diff_diff / denominator

                            self.adaptive_beta_.append(torch.max(adv_beta).item())
                            adv_beta.clamp_(min=1 - beta2_max, max=1 - beta2_min)

                            adv_beta_comp = 1 - adv_beta
                            grad_sel = grad[neq_mask]
                            exp_avg[neq_mask] = exp_avg[neq_mask] * adv_beta_comp + adv_beta * grad_sel
                            exp_avg_sq[neq_mask] = exp_avg_sq[neq_mask] * adv_beta_comp + adv_beta * (
                                        grad_sel * grad_sel)

                            # equiv_mask = ~neq_mask
                            # when they are not equal, the current solution must be optimal, corresponding to beta=0
                            state['total_w'][neq_mask] = state['total_w'][neq_mask] * adv_beta_comp + adv_beta

                        else:
                            adv_beta = w_diff_diff / denominator
                            # clamp the range
                            self.adaptive_beta_.append(torch.max(adv_beta).item())
                            adv_beta.clamp_(min=1 - beta2_max, max=1 - beta2_min)

                            adv_beta_comp = 1 - adv_beta
                            exp_avg.mul_(adv_beta_comp).add_(adv_beta * grad)
                            exp_avg_sq.mul_(adv_beta_comp).add_(adv_beta * grad * grad)

                            state['total_w'] = state['total_w'] * adv_beta_comp + adv_beta

                if group['use_sgd']:
                    ema_grad.mul_(beta1).add_(grad)
                else:
                    ema_grad.mul_(beta1).add_(1 - beta1, grad)

                if group['nesterov']:
                    ema_grad = ema_grad * beta1 + grad * (1-beta1)

                beta2_pesudo = 0.9995 # a constant independent of the actual beta2
                beta2_t = beta2_pesudo ** state['step']
                N_sma_max = 2 / (1 - beta2_pesudo) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if not group['no_adamw'] and group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['use_sgd']:
                        step_size = group['lr'] * math.sqrt(
                            (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) * (N_sma_max) / N_sma / (N_sma_max - 2))
                    else:
                        step_size = group['lr'] * math.sqrt((N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) * (N_sma_max) / N_sma / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    if group['adam_step']:
                        denom = exp_avg_sq / state['total_w']
                    else:
                        denom = ((exp_avg_sq - exp_avg ** 2 / state['total_w']) / state['total_w']).clamp(min=0)

                    if amsgrad:
                        torch.max(denom, state['max_var'], out=state['max_var'])
                        denom.copy_(state['max_var'])

                    # there is some small numerical error
                    denom.sqrt_().add_(group['eps'])

                    if self.need_update_size:
                        self.update_size_ = -step_size * ema_grad / denom
                        p_data_fp32.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p_data_fp32.addcdiv_(-step_size, ema_grad, denom)

                    if self.need_lr:
                        self.adaptive_lrs_ = step_size / denom
                else:
                    if group['use_sgd']:
                        step_size = group['lr']
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])

                    if self.need_update_size:
                        self.update_size_ = -step_size * ema_grad
                        p_data_fp32.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p_data_fp32.add_(-step_size, ema_grad)
                    if self.need_lr:
                        self.adaptive_lrs_ = step_size

                p.data.copy_(p_data_fp32)

        return loss


@register_optimizer('radam_nesdiv')
class FairseqRAdamNesdiv(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = RAdam_nesdiv(params, **self.optimizer_config)

    @property
    def supports_flat_params(self):
        return True

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beta-n-range', default='(0.9, 0.99)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-betas', default='(0.9, 0.99)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--amsgrad', default=False, type=distutils.util.strtobool)
        parser.add_argument('--total-var', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-lr', default=False, type=distutils.util.strtobool)
        # parser.add_argument('--nesterov', default=False, type=distutils.util.strtobool) # using nesterov by default
        parser.add_argument('--need-update-size', default=False, type=distutils.util.strtobool)

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
            'beta_n_range': eval(self.args.beta_n_range),
            'adam_betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'amsgrad': self.args.amsgrad,
            'total_var': self.args.total_var,
            'need_lr': self.args.need_lr,
            "need_update_size": self.args.need_update_size
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


class RAdam_nesdiv(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, adam_betas=(0.9, 0.99), beta_n_range=(0.5, 0.99), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 total_var=False, need_lr=False, need_update_size=False):
        defaults = dict(lr=lr, adam_betas=adam_betas, beta_n_range=beta_n_range, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        total_var=total_var)

        self.this_ratio = 1
        self.need_lr = need_lr
        self.need_update_size = need_update_size
        # self.cache_len = cache_len
        super(RAdam_nesdiv, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def adaptive_beta(self):
        if len(self.adaptive_beta_) > 0:
            return sum(self.adaptive_beta_) / len(self.adaptive_beta_)
        else:
            return None

    @property
    def adaptive_lrs(self):
        if self.need_lr:
            if isinstance(self.adaptive_lrs_, float):
                return self.adaptive_lrs_, self.adaptive_lrs_, self.adaptive_lrs_
            else:
                return torch.min(self.adaptive_lrs_[self.adaptive_lrs_ > 0]).item(), torch.max(self.adaptive_lrs_).item(), torch.median(
                    self.adaptive_lrs_).item()
        else:
            return None, None, None

    @property
    def update_size(self):
        if self.need_update_size:
            return torch.min(self.update_size_).item(), torch.max(
                self.update_size_).item(), torch.median(
                self.update_size_).item()
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

                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p_data_fp32)
                    state['ema_grad_sq'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = torch.zeros_like(p_data_fp32)
                    state['total_w'] = torch.zeros_like(p_data_fp32) if not group['total_var'] else torch.zeros([]).to(
                        p_data_fp32)
                else:
                    state['ema_grad'] = state['ema_grad'].type_as(p_data_fp32)
                    state['ema_grad_sq'] = state['ema_grad_sq'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = state['max_var'].type_as(p_data_fp32)

                    state['total_w'] = state['total_w'].type_as(p_data_fp32)

                exp_avg_sq = state['exp_avg_sq']
                ema_grad, ema_grad_sq = state['ema_grad'], state['ema_grad_sq']
                beta1, beta2 = group['adam_betas']
                beta_n_min, beta_n_max = group['beta_n_range']
                state['step'] += 1

                state['ema_grad'].mul_(beta1).add_(1 - beta1, grad)
                # ema_grad = state['ema_grad'] / (1 - beta1 ** state['step'])

                grad2 = grad * grad
                ema_grad_sq.mul_(beta1).add_(1 - beta1, grad2)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, grad2)

                if state['step'] == 1:
                    beta_n_est = beta1
                else:
                    # ema_grad_sq = ema_grad_sq / (1 - beta1 ** state['step'])
                    # if group['total_var']:
                    #     beta_n_est = 0.5 + 0.5 * (torch.sum(ema_grad_sq) - torch.sum(ema_grad ** 2)) / torch.sum((grad - ema_grad) ** 2)
                    #
                    #     self.adaptive_beta_.append(torch.mean(beta_n_est).item())
                    #     beta_n_est.clamp_(min=beta_n_min, max=beta_n_max)
                    #
                    # else:
                    #     neq_mask = (grad != ema_grad)
                    #     if not torch.all(neq_mask):
                    #         ema_grad_sel, ema_grad_sq_sel, grad_sel = ema_grad[neq_mask], ema_grad_sq[neq_mask], grad[neq_mask]
                    #         beta_n_est_sel = 0.5 + 0.5 * (ema_grad_sq_sel - ema_grad_sel ** 2) / ((grad_sel - ema_grad_sel) ** 2)
                    #
                    #         self.adaptive_beta_.append(torch.median(beta_n_est_sel).item())
                    #         beta_n_est_sel.clamp_(min=beta_n_min, max=beta_n_max)
                    #
                    #         beta_n_est = beta1 * torch.ones_like(ema_grad)
                    #         beta_n_est[neq_mask] = beta_n_est_sel
                    #
                    #     else:
                    #         beta_n_est = 0.5 + 0.5 * (ema_grad_sq - ema_grad ** 2) / ((grad - ema_grad) ** 2)
                    #         # clamp the range
                    #         self.adaptive_beta_.append(torch.median(beta_n_est).item())
                    #         beta_n_est.clamp_(min=beta_n_min, max=beta_n_max)
                    total_w = 1 - beta1 ** state['step']
                    moment_diff = ema_grad_sq - ema_grad ** 2
                    mean_diff_sq = (grad - ema_grad) ** 2
                    diff_sum = mean_diff_sq + moment_diff
                    denominator = total_w * (mean_diff_sq - moment_diff) + diff_sum
                    if group['total_var']:
                        beta_n_est = torch.sum(diff_sum) / torch.sum(denominator)

                        self.adaptive_beta_.append(torch.max(beta_n_est).item())
                        beta_n_est.clamp_(min=beta_n_min, max=beta_n_max)
                    else:
                        neq_mask = (denominator != 0)
                        if not torch.all(neq_mask):
                            denominator = denominator[neq_mask]
                            diff_sum = diff_sum[neq_mask]
                            beta_n_est_sel = diff_sum / denominator

                            self.adaptive_beta_.append(torch.max(beta_n_est_sel).item())
                            beta_n_est_sel.clamp_(min=beta_n_min, max=beta_n_max)

                            beta_n_est = beta1 * torch.ones_like(ema_grad)
                            beta_n_est[neq_mask] = beta_n_est_sel
                        else:
                            beta_n_est = diff_sum / denominator
                            # clamp the range
                            self.adaptive_beta_.append(torch.max(beta_n_est).item())
                            beta_n_est.clamp_(min=beta_n_min, max=beta_n_max)
                # the nesterov update
                ema_grad = state['ema_grad'] * beta_n_est + grad * (1-beta_n_est)

                beta2_pesudo = 0.9995 # a constant independent of the actual beta2
                beta2_t = beta2_pesudo ** state['step']
                N_sma_max = 2 / (1 - beta2_pesudo) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt((1 - beta2**state['step']) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) * (N_sma_max) / N_sma / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    if amsgrad:
                        torch.max(exp_avg_sq, state['max_var'], out=state['max_var'])
                        denom = state['max_var'].sqrt().add(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add(group['eps'])

                    if self.need_update_size:
                        self.update_size_ = -step_size * ema_grad / denom
                        p_data_fp32.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p_data_fp32.addcdiv_(-step_size, ema_grad, denom)

                    if self.need_lr:
                        self.adaptive_lrs_ = step_size / denom
                else:
                    step_size = group['lr'] * math.sqrt(1 - beta2**state['step']) / (1 - beta1 ** state['step'])

                    if self.need_update_size:
                        self.update_size_ = -step_size * ema_grad
                        p_data_fp32.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p_data_fp32.add_(-step_size, ema_grad)
                    if self.need_lr:
                        self.adaptive_lrs_ = step_size

                p.data.copy_(p_data_fp32)

        return loss


@register_optimizer('radam_mule')
class FairseqRAdamMule(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = RAdam_mule(params, **self.optimizer_config)

    @property
    def supports_flat_params(self):
        return True

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.99)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--amsgrad', default=False, type=distutils.util.strtobool)
        parser.add_argument('--use-sgd', default=False, type=distutils.util.strtobool)
        parser.add_argument('--adam-step', default=False, type=distutils.util.strtobool)
        parser.add_argument('--total-var', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-lr', default=False, type=distutils.util.strtobool)
        parser.add_argument('--nesterov', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-update-size', default=False, type=distutils.util.strtobool)
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
            'adam_betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'amsgrad': self.args.amsgrad,
            'use_sgd': self.args.use_sgd,
            'adam_step': self.args.adam_step,
            'total_var': self.args.total_var,
            'need_lr': self.args.need_lr,
            'nesterov': self.args.nesterov,
            "need_update_size": self.args.need_update_size
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


class RAdam_mule(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, adam_betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0, amsgrad=False, use_sgd=False,
                 adam_step=False, total_var=False, need_lr=False, nesterov=False, need_update_size=False):
        defaults = dict(lr=lr, adam_betas=adam_betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, use_sgd=use_sgd,
                        adam_step=adam_step, total_var=total_var, nesterov=nesterov)

        self.this_ratio = 1
        self.need_lr = need_lr
        self.need_update_size = need_update_size
        # self.cache_len = cache_len
        super(RAdam_mule, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def adaptive_beta(self):
        if len(self.adaptive_beta_) > 0:
            return sum(self.adaptive_beta_) / len(self.adaptive_beta_)
        else:
            return None

    @property
    def adaptive_lrs(self):
        if self.need_lr:
            if isinstance(self.adaptive_lrs_, float):
                return self.adaptive_lrs_, self.adaptive_lrs_, self.adaptive_lrs_
            else:
                return torch.min(self.adaptive_lrs_[self.adaptive_lrs_ > 0]).item(), torch.max(self.adaptive_lrs_).item(), torch.median(
                    self.adaptive_lrs_).item()
        else:
            return None, None, None

    @property
    def update_size(self):
        if self.need_update_size:
            return torch.min(self.update_size_).item(), torch.max(
                self.update_size_).item(), torch.median(
                self.update_size_).item()
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

                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p_data_fp32)
                    state['ema_grad_sq'] = torch.zeros_like(p_data_fp32)
                    # state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = torch.zeros_like(p_data_fp32)
                else:
                    state['ema_grad'] = state['ema_grad'].type_as(p_data_fp32)
                    state['ema_grad_sq'] = state['ema_grad_sq'].type_as(p_data_fp32)
                    # state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = state['max_var'].type_as(p_data_fp32)

                exp_avg_sq = state['exp_avg_sq']
                ema_grad, ema_grad_sq = state['ema_grad'], state['ema_grad_sq']
                beta1, beta2 = group['adam_betas']
                state['step'] += 1

                ema_grad.mul_(beta1).add_(1 - beta1, grad)
                ema_grad_sq.mul_(beta1).addcmul_(1 - beta1, grad, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']

                if state['step'] == 1:
                    ema_grad_var = exp_avg_sq
                else:
                    if bc1 < 1 - 1e-5:
                        ema_grad_var = (ema_grad_sq - ema_grad**2 / bc1) / bc1
                    else:
                        ema_grad_var = ema_grad_sq - ema_grad**2
                    ema_grad_var.clamp_(min=0)

                beta2_pesudo = 0.9995 # a constant independent of the actual beta2
                beta2_t = beta2_pesudo ** state['step']
                N_sma_max = 2 / (1 - beta2_pesudo) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if group['nesterov']:
                    ema_grad = ema_grad * beta1 + grad * (1-beta1)
                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt((N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) * (N_sma_max) / N_sma / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    step_size *= math.sqrt(2)
                    if bc2 < 1 - 1e-5:
                        denom = (exp_avg_sq / bc2 + ema_grad_var)
                    else:
                        denom = exp_avg_sq + ema_grad_var

                    if amsgrad:
                        torch.max(denom, state['max_var'], out=state['max_var'])
                        denom.copy_(state['max_var'])

                    # there is some small numerical error
                    denom.sqrt_().add_(group['eps'])

                    if self.need_update_size:
                        self.update_size_ = -step_size * ema_grad / denom
                        p_data_fp32.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p_data_fp32.addcdiv_(-step_size, ema_grad, denom)

                    if self.need_lr:
                        self.adaptive_lrs_ = step_size / denom
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])

                    if self.need_update_size:
                        self.update_size_ = -step_size * ema_grad
                        p_data_fp32.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p_data_fp32.add_(-step_size, ema_grad)
                    if self.need_lr:
                        self.adaptive_lrs_ = step_size

                p.data.copy_(p_data_fp32)

        return loss


@register_optimizer('adam_multibeta')
class FairseqRAdamMultibeta(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = RAdam_mule(params, **self.optimizer_config)

    @property
    def supports_flat_params(self):
        return True

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='betas for Adam optimizer')
        parser.add_argument('--beta2s', default='(0.9, 0.99)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--amsgrad', default=False, type=distutils.util.strtobool)
        parser.add_argument('--use-sgd', default=False, type=distutils.util.strtobool)
        parser.add_argument('--adam-step', default=False, type=distutils.util.strtobool)
        parser.add_argument('--total-var', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-lr', default=False, type=distutils.util.strtobool)
        parser.add_argument('--nesterov', default=False, type=distutils.util.strtobool)
        parser.add_argument('--need-update-size', default=False, type=distutils.util.strtobool)
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
            'beta2s': eval(self.args.beta2s),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'amsgrad': self.args.amsgrad,
            'use_sgd': self.args.use_sgd,
            'adam_step': self.args.adam_step,
            'total_var': self.args.total_var,
            'need_lr': self.args.need_lr,
            'nesterov': self.args.nesterov,
            "need_update_size": self.args.need_update_size
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


class Adam_multibeta(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2s=(0.9, 0.99), eps=1e-8,
                 weight_decay=0, amsgrad=False, use_sgd=False,
                 adam_step=False, total_var=False, need_lr=False, nesterov=False, need_update_size=False,
                 need_denoms=False, use_adabound=False, final_lr=0.1, gamma=1e-3, use_adamw=True):
        defaults = dict(lr=lr, beta1=beta1, beta2s=beta2s, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, use_sgd=use_sgd,
                        adam_step=adam_step, total_var=total_var, nesterov=nesterov, use_adabound=use_adabound,
                        final_lr=final_lr, gamma=gamma, use_adamw=use_adamw)

        self.this_ratio = 1
        self.need_lr = need_lr
        self.need_update_size = need_update_size
        self.need_denoms = need_denoms
        # self.cache_len = cache_len
        super(Adam_multibeta, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def adaptive_beta(self):
        if len(self.adaptive_beta_) > 0:
            return sum(self.adaptive_beta_) / len(self.adaptive_beta_)
        else:
            return None

    @property
    def adaptive_lrs(self):
        if self.need_lr:
            if isinstance(self.adaptive_lrs_, float):
                return self.adaptive_lrs_, self.adaptive_lrs_, self.adaptive_lrs_
            else:
                return torch.min(self.adaptive_lrs_[self.adaptive_lrs_ > 0]).item(), torch.max(self.adaptive_lrs_).item(), torch.median(
                    self.adaptive_lrs_).item()
        else:
            return None, None, None

    @property
    def update_size(self):
        if self.need_update_size:
            return torch.min(self.update_size_).item(), torch.max(
                self.update_size_).item(), torch.median(
                self.update_size_).item()
        else:
            return None, None, None

    @property
    def denoms(self):
        if self.need_denoms:
            return torch.min(self.adam_denom_).item(), \
                   torch.max(self.adam_denom_).item(), \
                   torch.median(self.adam_denom_).item(), \
                    torch.min(self.var_denom_).item(), torch.max(self.var_denom_).item(),\
                   torch.median(self.var_denom_).item()
        else:
            return None, None, None, None, None, None

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

        for group, base_lr in zip(self.param_groups, self.base_lrs):

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                if not group['use_adamw'] and group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'] * group['lr'], p_data_fp32)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p_data_fp32)
                    # state['ema_grad_sq'] = torch.zeros_like(p_data_fp32)
                    # state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = [torch.zeros_like(p_data_fp32) for _ in group['beta2s']]
                    if amsgrad:
                        state['max_var'] = torch.zeros_like(p_data_fp32)

                exp_avg_sq = state['exp_avg_sq']
                beta2s = group['beta2s']
                ema_grad = state['ema_grad']
                beta1 = group['beta1']
                state['step'] += 1

                ema_grad.mul_(beta1).add_(1 - beta1, grad)
                for b2, easq_ in zip(group['beta2s'], exp_avg_sq):
                    easq_.mul_(b2).addcmul_(1 - b2, grad, grad)
                    # pdb.set_trace() # check if the values have been changed successfully

                if group['use_adamw'] and group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if group['nesterov']:
                    ema_grad = ema_grad * beta1 + grad * (1-beta1)
                # more conservative since it's an approximated value
                step_size = group['lr'] / (1 - beta1 ** state['step'])

                denom = 0
                for b2, easq_ in zip(beta2s, exp_avg_sq):
                    bc2 = 1 - b2 ** state['step']
                    denom += easq_ / (bc2 * len(exp_avg_sq))
                if self.need_denoms:
                    self.adam_denom_ = exp_avg_sq[-1] / (1 - beta2s[-1] ** state['step'])
                    self.var_denom_ = exp_avg_sq[0] / (1 - beta2s[0] ** state['step'])

                if amsgrad:
                    torch.max(denom, state['max_var'], out=state['max_var'])
                    denom.copy_(state['max_var'])

                # there is some small numerical error
                denom.sqrt_().add_(group['eps'])
                if group['use_adabound']:
                    final_lr = group['final_lr'] * group['lr'] / base_lr
                    lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                    upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                    step_size = torch.full_like(denom, step_size)
                    step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(ema_grad)
                    p_data_fp32.add_(-step_size)
                else:
                    if self.need_update_size:
                        self.update_size_ = -step_size * ema_grad / denom
                        p_data_fp32.add_(self.update_size_)
                        self.update_size_ = self.update_size_.abs()
                    else:
                        p_data_fp32.addcdiv_(-step_size, ema_grad, denom)

                    if self.need_lr:
                        self.adaptive_lrs_ = step_size / denom

                p.data.copy_(p_data_fp32)

        return loss
