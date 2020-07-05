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


@register_optimizer('radam_nesada')
class FairseqRAdamNesada(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = RAdam_nesada(params, **self.optimizer_config)

    @property
    def supports_flat_params(self):
        return True

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--alpha-range', default='(0.9, 0.99)', metavar='B',
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
        parser.add_argument('--need-valid-ratio', default=False, type=distutils.util.strtobool)
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
            'alpha_range': eval(self.args.alpha_range),
            'adam_betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'amsgrad': self.args.amsgrad,
            'total_var': self.args.total_var,
            'need_lr': self.args.need_lr,
            "need_update_size": self.args.need_update_size,
            "need_valid_ratio": self.args.need_valid_ratio
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


class RAdam_nesada(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, adam_betas=(0.9, 0.99), alpha_range=(0, 0.9), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 total_var=False, need_lr=False, need_update_size=False, need_valid_ratio=False):
        defaults = dict(lr=lr, adam_betas=adam_betas, alpha_range=alpha_range, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        total_var=total_var)

        self.this_ratio = 1
        self.need_lr = need_lr
        self.need_update_size = need_update_size
        self.need_valid_ratio = need_valid_ratio
        # self.cache_len = cache_len
        super(RAdam_nesada, self).__init__(params, defaults)

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
    def valid_ratio(self):
        if self.need_valid_ratio:
            return getattr(self, "valid_ratio_", None)
        else:
            return None

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
                    state['last_alpha'] = torch.zeros_like(p_data_fp32) if not group['total_var'] else torch.zeros([]).to(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = torch.zeros_like(p_data_fp32)
                else:
                    state['ema_grad'] = state['ema_grad'].type_as(p_data_fp32)
                    state['ema_grad_sq'] = state['ema_grad_sq'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_var'] = state['max_var'].type_as(p_data_fp32)

                    state['last_alpha'] = state['last_alpha'].type_as(p_data_fp32)

                exp_avg_sq = state['exp_avg_sq']
                ema_grad, ema_grad_sq = state['ema_grad'], state['ema_grad_sq']
                beta1, beta2 = group['adam_betas']
                alpha_min, alpha_max = group['alpha_range']
                state['step'] += 1

                last_ema_grad = state['ema_grad'].clone()
                last_ema_grad_sq = state['ema_grad_sq'].clone()
                last_alpha = state['last_alpha']

                state['ema_grad'].mul_(beta1).add_(1 - beta1, grad)
                ema_grad = state['ema_grad']

                grad2 = grad * grad
                ema_grad_sq.mul_(beta1).add_(1 - beta1, grad2)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, grad2)

                if state['step'] == 1:
                    this_alpha = beta1
                else:
                    beta1_t = beta1 ** state['step']
                    beta1_t_last = beta1 ** (state['step'] - 1)
                    # if beta1_t_last > 1e-4:
                    #     mom_diff = (1 - beta1_t) * grad - ema_grad
                    #     mom2_diff = (1 - beta1_t) * grad2 - ema_grad_sq
                    #
                    #     this_alpha = (1 - beta1_t - (1 - beta1_t_last) * last_alpha) * mom2_diff \
                    #                     - 2 * (ema_grad - last_ema_grad * last_alpha) * mom_diff
                    #     this_alpha.div_(((1 - beta1_t) * mom2_diff - 2 * ema_grad * mom_diff).add(group['eps']))
                    # else:
                    #     # switch to a simpler formula
                    #     mom_diff = grad - ema_grad
                    #     mom2_diff = grad2 - ema_grad_sq
                    #
                    #     this_alpha = (1 - last_alpha) * mom2_diff - 2 * (ema_grad - last_ema_grad * last_alpha) * mom_diff
                    #     this_alpha.div_((mom2_diff - 2 * ema_grad * mom_diff).add(group['eps']))
                    # print(torch.min(this_alpha).item(), torch.max(this_alpha).item(), torch.median(this_alpha).item())

                    if beta1_t_last > 1e-4:
                        mom_diff = (1 - beta1_t_last) * ema_grad - (1 - beta1_t) * last_ema_grad
                        mom2_diff = (1 - beta1_t) * last_ema_grad_sq - (1 - beta1_t_last) * ema_grad_sq

                        this_alpha = ((1 - beta1_t_last) * last_alpha - (1 - beta1_t)) * mom2_diff \
                                        + 2 * (last_ema_grad * last_alpha - ema_grad) * mom_diff
                        this_alpha.div_(((1 - beta1_t) * mom2_diff + 2 * ema_grad * mom_diff).add(group['eps']))
                    else:
                        # switch to a simpler formula
                        mom_diff = ema_grad - last_ema_grad
                        mom2_diff = last_ema_grad_sq - ema_grad_sq

                        this_alpha = (last_alpha - 1) * mom2_diff + 2 * (last_ema_grad * last_alpha - ema_grad) * mom_diff
                        this_alpha.div_((mom2_diff + 2 * ema_grad * mom_diff).add(group['eps']))

                    if self.need_valid_ratio:
                        valid_total = torch.sum((torch.abs(this_alpha) > alpha_min) & (torch.abs(this_alpha) < alpha_max)).item()
                        self.valid_ratio_ = valid_total / float(this_alpha.numel())
                    # print(valid_total, valid_total / float(this_alpha.numel()))
                    # pdb.set_trace()
                    this_alpha.clamp_(min=alpha_min, max=alpha_max)

                # the nesterov update
                # ema_grad = state['ema_grad'] * beta_n_est + grad * (1-beta_n_est)
                ema_grad = (1 + this_alpha) * ema_grad - last_alpha * last_ema_grad
                if state['step'] == 1:
                    last_alpha[:] = this_alpha
                else:
                    last_alpha.copy_(this_alpha)

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

