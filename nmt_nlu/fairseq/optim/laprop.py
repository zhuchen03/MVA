import torch.optim

from . import FairseqOptimizer, register_optimizer
import math
import distutils.util
import pdb


@register_optimizer('lamadam')
class FairseqLaMAdam(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = LaMAdam(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--varscale-beta', default=0.9, type=float,
                            help='betas for LaProp optimizer')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--beta-min', default=0.5, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--varscale-eps', type=float, default=1e-15, metavar='D',
                            help='epsilon for LaProp optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--use-adam', default=False, action="store_true")
        parser.add_argument('--eps-schedule', default=False, action="store_true")
        parser.add_argument('--nesterov', default=False, action="store_true")
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
            'beta': self.args.varscale_beta,
            'momentum': self.args.momentum,
            'eps': self.args.varscale_eps,
            'weight_decay': self.args.weight_decay,
            'use_adam': self.args.use_adam,
            'beta_min': self.args.beta_min,
            'nesterov': self.args.nesterov
        }


class LaMAdam(torch.optim.Optimizer):
    """Implements Adadelta algorithm.
    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, beta=0.98, beta_min=0.5, scale=1,
            eps=1e-15, weight_decay=0, use_adam=False,
                 nesterov=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, beta=beta, beta_min=beta_min, scale=scale,
                        eps=eps, weight_decay=weight_decay, use_adam=use_adam, nesterov=nesterov)
        super(LaMAdam, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    @property
    def supports_flat_params(self):
        return True

    @property
    def var_adapt(self):
        if getattr(self, "var_adapt_", None) is not None:
            return self.var_adapt_
        else:
            return None

    @property
    def update_size(self):
        if getattr(self, "update_size_", None) is not None:
            return None, None, self.update_size_
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

        self.update_size_ = None
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adadelta does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['update_est'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['update_avg_lr'] = 0.
                    state['g_sq_est'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['g_est'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['total_w'] = torch.zeros_like(p)

                update_est, g_sq_est = state['update_est'], state['g_sq_est']
                momentum, beta = group['momentum'], group['beta']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if group['use_adam']:
                        grad = grad.add(p, alpha=group['weight_decay'])
                    else:
                        p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                if state['step'] > 1 and group['beta_min'] != beta:
                    total_w = state['total_w']
                    exp_avg_sq_unbiased = state['g_sq_est'] / total_w
                    exp_avg_unbiased = state['g_est'] / total_w
                    moment_diff = exp_avg_sq_unbiased - exp_avg_unbiased ** 2
                    mean_diff_sq = (grad - exp_avg_unbiased) ** 2
                    sum_diff = mean_diff_sq + moment_diff
                    denominator = (mean_diff_sq - moment_diff).mul_(total_w).add_(sum_diff)

                    adv_beta = sum_diff.div_(denominator.add_(1e-16))
                    # clamp the range
                    adv_beta.clamp_(min=group['beta_min'], max=beta)

                    all_beta = adv_beta
                    all_beta_comp = 1 - all_beta

                    state['g_est'].mul_(all_beta).add_(all_beta_comp * grad)
                    g_sq_est.mul_(all_beta).add_(all_beta_comp.mul(grad).mul_(grad))
                    total_w.mul_(all_beta).add_(all_beta_comp)
                else:
                    g_sq_est.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                    total_w = 1 - beta ** state['step']
                    if 'total_w' in state:
                        state['total_w'][:] = total_w
                        state['g_est'].mul_(beta).add_(1 - beta, grad)

                eps = group['eps']

                denom = g_sq_est.div(total_w).sqrt_().add_(eps)

                update_est.mul_(momentum).addcdiv_((1 - momentum) * group['lr'], grad, denom)

                state['update_avg_lr'] = state['update_avg_lr'] * momentum + group['lr'] * (1 - momentum)
                # typically happens in the first step with zero learning rate
                step_size = group['lr'] / state['update_avg_lr'] if state['update_avg_lr'] > 0 else group['lr']

                if group['nesterov']:
                    update_est = update_est.mul(momentum).addcdiv_((1 - momentum) * group['lr'], grad, denom)

                if True:
                    # need to return update size
                    update = -step_size * update_est
                    self.update_size_ = update.abs().mean().item()
                    p.add_(update)
                else:
                    p.add_(-step_size, update_est)

        return loss

