import torch


class LaMadam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, momentum=0.9, beta=0.98, beta_min=0.5,
                    eps=1e-15, weight_decay=0, use_adamw=False, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, beta=beta, beta_min=beta_min,
                        eps=eps, weight_decay=weight_decay,
                        use_adamw=use_adamw, amsgrad=amsgrad)
        super(LaMadam, self).__init__(params, defaults)

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

        for group in self.param_groups:
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
                    state['update_lr_bc'] = 0.
                    state['update_est'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['g_sq_est'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['g_est'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['total_w'] = torch.zeros_like(p)

                    if group['amsgrad']:
                        state['max_sq_est'] = torch.zeros_like(p)

                update_est, g_sq_est = state['update_est'], state['g_sq_est']
                momentum, beta = group['momentum'], group['beta']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if group['use_adamw']:
                        p.data.add_(-group['weight_decay'] * group['lr'], p.data)
                    else:
                        grad.data.add_(p, alpha=group['weight_decay'])

                if state['step'] > 1 and group['beta_min'] != beta:
                    total_w = state['total_w']
                    g_sq_est_unbiased = g_sq_est / total_w
                    g_est_unbiased = state['g_est'] / total_w
                    moment_diff = g_sq_est_unbiased - g_est_unbiased ** 2
                    mean_diff_sq = (grad - g_est_unbiased) ** 2
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

                if getattr(group, 'amsgrad', False):
                    torch.max(state['max_sq_est'], g_sq_est, out=state['max_sq_est'])
                    g_sq_est = state['max_sq_est']

                denom = g_sq_est.div(total_w).sqrt_().add_(group['eps'])

                update_est.mul_(momentum).addcdiv_((1 - momentum) * group['lr'], grad, denom)
                state['update_lr_bc'] = state['update_lr_bc'] * momentum + group['lr'] * (1 - momentum)

                step_size = group['lr'] / state['update_lr_bc']

                if True:
                    update = - step_size * update_est
                    p.add_(update)
                    self.update_size_ = torch.mean(update.abs()).item()
                else:
                    p.add_(-step_size, update_est)

        return loss
