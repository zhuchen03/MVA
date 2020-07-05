import numpy as np
import math
import os
import collections
import pickle
import pdb
from matplotlib import pyplot as plt


class Adam:
    def __init__(self, x, lr=0.1, beta1=0.9, beta2=0.99):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.steps = 0

        self.x_avg = np.zeros_like(x)
        self.x2_avg = np.zeros_like(x)

    def step(self, x, grad):
        self.x_avg = self.beta1 * self.x_avg + (1 - self.beta1) * grad
        self.x2_avg = self.beta2 * self.x2_avg + (1 - self.beta2) * (grad ** 2)
        self.steps += 1

        bc1 = 1 - self.beta1 ** self.steps
        bc2 = 1 - self.beta2 ** self.steps

        lr = self.lr

        return x - lr * self.x_avg / bc1 / (np.sqrt(self.x2_avg / bc2) + 1e-10)


class MAdam:
    def __init__(self, x, lr=0.1, beta1=0.9, beta2_range=(0.5, 0.99)):
        self.lr = lr
        self.beta1 = beta1
        self.beta2_range = beta2_range
        self.steps = 0

        self.x_avg = np.zeros_like(x)
        self.x2_avg = np.zeros_like(x)
        self.total_w = np.zeros_like(x)

    def step(self, x, grad):
        self.x_avg = self.beta1 * self.x_avg + (1 - self.beta1) * grad
        # madam step
        if self.steps == 0:
            adv_beta = self.beta2_range[1]
        else:
            moment_diff = self.x2_avg / self.total_w - (self.x_avg / self.total_w) ** 2
            mean_diff_sq = (grad - self.x_avg / self.total_w) ** 2
            # w_diff_diff = total_w * (mean_diff_sq - moment_diff)
            sum_diff = mean_diff_sq + moment_diff
            denominator = (mean_diff_sq - moment_diff) * self.total_w + sum_diff

            adv_beta = sum_diff / (denominator + 1e-16)
            # print("grad: {}, beta: {}".format(grad, adv_beta))
            adv_beta = np.clip(adv_beta, self.beta2_range[0], self.beta2_range[1])

        self.x2_avg = adv_beta * self.x2_avg + (1 - adv_beta) * (grad ** 2)
        self.steps += 1

        self.total_w = adv_beta * self.total_w + (1 - adv_beta)

        bc1 = 1 - self.beta1 ** self.steps

        lr = self.lr

        return x - lr * self.x_avg / bc1 / (np.sqrt(self.x2_avg / self.total_w) + 1e-10)


def runexp(optimizer, x_init, sigmas, hs, n_iters):
    traj = [x_init]

    x = x_init
    for t in range(n_iters):

        noise = np.random.randn(*x.shape)
        grad = hs * (x - sigmas * noise)

        x = optimizer.step(x, grad)
        traj.append(x)
    traj = np.stack(traj)
    return traj


def loss(x, hs):
    return 0.5 * np.sum(hs * (x*x))


if __name__ == "__main__":
    sigmas = np.array([0.2, 0.2])
    hs = np.array([1, 20])

    n_trials = 100
    adam_trajs, madam_trajs = collections.defaultdict(list), collections.defaultdict(list)
    adam_errs, madam_errs = collections.defaultdict(list), collections.defaultdict(list)
    adam_b2s = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    lr_adam_list = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2]
    lr_madam_list = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2]
    n_iters = 1000

    adam_min_err, madam_min_err = float('inf'), float('inf')
    for t in range(n_trials):
        x_init = np.random.randn(2)

        for lr_adam in lr_adam_list:
            for adam_b2 in adam_b2s:
                adam_opt = Adam(x_init, lr=lr_adam, beta1=0.9, beta2=adam_b2)
                traj_adam = runexp(adam_opt, x_init, sigmas, hs, n_iters)

                name = "lr{}_b_{}".format(lr_adam, adam_b2)
                adam_trajs[name].append(traj_adam)
                # err = np.sum(np.abs(traj_adam[-1]))
                err = np.sum(loss(traj_adam[-1], hs))
                adam_errs[name].append(err)

                if err < adam_min_err:
                    adam_min_err = err

                print("Trial {}, Adam lr {}, beta2 {}, error {:.2e}, current min: {:.2e}".format(t, lr_adam, adam_b2, err, adam_min_err))

        for lr_madam in lr_madam_list:
            madam_opt = MAdam(x_init, lr=lr_madam, beta1=0.9, beta2_range=(0.5, 0.99))
            traj_madam = runexp(madam_opt, x_init, sigmas, hs, n_iters)

            name = "lr{}_b_{}_{}".format(lr_madam, 0.5, 0.99)
            madam_trajs[name].append(traj_madam)
            # err = np.sum(np.abs(traj_madam[-1]))
            err = np.sum(loss(traj_madam[-1], hs))
            madam_errs[name].append(err)

            if err < madam_min_err:
                madam_min_err = err
            print("Trial {}, MAdam lr {}, beta2 {}-{}, error {:.2e}, current min: {:.2e}".format(t, lr_madam, 0.5, 0.99, err, madam_min_err))

    adam_min_err, madam_min_err = float('inf'), float('inf')
    for key, val in adam_errs.items():
        err = np.mean(val)
        if err < adam_min_err:
            adam_min_err = err
            adam_min_stder = np.std(val) / math.sqrt(n_trials)
            adam_min_lr = key.split('_')[0][2:]
            adam_min_beta = key.split('_')[-1]

    for key, val in madam_errs.items():
        err = np.mean(val)
        if err < madam_min_err:
            madam_min_err = err
            madam_min_stder = np.std(val) / math.sqrt(n_trials)
            madam_min_lr = key.split('_')[0][2:]

    print("error: Adam: {:.2e} ({:.2e}), lr: {}, beta2: {}, MADAM: {:.2e} ({:.2e}), lr: {}".format(adam_min_err,
                                adam_min_stder, adam_min_lr, adam_min_beta, madam_min_err, madam_min_stder, madam_min_lr))
    if not os.path.exists('quadratic_trajs'):
        os.makedirs('quadratic_trajs')

    out_res = {'MAdamTrajs': madam_trajs, 'AdamTrajs': adam_trajs, "MadamErrs": madam_errs, "AdamErrs": adam_errs}

    pickle.dump(out_res, open("quadratic_trajs/res_hs_{}_{}_sigma_{}_{}.pkl".format(hs[0], hs[1], sigmas[0], sigmas[1]), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)




