import torch
import torch.nn.functional as f
import torch.distributions as d
import torch.nn as nn


def loss_vae_bernouilli(x_hat, x, args, mus=None, log_vars=None, beta=1):
    rec = f.binary_cross_entropy(x_hat.view(x.size(0), -1), x.view(x.size(0), -1), reduction='none').sum(-1).mean(0)
    if mus is not None and log_vars is not None:  # vae
        kl_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp(), dim=[-1, -2]).mean(0)
    else:  # ae
        kl_loss = torch.zeros_like(rec)
    return rec + beta*kl_loss, rec, kl_loss