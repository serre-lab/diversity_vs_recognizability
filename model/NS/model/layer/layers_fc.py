import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

# # log var = log sigma^2 = 2 log sigma
# def f_std(logvar):
#     return torch.exp(0.5 * logvar)


# class FCResBlock(nn.Module):
#     """
#     Residual block.

#     Attributes:
#         dim: layer width.
#         num_layers: number of layers in residual block.
#         activation: specify activation.
#         batch_norm: use or not use batch norm.
#     """

#     def __init__(
#         self,
#         dim: int,
#         num_layers: int,
#         activation: nn,
#         batch_norm: bool = False,
#         dropout: bool = False,
#     ):
#         super(FCResBlock, self).__init__()
#         self.num_layers = num_layers
#         self.activation = activation
#         self.batch_norm = batch_norm
#         self.dropout = dropout

#         self.block = []
#         for _ in range(self.num_layers):
#             layer = [nn.Linear(dim, dim)]
#             if self.batch_norm:
#                 layer.append(nn.BatchNorm1d(num_features=dim))
#             if self.dropout:
#                 layer.append(nn.Dropout(p=0.1))
#             self.block.append(nn.ModuleList(layer))
#         self.block = nn.ModuleList(self.block)

#     def forward(self, x):
#         e = x + 0
#         for l, layer in enumerate(self.block):
#             for i in range(len(layer)):
#                 e = layer[i](e)
#             if l < (len(self.block) - 1):
#                 e = self.activation(e)
#         # residual
#         return self.activation(e + x)


# class Conv2d3x3(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
#         super(Conv2d3x3, self).__init__()
#         stride = 2 if downsample else 1
#         self.conv = nn.Conv2d(
#             in_channels, out_channels, kernel_size=3, padding=1, stride=stride
#         )

#     def forward(self, x):
#         return self.conv(x)


# class SharedConvEncoder(nn.Module):
#     def __init__(self, nc: int, activation: nn):
#         super(SharedConvEncoder, self).__init__()
#         self.activation = activation

#         self.conv_layers = nn.ModuleList(
#             [
#                 Conv2d3x3(in_channels=1, out_channels=nc),
#                 Conv2d3x3(in_channels=nc, out_channels=nc),
#                 Conv2d3x3(in_channels=nc, out_channels=nc, downsample=True),
#                 # nn.Conv2d(
#                 #     in_channels=nc, out_channels=nc, kernel_size=4, padding=1, stride=2
#                 # ),
#                 # shape is now (-1, 64, 14 , 14)
#                 Conv2d3x3(in_channels=nc, out_channels=2 * nc),
#                 Conv2d3x3(in_channels=2 * nc, out_channels=2 * nc),
#                 # Conv2d3x3(in_channels=2 * nc, out_channels=2 * nc, downsample=True),
#                 nn.Conv2d(
#                     in_channels=2 * nc, out_channels=2 * nc, kernel_size=4, padding=0, stride=2
#                 ),
#                 # shape is now (-1, 128, 7, 7)
#                 Conv2d3x3(in_channels=2 * nc, out_channels=4 * nc),
#                 Conv2d3x3(in_channels=4 * nc, out_channels=4 * nc),
#                 # Conv2d3x3(in_channels=4 * nc, out_channels=4 * nc, downsample=True)
#                 nn.Conv2d(
#                     in_channels=4 * nc, out_channels=4 * nc, kernel_size=4, padding=0, stride=2
#                 ),
#                 # shape is now (-1, 256, 4, 4)
#             ]
#         )

#         self.bn_layers = nn.ModuleList(
#             [
#                 nn.BatchNorm2d(num_features=nc),
#                 nn.BatchNorm2d(num_features=nc),
#                 nn.BatchNorm2d(num_features=nc),
#                 nn.BatchNorm2d(num_features=2 * nc),
#                 nn.BatchNorm2d(num_features=2 * nc),
#                 nn.BatchNorm2d(num_features=2 * nc),
#                 nn.BatchNorm2d(num_features=4 * nc),
#                 nn.BatchNorm2d(num_features=4 * nc),
#                 nn.BatchNorm2d(num_features=4 * nc),
#             ]
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = x.view(-1, 1,50,50)
#         for conv, bn in zip(self.conv_layers, self.bn_layers):
#             h = conv(h)
#             h = bn(h)
#             h = self.activation(h)
            
#         return h


# class PreProj(nn.Module):
#     def __init__(self, hid_dim: int, n_features: int, activation: nn):
#         super(PreProj, self).__init__()

#         self.n_features = n_features
#         self.hid_dim = hid_dim
#         self.activation = activation

#         # modules
#         self.fc = nn.Linear(self.n_features, self.hid_dim)
#         self.bn = nn.BatchNorm1d(self.hid_dim)

#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         # reshape and affine
#         e = h.view(-1, self.n_features)
#         e = self.fc(e)
#         e = self.bn(e)
#         e = self.activation(e)
#         return e


# class PostPool(nn.Module):
#     def __init__(self, hid_dim: int, c_dim: int, activation: nn, ladder=False):
#         super(PostPool, self).__init__()

#         self.hid_dim = hid_dim
#         self.c_dim = c_dim
#         self.activation = activation
#         self.drop = nn.Dropout(0.1)
#         self.ladder = ladder

#         self.k = 2
#         if ladder:
#             self.k = 3

#         # modules
#         self.linear_layers = nn.ModuleList(
#             [
#                 nn.Linear(self.hid_dim, self.hid_dim),
#                 nn.Linear(self.hid_dim, self.hid_dim),
#             ]
#         )
#         self.bn_layers = nn.ModuleList(
#             [nn.BatchNorm1d(self.hid_dim), nn.BatchNorm1d(self.hid_dim)]
#         )

#         self.linear_params = nn.Linear(self.hid_dim, self.k * self.c_dim)
#         self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

#     def forward(self, e):
#         e = e.view(-1, self.hid_dim)
#         for fc, bn in zip(self.linear_layers, self.bn_layers):
#             e = fc(e)
#             e = bn(e)
#             e = self.activation(e)

#         # affine transformation to parameters
#         e = self.linear_params(e)
#         # 'global' batch norm
#         e = e.view(-1, 1, self.k * self.c_dim)
#         e = self.bn_params(e)
#         e = e.view(-1, self.k * self.c_dim)

#         if self.ladder:
#             mean, logvar, feats = e.chunk(self.k, dim=1)
#             return mean, logvar, feats
#         else:
#             mean, logvar = e.chunk(self.k, dim=1)
#             return mean, logvar


# ########################################################################
# class StatistiC(nn.Module):
#     """
#     Compute the statistic q(c | X).
#     Encode X, preprocess, aggregate, postprocess.
#     Representation for context at layer L.

#     Attributes:
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         n_features: number of features in output of the shared encoder (256 * 4 * 4).
#         activation: specify activation.
#     """

#     def __init__(
#         self,
#         hid_dim: int,
#         c_dim: int,
#         n_features: int,
#         activation: nn,
#         dropout_sample: bool = False,
#         mode: str = "mean",
#     ):
#         super(StatistiC, self).__init__()

#         self.c_dim = c_dim
#         self.mode = mode
#         self.hid_dim = hid_dim
#         self.activation = activation

#         self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation)

#     def forward(self, h, bs, ns):
#         # aggregate samples
#         h = h.view(bs, -1, self.hid_dim)
#         if self.mode == "mean":
#             a = h.mean(dim=1)
#         elif self.mode == "max":
#             a = h.max(dim=1)[0]
#         # map to moments
#         mean, logvar = self.postpool(a)
#         return mean, logvar, a, None


# ########################################################################
# # Prior for z - p(z_l | z_{l+1}, c_l)
# class PriorZ(nn.Module):
#     """
#     Latent decoder for the sample.

#     Attributes:
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         z_dim: number of features for latent variable.
#         h_dim: number of features for the embedded samples.
#         activation: specify activation.
#     """

#     def __init__(
#         self,
#         num_layers: int,
#         hid_dim: int,
#         c_dim: int,
#         z_dim: int,
#         h_dim: int,
#         activation: nn,
#     ):
#         super(PriorZ, self).__init__()
#         self.num_layers = num_layers
#         self.hid_dim = hid_dim

#         self.c_dim = c_dim
#         self.z_dim = z_dim

#         self.activation = activation
#         self.drop = nn.Dropout(0.1)

#         # modules
#         self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
#         self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

#         self.residual_block = FCResBlock(
#             dim=self.hid_dim,
#             num_layers=self.num_layers,
#             activation=self.activation,
#             batch_norm=True,
#         )

#         self.linear_params = nn.Linear(self.hid_dim, 2 * self.z_dim)
#         self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

#     def forward(self, z, c, bs, ns):
#         """
#         Args:
#             z: latent for samples at layer l+1.
#             c: latent representations for context at layer l.
#         Returns:
#             Moments of the generative distribution for z at layer l.
#         """
#         # combine z and c
#         # embed z if we have more than one stochastic layer
#         if z is not None:
#             ez = z.view(-1, self.z_dim)
#             ez = self.linear_z(ez)
#             ez = ez.view(bs, -1, self.hid_dim)
#         else:
#             ez = c.new_zeros((bs, ns, self.hid_dim))

#         if c is not None:
#             ec = self.linear_c(c)
#             ec = ec.view(bs, -1, self.hid_dim).expand_as(ez)
#         else:
#             ec = ez.new_zeros((bs, ns, self.hid_dim))

#         # sum and reshape
#         e = ez + ec
#         e = e.view(-1, self.hid_dim)
#         e = self.activation(e)

#         # for layer in self.linear_block:
#         e = self.residual_block(e)

#         # affine transformation to parameters
#         e = self.linear_params(e)

#         # 'global' batch norm
#         e = e.view(-1, 1, 2 * self.z_dim)
#         e = self.bn_params(e)
#         e = e.view(-1, 2 * self.z_dim)
#         # e = self.drop(e)

#         mean, logvar = e.chunk(2, dim=1)
#         return mean, logvar


# ########################################################################
# class EncoderBottomUp(nn.Module):
#     """
#     Bottom-up deterministic Inference network h_l = g(h_{l-1}, c) return representations
#     for x=h_1 at each layer.

#     Attributes:
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         activation: specify activation.

#     """

#     def __init__(self, num_layers: int, hid_dim: int, activation: nn):
#         super(EncoderBottomUp, self).__init__()

#         self.hid_dim = hid_dim
#         self.activation = activation

#         # modules
#         self.fc = nn.Linear(self.hid_dim, self.hid_dim)
#         self.bn = nn.BatchNorm1d(self.hid_dim)

#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         """
#         Forward the deterministic path bottom-up h_{l} = f( g(h_{l-1}), c).
#         Combine h_{l-1}, and c to compute h_l.

#         Args:
#             h: conditional embedding samples for layer h_{l-1}.
#             c: latent variable for the context.
#             first: if h is the data, map from h_dim to hid_dim.

#         Returns:
#             a new representation h_l for the data at layer l.
#         """
#         # reshape and affine
#         e = h.view(-1, self.n_features)
#         e = self.fc(e)
#         e = self.bn(e)
#         e = self.activation(e)
#         return e


# ########################################################################
# # INFERENCE NETWORK for z - q(z_l | z_{l+1}, c_l, h_l)
# class PosteriorZ(nn.Module):
#     """
#     Inference network q(z|h, z, c) gives approximate posterior over latent variables.
#     In this formulation there is no sharing of parameters between generative and inference model.

#     Attributes:
#         batch_size: batch of datasets.
#         sample_size: number of samples in conditioning dataset.
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         z_dim: number of features for latent variable.
#         h_dim: number of features for the embedded samples.
#         activation: specify activation.
#     """

#     def __init__(
#         self,
#         num_layers: int,
#         hid_dim: int,
#         c_dim: int,
#         z_dim: int,
#         h_dim: int,
#         activation: nn,
#     ):
#         super(PosteriorZ, self).__init__()

#         self.num_layers = num_layers
#         self.hid_dim = hid_dim

#         self.c_dim = c_dim
#         self.z_dim = z_dim
#         self.h_dim = h_dim

#         self.activation = activation
#         self.drop = nn.Dropout(0.1)

#         # modules
#         self.linear_h = nn.Linear(self.hid_dim, self.hid_dim)
#         self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
#         self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

#         self.residual_block = FCResBlock(
#             dim=self.hid_dim,
#             num_layers=self.num_layers,
#             activation=self.activation,
#             batch_norm=True,
#         )

#         self.linear_params = nn.Linear(self.hid_dim, 2 * self.z_dim)
#         self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

#     def forward(self, h, z, c, bs, ns):
#         """
#         Args:
#             h: conditional embedding for layer h_l.
#             z: moments for p(z_{l-1} | z_l, c).
#             c: context latent representation for layer l.
#         Returns:
#             moments for the approximate posterior over z
#             at layer l.
#         """
#         # combine h, z, and c
#         # embed h
#         eh = h.view(-1, self.hid_dim)
#         eh = self.linear_h(eh)
#         eh = eh.view(bs, -1, self.hid_dim)

#         # embed z if we have more than one stochastic layer
#         if z is not None:
#             ez = z.view(-1, self.z_dim)
#             ez = self.linear_z(ez)
#             ez = ez.view(bs, -1, self.hid_dim)
#         else:
#             ez = eh.new_zeros(eh.size())

#         # embed c and expand for broadcast addition
#         if c is not None:
#             ec = self.linear_c(c)
#             ec = ec.view(bs, -1, self.hid_dim).expand_as(eh)
#         else:
#             ec = eh.new_zeros(eh.size())

#         # sum and reshape
#         e = eh + ez + ec
#         e = e.view(-1, self.hid_dim)
#         e = self.activation(e)

#         # for layer in self.linear_block:
#         e = self.residual_block(e)

#         # affine transformation to parameters
#         e = self.linear_params(e)
#         # 'global' batch norm
#         e = e.view(-1, 1, 2 * self.z_dim)
#         e = self.bn_params(e)
#         e = e.view(-1, 2 * self.z_dim)

#         mean, logvar = e.chunk(2, dim=1)
#         return mean, logvar


# #######################################################################


# class PriorC(nn.Module):
#     """
#     Latent decoder for the context.

#     Attributes:
#         batch_size: batch of datasets.
#         sample_size: number of samples in conditioning dataset.
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         z_dim: number of features for latent variable.
#         h_dim: number of features for the embedded samples.
#         activation: specify activation.
#     """

#     def __init__(
#         self,
#         num_layers: int,
#         hid_dim: int,
#         c_dim: int,
#         z_dim: int,
#         h_dim: int,
#         activation: nn,
#         mode: str = "mean",
#         ladder=False,
#     ):
#         super(PriorC, self).__init__()

#         self.num_layers = num_layers
#         self.hid_dim = hid_dim
#         self.mode = mode

#         self.ladder = ladder
#         self.k = 2
#         if ladder:
#             self.k = 3

#         self.c_dim = c_dim
#         self.z_dim = z_dim

#         self.activation = activation

#         self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
#         self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

#         self.residual_block = FCResBlock(
#             dim=self.hid_dim,
#             num_layers=self.num_layers,
#             activation=self.activation,
#             batch_norm=True,
#         )

#         self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation, self.ladder)

#     def forward(self, z, c, bs, ns):
#         """
#         Combine z and rc using the attention mechanism and aggregating.
#         Embed z if we have more than one stochastic layer.

#         Args:
#             z: latent for samples at layer l+1.
#             rc: latent representations for context at layer l+1.
#             (batch_size, sample_size, hid_dim)

#         Returns:
#             Moments of the generative distribution for c at layer l.
#             Representations over context at layer l.
#         """
#         ec = self.linear_c(c)
#         ec = ec.view(bs, -1, self.hid_dim)

#         if z is not None:
#             ez = self.linear_z(z)
#             ez = ez.view(bs, ns, self.hid_dim)
#         else:
#             ez = ec.new_zeros(ec.size())

#         e = ez + ec.expand_as(ez)
#         e = e.view(-1, self.hid_dim)
#         e = self.activation(e)
#         e = self.residual_block(e)

#         # map e to the moments
#         e = e.view(bs, -1, self.hid_dim)
#         if self.mode == "mean":
#             a = e.mean(dim=1)
#         elif self.mode == "max":
#             a = e.max(dim=1)[0]

#         a = a.unsqueeze(1) + ec

#         if self.ladder:
#             mean, logvar, feats = self.postpool(a)
#             return mean, logvar, a, None, feats

#         mean, logvar = self.postpool(a)
#         return mean, logvar, a, None, None


# #######################################################################


# class PosteriorC(nn.Module):
#     """
#     Inference network q(c_l|c_{l+1}, z_{l+1}, H)
#     gives approximate posterior over latent context.
#     In this formulation there is no sharing of
#     parameters between generative and inference model.

#     Attributes:
#         batch_size: batch of datasets.
#         sample_size: number of samples in conditioning dataset.
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         z_dim: number of features for latent variable.
#         h_dim: number of features for the embedded samples.
#         activation: specify activation.
#     """

#     def __init__(
#         self,
#         num_layers: int,
#         hid_dim: int,
#         c_dim: int,
#         z_dim: int,
#         h_dim: int,
#         activation: nn,
#         mode: str = "mean",
#         ladder=False,
#     ):
#         super(PosteriorC, self).__init__()

#         self.num_layers = num_layers
#         self.hid_dim = hid_dim
#         self.mode = mode
#         self.ladder = ladder
#         self.k = 2
#         if ladder:
#             self.k = 3

#         self.c_dim = c_dim
#         self.z_dim = z_dim
#         self.h_dim = h_dim

#         self.activation = activation

#         self.linear_h = nn.Linear(self.hid_dim, self.hid_dim)
#         self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
#         self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

#         self.residual_block = FCResBlock(
#             dim=self.hid_dim,
#             num_layers=self.num_layers,
#             activation=self.activation,
#             batch_norm=True,
#         )

#         self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation, self.ladder)

#     def forward(self, h, z, c, bs, ns):
#         """
#         Combine h, z, and c to parameterize the moments of the approximate
#         posterior over c.
#         Combine z and rc using the attention mechanism and aggregating.
#         Embed z if we have more than one stochastic layer.

#         Args:
#             z: latent for samples at layer l+1.
#             rc: latent representations for context at layer l+1. (batch_size, sample_size, hid_dim)
#             h: embedding of the data at layer l.
#             first: chek if first layer to match data dimension.

#         Returns:
#             Moments of the approximate posterior distribution for c at layer l.
#             Representations over context at layer l.
#         """
#         eh = self.linear_h(h)
#         eh = eh.view(bs, ns, -1)

#         # map z to queries
#         if z is not None:
#             ez = self.linear_z(z)
#             ez = ez.view(bs, ns, -1)
#         else:
#             ez = eh.new_zeros(eh.size())

#         # map rc to keys and values
#         ec = c.view(-1, self.c_dim)
#         ec = self.linear_c(ec)
#         ec = ec.view(bs, -1, self.hid_dim)

#         # (b, sc, hdim)
#         e = eh + ez + ec.expand_as(eh)
#         e = e.view(-1, self.hid_dim)
#         e = self.activation(e)
#         e = self.residual_block(e)

#         e = e.view(bs, -1, self.hid_dim)
#         if self.mode == "mean":
#             a = e.mean(dim=1)
#         elif self.mode == "max":
#             a = e.max(dim=1)[0]

#         a = a.unsqueeze(1) + ec

#         # map to moments
#         if self.ladder:
#             mean, logvar, feats = self.postpool(a)
#             return mean, logvar, a, None, feats

#         mean, logvar = self.postpool(a)
#         return mean, logvar, a, None, None


# #######################################################################

# # Observation Decoder p(x| \bar z, \bar c)
# class ObservationDecoder(nn.Module):
#     """
#     Parameterize (Bernoulli) observation model p(x | z, c).

#     Attributes:
#         batch_size: batch of datasets.
#         sample_size: number of samples in conditioning dataset.
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         z_dim: number of features for latent variable.
#         h_dim: number of features for the embedded samples.
#         n_stochastic_z: number of inference layers for z.
#         n_stochastic_c: number of inference layers for c.
#         activation: specify activation.
#     """

#     def __init__(
#         self,
#         num_layers: int,
#         hid_dim: int,
#         c_dim: int,
#         z_dim: int,
#         h_dim: int,
#         n_stochastic_z: int,
#         n_stochastic_c: int,
#         activation: nn,
#     ):
#         super(ObservationDecoder, self).__init__()

#         self.num_layers = num_layers
#         self.hid_dim = hid_dim

#         self.c_dim = c_dim
#         self.z_dim = z_dim
#         self.h_dim = h_dim

#         self.n_stochastic_z = n_stochastic_z
#         self.n_stochastic_c = n_stochastic_c
#         self.activation = activation

#         self.linear_zs = nn.Linear(self.n_stochastic_z * self.z_dim, self.hid_dim)
#         self.linear_cs = nn.Linear(self.n_stochastic_c * self.c_dim, self.hid_dim)

#         # self.linear_initial = nn.Linear(self.hid_dim, 256 * 7 * 7)
#         self.linear_initial = nn.Linear(self.hid_dim, 256 * 4 * 4)

#         self.conv_layers = nn.ModuleList(
#             [
#                 Conv2d3x3(256, 256),
#                 Conv2d3x3(256, 256),
#                 nn.ConvTranspose2d(256, 256, kernel_size=4, stride=3),
#                 Conv2d3x3(256, 128),
#                 Conv2d3x3(128, 128),
#                 nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
#                 Conv2d3x3(128, 64),
#                 nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=1),
#                 nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#             ]
#         )

#         self.bn_layers = nn.ModuleList(
#             [
#                 nn.BatchNorm2d(256),
#                 nn.BatchNorm2d(256),
#                 nn.BatchNorm2d(256),
#                 nn.BatchNorm2d(128),
#                 nn.BatchNorm2d(128),
#                 nn.BatchNorm2d(128),
#                 nn.BatchNorm2d(64),
#                 nn.BatchNorm2d(64),
#                 nn.BatchNorm2d(64),
#             ]
#         )

#         self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, zs, cs, bs, ns):
#         """
#         Args:
#             zs: collection of all latent variables in all layers.
#             cs: collection of all latent variables for context in all layers.

#         Returns:
#             Moments of observation model.
#         """
#         # use z for all layers
#         # print(zs.size(), cs.size())
#         ezs = self.linear_zs(zs)
#         ezs = ezs.view(bs, ns, -1)

#         if cs is not None:
#             ecs = self.linear_cs(cs)
#             ecs = ecs.view(bs, -1, self.hid_dim).expand_as(ezs)
#         else:
#             ecs = ezs.new_zeros(ezs.size())

#         # concatenate zs and c
#         e = ezs + ecs
#         e = self.activation(e)
#         e = e.view(-1, self.hid_dim)

#         e = self.linear_initial(e)
#         # e = e.view(-1, 256, 7, 7)
#         e = e.view(-1, 256, 4, 4)

#         for conv, bn in zip(self.conv_layers, self.bn_layers):
#             e = conv(e)
#             e = bn(e)
#             e = self.activation(e)

#         e = self.conv_final(e)
#         # moments for the Bernoulli
#         p = torch.sigmoid(e)
#         return p


# # class ObservationDecoderNew(nn.Module):
# #     """
# #     Parameterize (Bernoulli) observation model p(x | z, c).

# #     Attributes:
# #         batch_size: batch of datasets.
# #         sample_size: number of samples in conditioning dataset.
# #         num_layers: number of layers in residual block.
# #         hid_dim: number of features for layer.
# #         c_dim: number of features for latent summary.
# #         z_dim: number of features for latent variable.
# #         h_dim: number of features for the embedded samples.
# #         n_stochastic_z: number of inference layers for z.
# #         n_stochastic_c: number of inference layers for c.
# #         activation: specify activation.
# #     """

# #     def __init__(
# #         self,
# #         num_layers: int,
# #         hid_dim: int,
# #         c_dim: int,
# #         z_dim: int,
# #         h_dim: int,
# #         n_stochastic_z: int,
# #         n_stochastic_c: int,
# #         activation: nn,
# #     ):
# #         super(ObservationDecoderNew, self).__init__()

# #         self.num_layers = num_layers
# #         self.hid_dim = hid_dim

# #         self.c_dim = c_dim
# #         self.z_dim = z_dim
# #         self.h_dim = h_dim

# #         self.n_stochastic_z = n_stochastic_z
# #         self.n_stochastic_c = n_stochastic_c
# #         self.activation = activation

# #         self.linear_zcs = nn.Linear(self.n_stochastic_z * self.c_dim, self.hid_dim)

# #         self.linear_initial = nn.Linear(self.hid_dim, 256 * 4 * 4)

# #         self.conv_layers = nn.ModuleList(
# #             [
# #                 Conv2d3x3(256, 256),
# #                 Conv2d3x3(256, 256),
# #                 nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
# #                 Conv2d3x3(256, 128),
# #                 Conv2d3x3(128, 128),
# #                 nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
# #                 Conv2d3x3(128, 64),
# #                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
# #                 nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
# #             ]
# #         )

# #         self.bn_layers = nn.ModuleList(
# #             [
# #                 nn.BatchNorm2d(256),
# #                 nn.BatchNorm2d(256),
# #                 nn.BatchNorm2d(256),
# #                 nn.BatchNorm2d(128),
# #                 nn.BatchNorm2d(128),
# #                 nn.BatchNorm2d(128),
# #                 nn.BatchNorm2d(64),
# #                 nn.BatchNorm2d(64),
# #                 nn.BatchNorm2d(64),
# #             ]
# #         )

# #         self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

# #     def forward(self, out, bs, ns):
# #         """
# #         Args:
# #             zs: collection of all latent variables in all layers.
# #             cs: collection of all latent variables for context in all layers.

# #         Returns:
# #             Moments of observation model.
# #         """
# #         # use z for all layers
# #         cs_lst = [out["cqs"][i].unsqueeze(1) for i in range(self.n_stochastic_c)]
# #         cs = torch.cat(cs_lst, dim=1)
# #         cs = cs.repeat_interleave(repeats=ns, dim=0)

# #         zs_lst = [out["zqs"][i].unsqueeze(1) for i in range(self.n_stochastic_z)]
# #         zs = torch.cat(zs_lst, dim=1)

# #         # (bs*ns, l, h)
# #         e = zs + cs
# #         e = e.view(bs*ns, -1)

# #         # ezs = self.linear_zs(zs)
# #         # ezs = ezs.view(bs, ns, -1)

# #         # if cs is not None:
# #         #     ecs = self.linear_cs(cs)
# #         #     ecs = ecs.view(bs, -1, self.hid_dim).expand_as(ezs)
# #         # else:
# #         #     ecs = ezs.new_zeros(ezs.size())

# #         # concatenate zs and c
# #         # e = ezs + ecs

# #         e = self.activation(e)
# #         e = self.linear_zcs(e)
# #         e = e.view(-1, self.hid_dim)

# #         e = self.linear_initial(e)
# #         e = e.view(-1, 256, 4, 4)

# #         for conv, bn in zip(self.conv_layers, self.bn_layers):
# #             e = conv(e)
# #             e = bn(e)
# #             e = self.activation(e)

# #         e = self.conv_final(e)
# #         # moments for the Bernoulli
# #         p = torch.sigmoid(e)
# #         return p


# class RelationNetwork(nn.Module):
#     """
#     Compute relations between the n*k samples.

#     Input: encoding (Hs, Ys)
#     Output: class representations w_n
#     """

#     def __init__(
#         self,
#         batch_size: int,
#         sample_size: int,
#         num_layers: int,
#         hid_dim: int,
#         activation: nn,
#     ):
#         super(RelationNetwork, self).__init__()
#         self.batch_size = batch_size
#         self.sample_size = sample_size

#         self.linear = nn.Linear(2 * hid_dim, hid_dim)
#         self.relation_net = FCResBlock(
#             dim=hid_dim, num_layers=num_layers, activation=activation, batch_norm=True
#         )

#     def forward(self, erc, eh):
#         # erc (b, q, h)
#         # eh  (b, s, h)

#         b, q, h = erc.size()
#         _, s, _ = eh.size()

#         # repeat
#         # copy n*k times each element in the dataset
#         # [1, 2, 3] n=3, k=1
#         # [1, 1, 1, 2, 2, 2, 3, 3, 3]
#         left = erc.repeat_interleave(repeats=s, dim=1)

#         # copy the dataset n*k times
#         # [1, 2, 3] n=3, k=1
#         # [1, 2, 3, 1, 2, 3, 1, 2, 3]
#         right = eh.repeat(1, q, 1)

#         e = torch.cat([left, right], -1)
#         print(e.size())

#         e = e.view(-1, 2 * h)
#         e = self.linear(e)
#         e = self.relation_net(e)
#         print(e.size())

#         # (b, q, s, h)
#         e = e.view(b, q, s, h)
#         print(e.size())
#         # sum over the n*k relations for each sample
#         # (b, q, dim)
#         e = e.sum(2) / s
#         return e






# log var = log sigma^2 = 2 log sigma
def f_std(logvar):
    return torch.exp(0.5 * logvar)


class FCResBlock(nn.Module):
    """
    Residual block.

    Attributes:
        dim: layer width.
        num_layers: number of layers in residual block.
        activation: specify activation.
        batch_norm: use or not use batch norm.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        activation: nn,
        batch_norm: bool = False,
        dropout: bool = False,
    ):
        super(FCResBlock, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.block = []
        for _ in range(self.num_layers):
            layer = [nn.Linear(dim, dim)]
            if self.batch_norm:
                layer.append(nn.BatchNorm1d(num_features=dim))
            if self.dropout:
                layer.append(nn.Dropout(p=0.1))
            self.block.append(nn.ModuleList(layer))
        self.block = nn.ModuleList(self.block)

    def forward(self, x):
        e = x + 0
        for l, layer in enumerate(self.block):
            for i in range(len(layer)):
                e = layer[i](e)
            if l < (len(self.block) - 1):
                e = self.activation(e)
        # residual
        return self.activation(e + x)


class Conv2d3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super(Conv2d3x3, self).__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )

    def forward(self, x):
        return self.conv(x)


class SharedConvEncoder(nn.Module):
    def __init__(self, nc: int, activation: nn):
        super(SharedConvEncoder, self).__init__()
        self.activation = activation

        self.conv_layers = nn.ModuleList(
            [   
                Conv2d3x3(in_channels=1, out_channels=nc//2),
                Conv2d3x3(in_channels=nc//2, out_channels=nc//2),
                Conv2d3x3(in_channels=nc//2, out_channels=nc//2, downsample=True),
                Conv2d3x3(in_channels=nc//2, out_channels=nc),
                Conv2d3x3(in_channels=nc, out_channels=nc),
                Conv2d3x3(in_channels=nc, out_channels=nc, downsample=True),
                # shape is now (-1, 64, 14 , 14)
                Conv2d3x3(in_channels=nc, out_channels=2 * nc),
                Conv2d3x3(in_channels=2 * nc, out_channels=2 * nc),
                Conv2d3x3(in_channels=2 * nc, out_channels=2 * nc, downsample=True),
                # shape is now (-1, 128, 7, 7)
                Conv2d3x3(in_channels=2 * nc, out_channels=4 * nc),
                Conv2d3x3(in_channels=4 * nc, out_channels=4 * nc),
                Conv2d3x3(in_channels=4 * nc, out_channels=4 * nc, downsample=True)
                # shape is now (-1, 256, 4, 4)
            ]
        )

        self.bn_layers = nn.ModuleList(
            [
                nn.BatchNorm2d(num_features=nc//2),
                nn.BatchNorm2d(num_features=nc//2),
                nn.BatchNorm2d(num_features=nc//2),
                nn.BatchNorm2d(num_features=nc),
                nn.BatchNorm2d(num_features=nc),
                nn.BatchNorm2d(num_features=nc),
                nn.BatchNorm2d(num_features=2 * nc),
                nn.BatchNorm2d(num_features=2 * nc),
                nn.BatchNorm2d(num_features=2 * nc),
                nn.BatchNorm2d(num_features=4 * nc),
                nn.BatchNorm2d(num_features=4 * nc),
                nn.BatchNorm2d(num_features=4 * nc),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.view(-1, 1,50,50)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h)
            h = self.activation(h)
        #     print(h.shape)
        # exit()
        return h


class PreProj(nn.Module):
    def __init__(self, hid_dim: int, n_features: int, activation: nn):
        super(PreProj, self).__init__()

        self.n_features = n_features
        self.hid_dim = hid_dim
        self.activation = activation

        # modules
        self.fc = nn.Linear(self.n_features, self.hid_dim)
        self.bn = nn.BatchNorm1d(self.hid_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # reshape and affine
        e = h.view(-1, self.n_features)
        e = self.fc(e)
        e = self.bn(e)
        e = self.activation(e)
        return e


class PostPool(nn.Module):
    def __init__(self, hid_dim: int, c_dim: int, activation: nn, ladder=False):
        super(PostPool, self).__init__()

        self.hid_dim = hid_dim
        self.c_dim = c_dim
        self.activation = activation
        self.drop = nn.Dropout(0.1)
        self.ladder = ladder

        self.k = 2
        if ladder:
            self.k = 3

        # modules
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(self.hid_dim, self.hid_dim),
                nn.Linear(self.hid_dim, self.hid_dim),
            ]
        )
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(self.hid_dim), nn.BatchNorm1d(self.hid_dim)]
        )

        self.linear_params = nn.Linear(self.hid_dim, self.k * self.c_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        e = e.view(-1, self.hid_dim)
        for fc, bn in zip(self.linear_layers, self.bn_layers):
            e = fc(e)
            e = bn(e)
            e = self.activation(e)

        # affine transformation to parameters
        e = self.linear_params(e)
        # 'global' batch norm
        e = e.view(-1, 1, self.k * self.c_dim)
        e = self.bn_params(e)
        e = e.view(-1, self.k * self.c_dim)

        if self.ladder:
            mean, logvar, feats = e.chunk(self.k, dim=1)
            return mean, logvar, feats
        else:
            mean, logvar = e.chunk(self.k, dim=1)
            return mean, logvar


########################################################################
class StatistiC(nn.Module):
    """
    Compute the statistic q(c | X).
    Encode X, preprocess, aggregate, postprocess.
    Representation for context at layer L.

    Attributes:
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        n_features: number of features in output of the shared encoder (256 * 4 * 4).
        activation: specify activation.
    """

    def __init__(
        self,
        hid_dim: int,
        c_dim: int,
        n_features: int,
        activation: nn,
        dropout_sample: bool = False,
        mode: str = "mean",
    ):
        super(StatistiC, self).__init__()

        self.c_dim = c_dim
        self.mode = mode
        self.hid_dim = hid_dim
        self.activation = activation

        self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation)

    def forward(self, h, bs, ns):
        # aggregate samples
        h = h.view(bs, -1, self.hid_dim)
        if self.mode == "mean":
            a = h.mean(dim=1)
        elif self.mode == "max":
            a = h.max(dim=1)[0]
        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, None


########################################################################
# Prior for z - p(z_l | z_{l+1}, c_l)
class PriorZ(nn.Module):
    """
    Latent decoder for the sample.

    Attributes:
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        activation: nn,
    ):
        super(PriorZ, self).__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.c_dim = c_dim
        self.z_dim = z_dim

        self.activation = activation
        self.drop = nn.Dropout(0.1)

        # modules
        self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.residual_block = FCResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.linear_params = nn.Linear(self.hid_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, z, c, bs, ns):
        """
        Args:
            z: latent for samples at layer l+1.
            c: latent representations for context at layer l.
        Returns:
            Moments of the generative distribution for z at layer l.
        """
        # combine z and c
        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.linear_z(ez)
            ez = ez.view(bs, -1, self.hid_dim)
        else:
            ez = c.new_zeros((bs, ns, self.hid_dim))

        if c is not None:
            ec = self.linear_c(c)
            ec = ec.view(bs, -1, self.hid_dim).expand_as(ez)
        else:
            ec = ez.new_zeros((bs, ns, self.hid_dim))

        # sum and reshape
        e = ez + ec
        e = e.view(-1, self.hid_dim)
        e = self.activation(e)

        # for layer in self.linear_block:
        e = self.residual_block(e)

        # affine transformation to parameters
        e = self.linear_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)
        # e = self.drop(e)

        mean, logvar = e.chunk(2, dim=1)
        return mean, logvar


########################################################################
class EncoderBottomUp(nn.Module):
    """
    Bottom-up deterministic Inference network h_l = g(h_{l-1}, c) return representations
    for x=h_1 at each layer.

    Attributes:
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        activation: specify activation.

    """

    def __init__(self, num_layers: int, hid_dim: int, activation: nn):
        super(EncoderBottomUp, self).__init__()

        self.hid_dim = hid_dim
        self.activation = activation

        # modules
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.bn = nn.BatchNorm1d(self.hid_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward the deterministic path bottom-up h_{l} = f( g(h_{l-1}), c).
        Combine h_{l-1}, and c to compute h_l.

        Args:
            h: conditional embedding samples for layer h_{l-1}.
            c: latent variable for the context.
            first: if h is the data, map from h_dim to hid_dim.

        Returns:
            a new representation h_l for the data at layer l.
        """
        # reshape and affine
        e = h.view(-1, self.n_features)
        e = self.fc(e)
        e = self.bn(e)
        e = self.activation(e)
        return e


########################################################################
# INFERENCE NETWORK for z - q(z_l | z_{l+1}, c_l, h_l)
class PosteriorZ(nn.Module):
    """
    Inference network q(z|h, z, c) gives approximate posterior over latent variables.
    In this formulation there is no sharing of parameters between generative and inference model.

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        activation: nn,
    ):
        super(PosteriorZ, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.activation = activation
        self.drop = nn.Dropout(0.1)

        # modules
        self.linear_h = nn.Linear(self.hid_dim, self.hid_dim)
        self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.residual_block = FCResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.linear_params = nn.Linear(self.hid_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, h, z, c, bs, ns):
        """
        Args:
            h: conditional embedding for layer h_l.
            z: moments for p(z_{l-1} | z_l, c).
            c: context latent representation for layer l.
        Returns:
            moments for the approximate posterior over z
            at layer l.
        """
        # combine h, z, and c
        # embed h
        eh = h.view(-1, self.hid_dim)
        eh = self.linear_h(eh)
        eh = eh.view(bs, -1, self.hid_dim)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.linear_z(ez)
            ez = ez.view(bs, -1, self.hid_dim)
        else:
            ez = eh.new_zeros(eh.size())

        # embed c and expand for broadcast addition
        if c is not None:
            ec = self.linear_c(c)
            ec = ec.view(bs, -1, self.hid_dim).expand_as(eh)
        else:
            ec = eh.new_zeros(eh.size())

        # sum and reshape
        e = eh + ez + ec
        e = e.view(-1, self.hid_dim)
        e = self.activation(e)

        # for layer in self.linear_block:
        e = self.residual_block(e)

        # affine transformation to parameters
        e = self.linear_params(e)
        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e.chunk(2, dim=1)
        return mean, logvar


#######################################################################


class PriorC(nn.Module):
    """
    Latent decoder for the context.

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        activation: nn,
        mode: str = "mean",
        ladder=False,
    ):
        super(PriorC, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.mode = mode

        self.ladder = ladder
        self.k = 2
        if ladder:
            self.k = 3

        self.c_dim = c_dim
        self.z_dim = z_dim

        self.activation = activation

        self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.residual_block = FCResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation, self.ladder)

    def forward(self, z, c, bs, ns):
        """
        Combine z and rc using the attention mechanism and aggregating.
        Embed z if we have more than one stochastic layer.

        Args:
            z: latent for samples at layer l+1.
            rc: latent representations for context at layer l+1.
            (batch_size, sample_size, hid_dim)

        Returns:
            Moments of the generative distribution for c at layer l.
            Representations over context at layer l.
        """
        ec = self.linear_c(c)
        ec = ec.view(bs, -1, self.hid_dim)

        if z is not None:
            ez = self.linear_z(z)
            ez = ez.view(bs, ns, self.hid_dim)
        else:
            ez = ec.new_zeros(ec.size())

        e = ez + ec.expand_as(ez)
        e = e.view(-1, self.hid_dim)
        e = self.activation(e)
        e = self.residual_block(e)

        # map e to the moments
        e = e.view(bs, -1, self.hid_dim)
        if self.mode == "mean":
            a = e.mean(dim=1)
        elif self.mode == "max":
            a = e.max(dim=1)[0]

        a = a.unsqueeze(1) + ec

        if self.ladder:
            mean, logvar, feats = self.postpool(a)
            return mean, logvar, a, None, feats

        mean, logvar = self.postpool(a)
        return mean, logvar, a, None, None


#######################################################################


class PosteriorC(nn.Module):
    """
    Inference network q(c_l|c_{l+1}, z_{l+1}, H)
    gives approximate posterior over latent context.
    In this formulation there is no sharing of
    parameters between generative and inference model.

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        activation: nn,
        mode: str = "mean",
        ladder=False,
    ):
        super(PosteriorC, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.mode = mode
        self.ladder = ladder
        self.k = 2
        if ladder:
            self.k = 3

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.activation = activation

        self.linear_h = nn.Linear(self.hid_dim, self.hid_dim)
        self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.residual_block = FCResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation, self.ladder)

    def forward(self, h, z, c, bs, ns):
        """
        Combine h, z, and c to parameterize the moments of the approximate
        posterior over c.
        Combine z and rc using the attention mechanism and aggregating.
        Embed z if we have more than one stochastic layer.

        Args:
            z: latent for samples at layer l+1.
            rc: latent representations for context at layer l+1. (batch_size, sample_size, hid_dim)
            h: embedding of the data at layer l.
            first: chek if first layer to match data dimension.

        Returns:
            Moments of the approximate posterior distribution for c at layer l.
            Representations over context at layer l.
        """
        eh = self.linear_h(h)
        eh = eh.view(bs, ns, -1)

        # map z to queries
        if z is not None:
            ez = self.linear_z(z)
            ez = ez.view(bs, ns, -1)
        else:
            ez = eh.new_zeros(eh.size())

        # map rc to keys and values
        ec = c.view(-1, self.c_dim)
        ec = self.linear_c(ec)
        ec = ec.view(bs, -1, self.hid_dim)

        # (b, sc, hdim)
        e = eh + ez + ec.expand_as(eh)
        e = e.view(-1, self.hid_dim)
        e = self.activation(e)
        e = self.residual_block(e)

        e = e.view(bs, -1, self.hid_dim)
        if self.mode == "mean":
            a = e.mean(dim=1)
        elif self.mode == "max":
            a = e.max(dim=1)[0]

        a = a.unsqueeze(1) + ec

        # map to moments
        if self.ladder:
            mean, logvar, feats = self.postpool(a)
            return mean, logvar, a, None, feats

        mean, logvar = self.postpool(a)
        return mean, logvar, a, None, None


#######################################################################

# Observation Decoder p(x| \bar z, \bar c)
class ObservationDecoder(nn.Module):
    """
    Parameterize (Bernoulli) observation model p(x | z, c).

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        n_stochastic_z: number of inference layers for z.
        n_stochastic_c: number of inference layers for c.
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        n_stochastic_z: int,
        n_stochastic_c: int,
        activation: nn,
    ):
        super(ObservationDecoder, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.n_stochastic_z = n_stochastic_z
        self.n_stochastic_c = n_stochastic_c
        self.activation = activation

        self.linear_zs = nn.Linear(self.n_stochastic_z * self.z_dim, self.hid_dim)
        self.linear_cs = nn.Linear(self.n_stochastic_c * self.c_dim, self.hid_dim)

        # self.linear_initial = nn.Linear(self.hid_dim, 256 * 7 * 7)
        self.linear_initial = nn.Linear(self.hid_dim, 256 * 4 * 4)

        self.conv_layers = nn.ModuleList(
            [   
                Conv2d3x3(256, 256),
                Conv2d3x3(256, 256),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                Conv2d3x3(256, 128),
                Conv2d3x3(128, 128),
                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                Conv2d3x3(128, 64),
                nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0),
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                Conv2d3x3(64, 32),
                nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ]
        )

        self.bn_layers = nn.ModuleList(
            [
                nn.BatchNorm2d(256),
                nn.BatchNorm2d(256),
                nn.BatchNorm2d(256),
                nn.BatchNorm2d(128),
                nn.BatchNorm2d(128),
                nn.BatchNorm2d(128),
                nn.BatchNorm2d(64),
                nn.BatchNorm2d(64),
                nn.BatchNorm2d(64),
                nn.BatchNorm2d(32),
                nn.BatchNorm2d(32),
                nn.BatchNorm2d(32),
            ]
        )

        self.conv_final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, zs, cs, bs, ns):
        """
        Args:
            zs: collection of all latent variables in all layers.
            cs: collection of all latent variables for context in all layers.

        Returns:
            Moments of observation model.
        """
        # use z for all layers
        # print(zs.size(), cs.size())
        ezs = self.linear_zs(zs)
        ezs = ezs.view(bs, ns, -1)

        if cs is not None:
            ecs = self.linear_cs(cs)
            ecs = ecs.view(bs, -1, self.hid_dim).expand_as(ezs)
        else:
            ecs = ezs.new_zeros(ezs.size())

        # concatenate zs and c
        e = ezs + ecs
        e = self.activation(e)
        e = e.view(-1, self.hid_dim)

        e = self.linear_initial(e)
        # e = e.view(-1, 256, 7, 7)
        e = e.view(-1, 256, 4, 4)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = self.activation(e)

        e = self.conv_final(e)
        # moments for the Bernoulli
        p = torch.sigmoid(e)
        return p


# class ObservationDecoderNew(nn.Module):
#     """
#     Parameterize (Bernoulli) observation model p(x | z, c).

#     Attributes:
#         batch_size: batch of datasets.
#         sample_size: number of samples in conditioning dataset.
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         z_dim: number of features for latent variable.
#         h_dim: number of features for the embedded samples.
#         n_stochastic_z: number of inference layers for z.
#         n_stochastic_c: number of inference layers for c.
#         activation: specify activation.
#     """

#     def __init__(
#         self,
#         num_layers: int,
#         hid_dim: int,
#         c_dim: int,
#         z_dim: int,
#         h_dim: int,
#         n_stochastic_z: int,
#         n_stochastic_c: int,
#         activation: nn,
#     ):
#         super(ObservationDecoderNew, self).__init__()

#         self.num_layers = num_layers
#         self.hid_dim = hid_dim

#         self.c_dim = c_dim
#         self.z_dim = z_dim
#         self.h_dim = h_dim

#         self.n_stochastic_z = n_stochastic_z
#         self.n_stochastic_c = n_stochastic_c
#         self.activation = activation

#         self.linear_zcs = nn.Linear(self.n_stochastic_z * self.c_dim, self.hid_dim)

#         self.linear_initial = nn.Linear(self.hid_dim, 256 * 4 * 4)

#         self.conv_layers = nn.ModuleList(
#             [
#                 Conv2d3x3(256, 256),
#                 Conv2d3x3(256, 256),
#                 nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
#                 Conv2d3x3(256, 128),
#                 Conv2d3x3(128, 128),
#                 nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
#                 Conv2d3x3(128, 64),
#                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#                 nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#             ]
#         )

#         self.bn_layers = nn.ModuleList(
#             [
#                 nn.BatchNorm2d(256),
#                 nn.BatchNorm2d(256),
#                 nn.BatchNorm2d(256),
#                 nn.BatchNorm2d(128),
#                 nn.BatchNorm2d(128),
#                 nn.BatchNorm2d(128),
#                 nn.BatchNorm2d(64),
#                 nn.BatchNorm2d(64),
#                 nn.BatchNorm2d(64),
#             ]
#         )

#         self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, out, bs, ns):
#         """
#         Args:
#             zs: collection of all latent variables in all layers.
#             cs: collection of all latent variables for context in all layers.

#         Returns:
#             Moments of observation model.
#         """
#         # use z for all layers
#         cs_lst = [out["cqs"][i].unsqueeze(1) for i in range(self.n_stochastic_c)]
#         cs = torch.cat(cs_lst, dim=1)
#         cs = cs.repeat_interleave(repeats=ns, dim=0)

#         zs_lst = [out["zqs"][i].unsqueeze(1) for i in range(self.n_stochastic_z)]
#         zs = torch.cat(zs_lst, dim=1)

#         # (bs*ns, l, h)
#         e = zs + cs
#         e = e.view(bs*ns, -1)

#         # ezs = self.linear_zs(zs)
#         # ezs = ezs.view(bs, ns, -1)

#         # if cs is not None:
#         #     ecs = self.linear_cs(cs)
#         #     ecs = ecs.view(bs, -1, self.hid_dim).expand_as(ezs)
#         # else:
#         #     ecs = ezs.new_zeros(ezs.size())

#         # concatenate zs and c
#         # e = ezs + ecs

#         e = self.activation(e)
#         e = self.linear_zcs(e)
#         e = e.view(-1, self.hid_dim)

#         e = self.linear_initial(e)
#         e = e.view(-1, 256, 4, 4)

#         for conv, bn in zip(self.conv_layers, self.bn_layers):
#             e = conv(e)
#             e = bn(e)
#             e = self.activation(e)

#         e = self.conv_final(e)
#         # moments for the Bernoulli
#         p = torch.sigmoid(e)
#         return p


class RelationNetwork(nn.Module):
    """
    Compute relations between the n*k samples.

    Input: encoding (Hs, Ys)
    Output: class representations w_n
    """

    def __init__(
        self,
        batch_size: int,
        sample_size: int,
        num_layers: int,
        hid_dim: int,
        activation: nn,
    ):
        super(RelationNetwork, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.linear = nn.Linear(2 * hid_dim, hid_dim)
        self.relation_net = FCResBlock(
            dim=hid_dim, num_layers=num_layers, activation=activation, batch_norm=True
        )

    def forward(self, erc, eh):
        # erc (b, q, h)
        # eh  (b, s, h)

        b, q, h = erc.size()
        _, s, _ = eh.size()

        # repeat
        # copy n*k times each element in the dataset
        # [1, 2, 3] n=3, k=1
        # [1, 1, 1, 2, 2, 2, 3, 3, 3]
        left = erc.repeat_interleave(repeats=s, dim=1)

        # copy the dataset n*k times
        # [1, 2, 3] n=3, k=1
        # [1, 2, 3, 1, 2, 3, 1, 2, 3]
        right = eh.repeat(1, q, 1)

        e = torch.cat([left, right], -1)
        print(e.size())

        e = e.view(-1, 2 * h)
        e = self.linear(e)
        e = self.relation_net(e)
        print(e.size())

        # (b, q, s, h)
        e = e.view(b, q, s, h)
        print(e.size())
        # sum over the n*k relations for each sample
        # (b, q, dim)
        e = e.sum(2) / s
        return e
