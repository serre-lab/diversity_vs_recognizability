import torch.nn as nn
import torch
from .attention_function import read_transformer, write_transformer,\
    compute_theta_general
import numpy as np
import torch.distributions as D


class VaeStn(nn.Module):
    """
        Learning the scale and the translation parameters of the affine transformation
    """

    def __init__(self, args):
        super(VaeStn, self).__init__()
        self.args = args
        self.z_size = args.z_size
        self.dataset = args.dataset
        self.write_size = args.write_size
        self.read_size = args.read_size
        self.input_shape = args.input_shape
        # self.imagette_size = args.imagette_size
        self.dim_lstm = args.lstm_size
        self.loc_size = 100
        self.time_steps = args.time_step
        self.model_name = args.model_name
        # encoder

        if self.model_name != 'vae_rec':
            self.fc_loc = nn.Sequential(
                nn.Linear(self.loc_size, 64),
                nn.ReLU(True),
                nn.Linear(64, 32),
                nn.ReLU(True),
                nn.Linear(32, 6)
            )


            self.fc_loc[-1].weight.data.zero_()
            self.fc_loc[-1].bias.data.zero_()
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float))

        if args.exemplar:
            #self.additional_dim = self.imagette_size[-1]*self.imagette_size[-1]
            self.additional_dim = np.prod(self.read_size)
        else:
            self.additional_dim = 0

        self.q_z_mean = nn.Sequential(
            nn.Linear(self.additional_dim + np.prod(self.read_size) * 2 + self.dim_lstm, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, self.z_size)
        )

        self.q_z_var = nn.Sequential(
            nn.Linear(self.additional_dim + np.prod(self.read_size) * 2 + self.dim_lstm, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, self.z_size)
        )


        self.to_imagette = nn.Sequential(
            nn.Linear(self.dim_lstm-self.loc_size, 200),
            nn.ReLU(True),
            nn.Linear(200, np.prod(self.write_size))
        )
        self.rnn = nn.LSTMCell(self.z_size+self.additional_dim, self.dim_lstm)

    @staticmethod
    def reparameterize(mu, logvar, std=False):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        eps = torch.randn_like(mu)
        if std:
            std = logvar
        else:
            std = torch.exp(0.5 * logvar)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x, exemplar=None, low_memory=False):
        read_size = (x.size(0),  *self.read_size)
        write_size = (x.size(0),  *self.write_size)
        mus, log_vars = [0] * self.time_steps, [0] * self.time_steps
        h = torch.zeros(x.size(0), self.dim_lstm).to(x.device)
        c = torch.zeros_like(h)
        x_size = x.size()
        if not low_memory:
            all_bu_att = [torch.zeros(read_size).to(self.args.device)] * (self.time_steps+1)
            decoded_imagette = [torch.zeros(write_size).to(self.args.device)] * (self.time_steps+1)
            all_td_att = [torch.zeros_like(x)] * (self.time_steps+1)
            all_gene = [torch.zeros_like(x)] * (self.time_steps+1)
        #theta = self.fc_loc(h[:, :self.loc_size])
        theta = torch.zeros(x.size(0), 6).to(self.args.device)
        theta[:, 0] = 1
        theta[:, 3] = 1
        canvas = torch.zeros_like(x).to(x.device)

        for t in range(self.time_steps):
            x_hat = x - torch.sigmoid(canvas)

            # Bu attention

            theta_mat_bu = compute_theta_general(theta, invert=False)
            r_hat = read_transformer(theta_mat_bu, x_hat, read_size)
            r = read_transformer(theta_mat_bu, x, read_size)

            if not low_memory:
                all_bu_att[t+1] = r.detach()

            if exemplar is not None:
                r_exemplar = read_transformer(theta_mat_bu, exemplar, read_size)
                r = torch.cat([r_hat, r, r_exemplar], dim=1)
            else:
                r = torch.cat([r_hat, r], dim=1)
            state = torch.cat([h, r.view(x.size(0), -1)], dim=1)

            mus[t], log_vars[t] = self.q_z_mean(state), self.q_z_var(state)
            z = self.reparameterize(mus[t], log_vars[t])
            if exemplar is not None:
                z = torch.cat([z, r_exemplar.view(x.size(0), -1)], dim=1)



            imagette = h[:, self.loc_size:]
            imagette = self.to_imagette(imagette).view(write_size)
            # td attention
            if self.model_name == 'vae_rec':
                td_att = imagette
            else:
                theta_mat_td = compute_theta_general(theta, invert=True)
                td_att = write_transformer(theta_mat_td, imagette, x_size)
            h, c = self.rnn(z, (h, c))
            theta = self.fc_loc(h[:, :self.loc_size])
            canvas = canvas + td_att
            if not low_memory:
                all_gene[t+1] = torch.sigmoid(canvas)
                all_td_att[t + 1] = td_att.detach()
                decoded_imagette[t + 1] = imagette.detach()

        reco = torch.sigmoid(canvas)

        if not low_memory:
            return torch.stack(mus, dim=-1), torch.stack(log_vars, dim=-1), reco, torch.stack(all_bu_att, dim=1), \
                torch.stack(all_td_att, dim=1), torch.stack(all_gene, dim=1), torch.stack(decoded_imagette, dim=1)
        else:
            return torch.stack(mus, dim=-1), torch.stack(log_vars, dim=-1), reco, None, None, None, None

    def generate(self, n_samples, exemplar=None, low_memory=False):
        read_size = (n_samples, *self.read_size)
        write_size = (n_samples, *self.write_size)
        x_size = (n_samples, *self.input_shape)
        if not low_memory:
            all_td_att = [torch.zeros(x_size).to(self.args.device)] * (self.time_steps+1)
            all_gene = [torch.zeros(x_size).to(self.args.device)] * (self.time_steps+1)
        canvas = torch.zeros(x_size).to(self.args.device)
        h = torch.zeros(n_samples, self.dim_lstm).to(self.args.device)
        c = torch.zeros_like(h)
        #theta = self.fc_loc(h[:, :self.loc_size])
        theta = torch.zeros(n_samples, 6).to(self.args.device)
        theta[:, 0] = 1
        theta[:, 3] = 1

        for t in range(self.time_steps):
            z = torch.randn(n_samples, self.args.z_size).to(self.args.device)

            if exemplar is not None:
                theta_mat_bu = compute_theta_general(theta, invert=False)
                r_exemplar = read_transformer(theta_mat_bu, exemplar, read_size)
                z = torch.cat([z, r_exemplar.view(z.size(0), -1)], dim=1)

            imagette = h[:, self.loc_size:]
            imagette = self.to_imagette(imagette).view(write_size)
            theta_mat_td = compute_theta_general(theta, invert=True)
            td_att = write_transformer(theta_mat_td, imagette, x_size)
            h, c = self.rnn(z, (h, c))
            theta = self.fc_loc(h[:, :self.loc_size])
            canvas = canvas + td_att

            if not low_memory:
                all_td_att[t+1] = td_att.detach()
                all_gene[t+1] = torch.sigmoid(canvas)
        reco = torch.sigmoid(canvas)
        if not low_memory:
            return reco, torch.stack(all_td_att, dim=1), torch.stack(all_gene, dim=1)
        else:
            return reco, None, None

