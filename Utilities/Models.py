import torch
import numpy as np
import torch.nn as nn


def choose_nonlinearity(name):
    nl = None

    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("non-linearity not recognized")

    return nl


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight)

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)


class PSD(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, diag_dim, nonlinearity='tanh'):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        if diag_dim == 1:
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, diag_dim)

            for l in [self.linear1, self.linear2, self.linear3]:
                torch.nn.init.orthogonal_(l.weight)

            self.nonlinearity = choose_nonlinearity(nonlinearity)
        else:
            assert diag_dim > 1
            self.diag_dim = diag_dim
            self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = torch.nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

            for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
                torch.nn.init.orthogonal_(l.weight)

            self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, q):
        if self.diag_dim == 1:
            h = self.nonlinearity(self.linear1(q))
            h = self.nonlinearity(self.linear2(h))
            h = self.nonlinearity(self.linear3(h))
            return h * h + 0.1
        else:
            bs = q.shape[0]
            h = self.nonlinearity(self.linear1(q))
            h = self.nonlinearity(self.linear2(h))
            h = self.nonlinearity(self.linear3(h))
            diag, off_diag = torch.split(self.linear4(h), [self.diag_dim, self.off_diag_dim], dim=1)

            L = torch.diag_embed(diag)

            ind = np.tril_indices(self.diag_dim, k=-1)
            flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            L = torch.flatten(L, start_dim=1)
            L[:, flat_ind] = off_diag
            L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

            D = torch.bmm(L, L.permute(0, 2, 1))
            for i in range(self.diag_dim):
                D[:, i, i] += 0.1
            return D


class DynNet(nn.Module):
    def __init__(self, q_dim, u_dim, g_net=None, Minv_net=None, C_net=None):
        super(DynNet, self).__init__()
        self.q_dim = q_dim
        self.u_dim = u_dim

        self.g_net = g_net
        self.Minv_net = Minv_net
        self.C_net = C_net

    def forward(self, t, x):
        s, u = x.split([6, 2], dim=1)

        cos_q1_sin_q1_cos_q2_sin_q2, q1_dot_q2_dot = s.split([4, 2], dim=1)

        s1, s2, s3, s4 = cos_q1_sin_q1_cos_q2_sin_q2.split([1, 1, 1, 1], dim=1)

        s5, s6 = q1_dot_q2_dot.split([1, 1], dim=1)

        c_feed = torch.cat([s4, s5, s6], dim=1)

        g_feed = torch.cat([s1, s2, s3, s4], dim=1)

        d_s1 = -s2 * s5
        d_s2 = s1 * s5
        d_s3 = -s4 * s6
        d_s4 = s3 * s6

        temp = u - self.C_net(c_feed) - self.g_net(g_feed)

        M_SHAPE_0, _, _ = self.Minv_net(s3).shape
        q1_ddot_q2_ddot = []

        for k in range(M_SHAPE_0):
            ddot = torch.matmul(self.Minv_net(s3)[k], temp[k].t())

            q1_ddot_q2_ddot.append(ddot)

        q1_ddot_q2_ddot = torch.stack(q1_ddot_q2_ddot)

        d_s5, d_s6 = q1_ddot_q2_ddot.split([self.q_dim, self.q_dim], dim=1)
        d_u1 = torch.zeros(d_s4.shape[0], 1)
        d_u2 = torch.zeros(d_s4.shape[0], 1)

        return torch.cat([d_s1, d_s2, d_s3, d_s4, d_s5, d_s6, d_u1, d_u2], dim=1)
