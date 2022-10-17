import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings

warnings.filterwarnings("ignore")

from torchdiffeq import odeint
import numpy as np
import random
import sys

from DataGenerator.DataGenerator import ArrangeData, TrajDataset, ProperState, Xdot, DataGenerator
from Utilities.Models import *

losses = []
device = torch.device('mps')

def my_collate(batch):
    x, u = zip(*batch)

    x = [torch.as_tensor(b) for b in x]
    u = [torch.as_tensor(b) for b in u]

    elem = x[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        numel = sum([b.numel() for b in x])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    x_collate = torch.stack(x, 1, out=out)

    elem = u[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        numel = sum([b.numel() for b in u])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    u_collate = torch.stack(u, 0, out=out)

    return [x_collate, u_collate]


class Model(pl.LightningModule):
    def __init__(self, data, w_loss, hparams = {'batch_size': 20,'learning_rate': 1e-3}):
        super(Model, self).__init__()
        # misc. vars
        self.x_pred = None
        self.t_eval = None
        self.w_loss = w_loss
        # hyperparameters for the model
        for key in hparams.keys():
            self.hparams[key]=hparams[key]

        # dataset for the model
        self.data = data

        # number of timesteps per sub-trajectory.
        self.num_steps = 4
        self.loss_fn = nn.MSELoss(reduction="none")

        # coriolis term
        self.C_net = MLP(3, 50, 2)
        # gravity term
        self.G_net = MLP(4, 50, 2)
        # mass matrix
        self.M_net = PSD(1, 50, 2)

        # system dynamics to be trained using neural ode model (adjoint disabled right now)
        self.ode = DynNet(q_dim=1, u_dim=2, g_net=self.G_net, Minv_net=self.M_net, C_net=self.C_net)

        # training dataset not yet modified using our custom dataset class, placeholder for the same.
        self.train_dataset = None
        self.non_ctrl_ind = 1



    def train_dataloader(self):
        if self.train_dataset is None:
            self.train_dataset = TrajDataset(self.data, self.num_steps)

        if self.current_epoch < 1000:

            u_idx = self.non_ctrl_ind
            self.non_ctrl_ind += 1

            if self.non_ctrl_ind == 6:
                self.non_ctrl_ind = 1

        else:
            u_idx = self.current_epoch % 6

        self.train_dataset.u_idx = u_idx
        self.t_eval = torch.from_numpy(self.train_dataset.t_eval)

        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], collate_fn=my_collate)

    def forward(self, x, u):
        [_, bs, d] = x.shape
        T = len(self.t_eval)

        u_to_concatenate = torch.empty(x[0].shape[0], u.shape[1]).to(device)

        for step in range(x[0].shape[0]):
            u_to_concatenate[step] = u[0]


        z0_u = torch.cat((x[0], u_to_concatenate), dim=1).to()
        zT_u = odeint(self.ode, z0_u, self.t_eval)

        q, _, _ = zT_u.split([6, 1, 1], dim=-1)
        self.x_pred = q

        self.x_pred = self.x_pred.view([T, bs, d])

    def training_step(self, train_batch, batch_idx):

        x_train, u = train_batch
        self.forward(x_train, u)

        if(self.w_loss):
            loss = self.loss_fn(self.x_pred[:, :, :4], x_train[:, :, :4]).sum() + 1000*self.loss_fn(self.x_pred[:, :, 4:], x_train[:, :, 4:]).sum()
        else:
            loss = self.loss_fn(x_train, self.x_pred).sum()

        losses.append(loss.item())

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams['learning_rate'])


if __name__ == "__main__":
    import os

    PARENT_DIR = "/Users/aditya/PycharmProjects/pythonProject"


    def main(data):
        model = Model(data=data, w_loss=False)
        checkpoint_callback = ModelCheckpoint(monitor='loss',
                                              save_top_k=1,
                                              save_last=True)

        trainer = Trainer(
            deterministic=True,
            default_root_dir=os.path.join(PARENT_DIR, 'logs'),
            callbacks=[checkpoint_callback],
            accelerator='gpu'
        )

            # .from_argparse_args(args=model.hparams,
            #                                  deterministic=True,
            #                                  default_root_dir=os.path.join(PARENT_DIR, 'logs'),
            #                                  callbacks=[checkpoint_callback])

        trainer.fit(model)


    gen = DataGenerator(num_steps=20, num_init=20, controls=[1, 0.9, 0.8, 0.7, 0.6, 0.5])
    data = gen.generate()
    main(data)

