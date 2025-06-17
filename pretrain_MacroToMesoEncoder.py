import argparse
from functools import partial

import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataloading import (
    METR_LAGraphDataset, METR_LATrainDataset, METR_LAValidDataset,
    PEMS_BAYGraphDataset, PEMS_BAYTrainDataset, PEMS_BAYValidDataset
)
from dcrnn import DiffConv
from model_Boltzmann import (
    MacroToMesoEncoder, MesoToMacroDecoder
)
from utils import NormalizationLayer, masked_mae_loss

class MacroToMesoPretrainer(nn.Module):
    def __init__(self,
                 d_features: int,
                 Q_mesoscale: int,
                 xi_velocities: torch.Tensor,
                 encoder_layers: int,
                 spatial_conv_type: str,
                 conv_params: dict):
        super().__init__()

        self.encoder = MacroToMesoEncoder(
            d_features=d_features,
            Q_mesoscale=Q_mesoscale,
            num_layers=encoder_layers,
            spatial_conv_type=spatial_conv_type,
            conv_params=conv_params,
            is_SGRNN=False
        )
        self.decoder = MesoToMacroDecoder(
            Q_mesoscale=Q_mesoscale,
            d_features=d_features,
            xi_velocities=xi_velocities
        )

    def forward(self, g, macro_x):
        T, N, _ = macro_x.shape
        recon_seq = []
        for t in range(T):
            f_t = self.encoder(g, macro_x[t,:,:])
            recon_t = self.decoder(f_t, macro_x[t,:,:1])
            recon_seq.append(recon_t) 
        return torch.stack(recon_seq, dim=0)


def train_one_epoch(model, graph, loader,
                    normalizer, loss_fn,
                    optimizer, device):
    model.train()
    epoch_loss = []
    for x, _ in loader:
        if x.shape[0] != loader.batch_size:
            x_buff = torch.zeros(loader.batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[: x.shape[0], :, :, :] = x
            x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                loader.batch_size - x.shape[0], 1, 1, 1
            )
            x = x_buff

        x = x.permute(1, 0, 2, 3)
        x_norm = x.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
        # x_norm = (
        #     normalizer.normalize(x)
        #     .reshape(x.shape[0], -1, x.shape[3])
        #     .float()
        #     .to(device)
        # )

        optimizer.zero_grad()
        recon = model(graph.to(device), x_norm)
        x_tgt = x_norm[..., :1]

        loss = loss_fn(recon, x_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        epoch_loss.append(float(loss))
    print(f'target: {x_norm[:,0,0]}')
    print(f'recon: {recon[:,0,0]}')
    return np.mean(epoch_loss)


@torch.no_grad()
def evaluate(model, graph, loader,
             normalizer, loss_fn, device):
    model.eval()
    epoch_loss = []
    for x, _ in loader:
        if x.shape[0] != loader.batch_size:
            x_buff = torch.zeros(loader.batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[: x.shape[0], :, :, :] = x
            x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                loader.batch_size - x.shape[0], 1, 1, 1
            )
            x = x_buff

        x = x.permute(1, 0, 2, 3)
        x_norm = x.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
        # x_norm = (
        #     normalizer.normalize(x)
        #     .reshape(x.shape[0], -1, x.shape[3])
        #     .float()
        #     .to(device)
        # )

        recon = model(graph.to(device), x_norm)
        loss = loss_fn(recon, x_norm[..., :1])
        epoch_loss.append(float(loss))
    print(f'target: {x_norm[:,0,0]}')
    print(f'recon: {recon[:,0,0]}')
    return np.mean(epoch_loss)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LA', choices=['LA', 'BAY'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--diffsteps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.dataset == 'LA':
        g_raw = METR_LAGraphDataset()
        train_ds, val_ds = METR_LATrainDataset(), METR_LAValidDataset()
    else:
        g_raw = PEMS_BAYGraphDataset()
        train_ds, val_ds = PEMS_BAYTrainDataset(), PEMS_BAYValidDataset()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    normalizer = NormalizationLayer(train_ds.mean, train_ds.std)

    batch_g = dgl.batch([g_raw] * args.batch_size).to(device)
    out_gs, in_gs = DiffConv.attach_graph(batch_g, args.diffsteps)
    conv_params = dict(k=args.diffsteps,
                       in_graph_list=in_gs,
                       out_graph_list=out_gs)

    Q = 30
    xi = torch.linspace(-10,10, Q).to(device)
    model = MacroToMesoPretrainer(
        d_features=2,
        Q_mesoscale=Q,
        xi_velocities=xi,
        encoder_layers=1,
        spatial_conv_type='gaan',
        conv_params=conv_params
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
    loss_fn = masked_mae_loss

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, batch_g, train_loader,
                                  normalizer, loss_fn, optimizer, device)
        val_loss = evaluate(model, batch_g, val_loader,
                            normalizer, loss_fn, device)
        scheduler.step()

        print(f'Epoch {epoch}: train {tr_loss:.4f}  val {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.encoder.state_dict(), 'encoder_state_dict.pth')
            print('Saved best encoder_state_dict.pth')

    print('Finished pretraining!')

if __name__ == '__main__':
    main()
