"""
pretrain_macro_to_meso.py

预训练 MacroToMesoEncoder → MesoToMacroDecoder 以实现自监督重构
python pretrain_macro_to_meso.py --dataset LA --epochs 20
"""
import argparse
from functools import partial

import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# ===== 项目内依赖 =====
from dataloading import (          # :contentReference[oaicite:0]{index=0}
    METR_LAGraphDataset, METR_LATrainDataset, METR_LAValidDataset,
    PEMS_BAYGraphDataset, PEMS_BAYTrainDataset, PEMS_BAYValidDataset
)
from dcrnn import DiffConv          # :contentReference[oaicite:1]{index=1}
from model_Boltzmann import (       # :contentReference[oaicite:2]{index=2}
    MacroToMesoEncoder, MesoToMacroDecoder
)
from utils import NormalizationLayer, masked_mae_loss  # :contentReference[oaicite:3]{index=3}

# ------------------ 1. 预训练包装 ------------------
class MacroToMesoPretrainer(nn.Module):
    """
    Macro → Meso Encoder + 轻量 Decoder 做自监督重构
    """
    def __init__(self,
                 d_features: int,
                 Q_mesoscale: int,
                 xi_velocities: torch.Tensor,
                 encoder_layers: int,
                 spatial_conv_type: str,
                 conv_params: dict):
        super().__init__()

        self.register_buffer('xi_velocities', xi_velocities)

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
            xi_velocities=self.xi_velocities
        )

    def forward(self, g, macro_x):
        """
        macro_x: [T, N, d]  这里只用 d=1 的速度
        """
        T, N, _ = macro_x.shape
        recon_seq = []
        for t in range(T):
            f_t = self.encoder(g, macro_x[t,:,:1])
            recon_t = self.decoder(f_t)          # 仅输出速度 [N,1]
            recon_seq.append(recon_t)
        return torch.stack(recon_seq, dim=0)            # [T,N,1]

# ------------------ 2. 训练入口 ------------------
def train_one_epoch(model, graph, loader,
                    normalizer, loss_fn,
                    optimizer, device):
    model.train()
    epoch_loss = []
    graph = graph.to(device)
    for x, _ in loader:               # 这里只需要 x
        # shape: [B, seq, N, d]  →  [seq, B*N, d]
        x = x.permute(1, 0, 2, 3)                 # [T,B,N,d]
        x = x.reshape(x.shape[0], -1, x.shape[3]) # [T,B*N,d]

        x_norm = normalizer.normalize(x).float().to(device)

        optimizer.zero_grad()
        recon = model(graph, x_norm)              # [T,BN,1]
        x_tgt = x_norm[..., :1]                   # 只对速度做重构

        loss = loss_fn(recon, x_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        epoch_loss.append(float(loss))
    return np.mean(epoch_loss)


@torch.no_grad()
def evaluate(model, graph, loader,
             normalizer, loss_fn, device):
    model.eval()
    graph = graph.to(device)
    epoch_loss = []
    for x, _ in loader:
        x = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1, x.shape[3])
        x_norm = normalizer.normalize(x)[...,:1].float().to(device)
        recon = model(graph, x_norm)
        loss = loss_fn(recon, x_norm[..., :1])
        epoch_loss.append(float(loss))
    return np.mean(epoch_loss)


# ------------------ 3. main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LA', choices=['LA', 'BAY'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--diffsteps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # 3.1 数据集与图
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

    # 3.2 预计算 DiffConv 图
    batch_g = dgl.batch([g_raw] * args.batch_size).to(device)
    out_gs, in_gs = DiffConv.attach_graph(batch_g, args.diffsteps)
    conv_params = dict(k=args.diffsteps,
                       in_graph_list=in_gs,
                       out_graph_list=out_gs)

    # 3.3 模型
    Q = 12                                  # 可自定义
    xi = torch.linspace(0, 1, Q).to(device) # 正速度网格
    model = MacroToMesoPretrainer(
        d_features=1,                       # 这里只预训练速度
        Q_mesoscale=Q,
        xi_velocities=xi,
        encoder_layers=1,
        spatial_conv_type='diffconv',       # 或 'gaan'
        conv_params=conv_params
    ).to(device)

    # 3.4 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
    loss_fn = masked_mae_loss               # MAE 更鲁棒

    # 3.5 训练
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, batch_g, train_loader,
                                  normalizer, loss_fn, optimizer, device)
        val_loss = evaluate(model, batch_g, val_loader,
                            normalizer, loss_fn, device)
        scheduler.step()

        print(f'Epoch {epoch:03d}: train {tr_loss:.4f}  val {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.encoder.state_dict(), 'encoder_state_dict.pth')
            print('  ✓ Saved best encoder_state_dict.pth')

    print('Finished pretraining!')

if __name__ == '__main__':
    main()
