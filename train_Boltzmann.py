import argparse
from functools import partial

import dgl

import numpy as np
import torch
import torch.nn as nn
from dataloading import (
    METR_LAGraphDataset,
    METR_LATestDataset,
    METR_LATrainDataset,
    METR_LAValidDataset,
    PEMS_BAYGraphDataset,
    PEMS_BAYTestDataset,
    PEMS_BAYTrainDataset,
    PEMS_BAYValidDataset,
)
from dcrnn import DiffConv
from gaan import GatedGAT
# from model import GraphRNN
from model_Boltzmann import KineticForecastingFramework as GraphRNN
from torch.utils.data import DataLoader
from utils import get_learning_rate, masked_mae_loss, NormalizationLayer

batch_cnt = [0]


def train(
    model,
    graph,
    dataloader,
    optimizer,
    scheduler,
    normalizer,
    loss_fn,
    device,
    args,
):
    predict_loss = []
    reconstructed_loss = []
    graph = graph.to(device)
    model.train()
    batch_size = args.batch_size
    for i, (x, y) in enumerate(dataloader):
        model.source_hidden = None
        optimizer.zero_grad()
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[: x.shape[0], :, :, :] = x
            x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            y_buff[: x.shape[0], :, :, :] = y
            y_buff[x.shape[0] :, :, :, :] = y[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            x = x_buff
            y = y_buff
        # Permute the dimension for shaping
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        # x_norm = (
        #     normalizer.normalize(x)
        #     .reshape(x.shape[0], -1, x.shape[3])
        #     .float()
        #     .to(device)
        # )
        # y_norm = (
        #     normalizer.normalize(y)
        #     .reshape(x.shape[0], -1, x.shape[3])
        #     .float()
        #     .to(device)
        # )
        x = x.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(y.shape[0], -1, y.shape[3]).float().to(device)

        # batch_graph = dgl.batch([graph] * batch_size)
        # if x_norm.shape != y_norm.shape:
        #     raise ValueError(
        #         f"Shape mismatch: x_norm {x_norm.shape} vs y_norm {y_norm.shape}"
        #     )

        output, reconstruct = model(graph=graph, macro_features_sequence=x, num_pred_steps=x.shape[0], target_sequence=y, batch_cnt = batch_cnt[0])
        loss_predict = loss_fn(output, y[...,:1])
        loss_reconstruct = loss_fn(reconstruct, x[...,:1])
        loss = loss_predict + loss_reconstruct
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if get_learning_rate(optimizer) > args.minimum_lr:
            scheduler.step()
        reconstructed_loss.append(float(loss_reconstruct))
        predict_loss.append(float(loss_predict))
        batch_cnt[0] += 1
        # print("\rBatch: ", i, end="")
        print(f"\rBatch: {i} Predict Loss: {loss_predict:.4f} Reconstruct Loss: {loss_reconstruct:.4f}", end="")
        # print(f'\nTarget Velocity of Node 0: {y[:,0,0]}')
        # print(f'\nPredicted Velocity of Node 0: {output[:,0,0]}')
        # print(f'\nTarget Reconstructed Velocity of Node 0: {x[:,0,0]}')
        # print(f'\nPredicted Reconstructed Velocity of Node 0: {reconstruct[:,0,0]}')
    return np.mean(predict_loss), np.mean(reconstructed_loss)


def eval(model, graph, dataloader, normalizer, loss_fn, device, args):
    predict_loss = []
    reconstructed_loss = []
    graph = graph.to(device)
    model.eval()
    batch_size = args.batch_size
    for i, (x, y) in enumerate(dataloader):
        model.source_hidden = None
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[: x.shape[0], :, :, :] = x
            x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            y_buff[: x.shape[0], :, :, :] = y
            y_buff[x.shape[0] :, :, :, :] = y[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            x = x_buff
            y = y_buff
        # Permute the order of dimension
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        # x_norm = (
        #     normalizer.normalize(x)
        #     .reshape(x.shape[0], -1, x.shape[3])
        #     .float()
        #     .to(device)
        # )
        # y_norm = (
        #     normalizer.normalize(y)
        #     .reshape(x.shape[0], -1, x.shape[3])
        #     .float()
        #     .to(device)
        # )
        x = x.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(y.shape[0], -1, y.shape[3]).float().to(device)

        # batch_graph = dgl.batch([graph] * batch_size)
        output, y_reconstruct = model(graph = graph, macro_features_sequence=x, num_pred_steps=x.shape[0], target_sequence=y, batch_cnt = batch_cnt[0])
        loss_predict = loss_fn(output, y[...,:1])
        loss_reconstruct = loss_fn(y_reconstruct, x[...,:1])
        predict_loss.append(float(loss_predict))
        reconstructed_loss.append(float(loss_reconstruct))
    return np.mean(predict_loss), np.mean(reconstructed_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Size of batch for minibatch Training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for parallel dataloading",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gaan",
        help="Which model to use DCRNN vs GaAN",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU indexm -1 for CPU training"
    )
    parser.add_argument(
        "--diffsteps",
        type=int,
        default=2,
        help="Step of constructing the diffusiob matrix",
    )
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of multiattention head"
    )
    parser.add_argument(
        "--map_feats", type=int, default=128
    )
    parser.add_argument(
        "--decay_steps",
        type=int,
        default=2000,
        help="Teacher forcing probability decay ratio",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Initial learning rate"
    )
    parser.add_argument(
        "--minimum_lr",
        type=float,
        default=1e-7,
        help="Lower bound of learning rate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LA",
        help="dataset LA for METR_LA; BAY for PEMS_BAY",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epoches for training"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for update parameters",
    )
    args = parser.parse_args()
    # Load the datasets
    if args.dataset == "LA":
        g = METR_LAGraphDataset()
        train_data = METR_LATrainDataset()
        test_data = METR_LATestDataset()
        valid_data = METR_LAValidDataset()
        print(f'Min and max of the training data X: {train_data.x.min()}, {train_data.x.max()}')
        print(f'Min and max of the training data Y: {train_data.y.min()}, {train_data.y.max()}')
        print(f'Min and max of the test data X: {test_data.x.min()}, {test_data.x.max()}')
        print(f'Min and max of the test data Y: {test_data.y.min()}, {test_data.y.max()}')
        print(f'Min and max of the validation data X: {valid_data.x.min()}, {valid_data.x.max()}')
        print(f'Min and max of the validation data Y: {valid_data.y.min()}, {valid_data.y.max()}')
    elif args.dataset == "BAY":
        g = PEMS_BAYGraphDataset()
        train_data = PEMS_BAYTrainDataset()
        test_data = PEMS_BAYTestDataset()
        valid_data = PEMS_BAYValidDataset()
        print(f'Min and max of the training data X: {train_data.x.min()}, {train_data.x.max()}')
        print(f'Min and max of the training data Y: {train_data.y.min()}, {train_data.y.max()}')
        print(f'Min and max of the test data X: {test_data.x.min()}, {test_data.x.max()}')
        print(f'Min and max of the test data Y: {test_data.y.min()}, {test_data.y.max()}')
        print(f'Min and max of the validation data X: {valid_data.x.min()}, {valid_data.x.max()}')
        print(f'Min and max of the validation data Y: {valid_data.y.min()}, {valid_data.y.max()}')

    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    normalizer = NormalizationLayer(train_data.mean, train_data.std)

    # if args.model == "diffconv":
    #     batch_g = dgl.batch([g] * args.batch_size).to(device)
    #     out_gs, in_gs = DiffConv.attach_graph(batch_g, args.diffsteps)
    #     net = partial(
    #         DiffConv,
    #         k=args.diffsteps,
    #         in_graph_list=in_gs,
    #         out_graph_list=out_gs,
    #     )
    # elif args.model == "gaan":
    #     net = partial(GatedGAT, map_feats=64, num_heads=args.num_heads)

    batch_g = dgl.batch([g] * args.batch_size).to(device)

    # k = 2
    # out_graph_list, in_graph_list = DiffConv.attach_graph(batch_g, k)

    conv_params = {
        # "k": 2,
        # "in_graph_list": in_graph_list if args.model == "diffconv" else None,
        # "out_graph_list": out_graph_list if args.model == "diffconv" else None,
        'map_feats': args.map_feats if args.model == "gaan" else None,
        'num_heads': args.num_heads if args.model == "gaan" else None,
    }
    
    Q_mesoscale = 71
    min_macrovelocity = 0
    max_macrovelocity = 70

    num_macro_to_meso_layers = 1
    num_layers_collision = 10
    hidden_dim_collision = 128
    source_mlp_num_layers = 10
    source_mlp_hidden_dim = 128

    dcrnn = GraphRNN(
        d_features=2,
        d_features_source=2,
        Q_mesoscale=Q_mesoscale,
        min_macrovelocity=min_macrovelocity,
        max_macrovelocity=max_macrovelocity,
        num_layers_macro_to_meso=num_macro_to_meso_layers,
        spatial_conv_type=args.model,
        conv_params=conv_params,
        collision_constraint='hard',
        dt=5/60,
        decay_steps=args.decay_steps,
        device=device,
        num_layers_collision=num_layers_collision,
        hidden_dim_collision=hidden_dim_collision,
        base_graph= batch_g,
        source_mlp_num_layers=source_mlp_num_layers,
        source_mlp_hidden_dim=source_mlp_hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(dcrnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = masked_mae_loss
    total_params = sum(p.numel() for p in dcrnn.parameters())
    print(f"Total number of parameters: {total_params}")

    trainable_params = sum(p.numel() for p in dcrnn.parameters() if p.requires_grad)
    print(f"Trainable number of parameters: {trainable_params}")

    for e in range(args.epochs):
        train_loss_predict, train_loss_reconstruct = train(
            dcrnn,
            batch_g,
            train_loader,
            optimizer,
            scheduler,
            normalizer,
            loss_fn,
            device,
            args,
        )
        valid_loss_predict, valid_loss_reconstruct = eval(
            dcrnn, batch_g, valid_loader, normalizer, loss_fn, device, args
        )
        test_loss_predict, test_loss_reconstruct = eval(
            dcrnn, batch_g, test_loader, normalizer, loss_fn, device, args
        )
        print(
            "\rEpoch: {} Train Loss Predict: {:.4f} Valid Loss Predict: {:.4f} Test Loss Predict: {:.4f}".format(
                e, train_loss_predict, valid_loss_predict, test_loss_predict
            )
        )
        print(
            "\rEpoch: {} Train Loss Reconstruct: {:.4f} Valid Loss Reconstruct: {:.4f} Test Loss Reconstruct: {:.4f}".format(
                e, train_loss_reconstruct, valid_loss_reconstruct, test_loss_reconstruct
            )
        )