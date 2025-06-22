import argparse
from functools import partial

import dgl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
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
import os

batch_cnt = [0]

def save_checkpoint(model, optimizer, scheduler, epoch, batch_cnt, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'batch_cnt': batch_cnt,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best model if specified
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved at {best_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        batch_cnt[0] = checkpoint.get('batch_cnt', 0)
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
        return start_epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # First try to find latest_checkpoint.pth
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Otherwise find the checkpoint with highest epoch number
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the maximum
    epoch_numbers = []
    for file in checkpoint_files:
        try:
            epoch_num = int(file.split('checkpoint_epoch_')[1].split('.pth')[0])
            epoch_numbers.append((epoch_num, file))
        except:
            continue
    
    if epoch_numbers:
        latest_file = max(epoch_numbers, key=lambda x: x[0])[1]
        return latest_file
    
    return None


def train(
    model,
    graph,
    dataloader,
    optimizer,
    scheduler,
    normalizer,
    loss_fn,
    device,
    node_position_batch,
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

        output, reconstruct = model(graph=graph, macro_features_sequence=x, num_pred_steps=x.shape[0], target_sequence=y, batch_cnt = batch_cnt[0], node_position=node_position_batch)
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
    for node_id in range(0, 100):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[1].plot(output[:,node_id,0].detach().cpu().numpy(), label='Predicted Velocity')
        ax[1].plot(y[:,node_id,0].detach().cpu().numpy(), label='Target Velocity')
        ax[1].set_title(f'Node {node_id} - Predict')
        ax[0].plot(reconstruct[:,node_id,0].detach().cpu().numpy(), label='Predicted Reconstructed Velocity')
        ax[0].plot(x[:,node_id,0].detach().cpu().numpy(), label='Target Reconstructed Velocity')
        ax[0].set_title(f'Node {node_id} - Reconstructed')
        ax[0].legend()
        ax[1].legend()
        save_dir = os.path.join("figures", "train")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'train_predicted_vs_target_velocity_{node_id}.png')
        fig.savefig(save_path)
        plt.close(fig)
    print('\ntraining figures saved...')


    return np.mean(predict_loss), np.mean(reconstructed_loss)


def eval(model, graph, dataloader, normalizer, loss_fn, device, node_position_batch, args):
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
        with torch.no_grad():
            output, y_reconstruct = model(graph = graph, macro_features_sequence=x, num_pred_steps=x.shape[0], target_sequence=y, batch_cnt = batch_cnt[0], node_position=node_position_batch)
        loss_predict = loss_fn(output, y[...,:1])
        loss_reconstruct = loss_fn(y_reconstruct, x[...,:1])
        predict_loss.append(float(loss_predict))
        reconstructed_loss.append(float(loss_reconstruct))
    for node_id in range(0, 100):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[1].plot(output[:,node_id,0].detach().cpu().numpy(), label='Predicted Velocity')
        ax[1].plot(y[:,node_id,0].detach().cpu().numpy(), label='Target Velocity')
        ax[1].set_title(f'Node {node_id} - Predict')
        ax[0].plot(y_reconstruct[:,node_id,0].detach().cpu().numpy(), label='Predicted Reconstructed Velocity')
        ax[0].plot(x[:,node_id,0].detach().cpu().numpy(), label='Target Reconstructed Velocity')
        ax[0].set_title(f'Node {node_id} - Reconstructed')
        ax[0].legend()
        ax[1].legend()
        save_dir = os.path.join("figures", "eval")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'eval_predicted_vs_target_velocity_{node_id}.png')
        fig.savefig(save_path)
        plt.close(fig)

    print('\nevaluation figures saved...')
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
        default=1e-6,
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
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="trainedmodel",
        help="Directory to save/load model checkpoints",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=2,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
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

    node_position = torch.arange(g.num_nodes()).float().unsqueeze(1).to(device)
    node_position_batch0 = node_position.repeat(args.batch_size,1)
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
    num_layers_collision = 5
    hidden_dim_collision = 64
    source_mlp_num_layers = 8
    source_mlp_hidden_dim = 64

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
        is_BGK=True,
        is_using_feq=False
    ).to(device)

    # optimizer = torch.optim.Adam(dcrnn.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(dcrnn.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = masked_mae_loss
    total_params = sum(p.numel() for p in dcrnn.parameters())
    print(f"Total number of parameters: {total_params}")

    trainable_params = sum(p.numel() for p in dcrnn.parameters() if p.requires_grad)
    print(f"Trainable number of parameters: {trainable_params}")

    start_epoch = 0
    best_loss = float('inf')

    if args.resume or os.path.exists(args.checkpoint_dir):
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            start_epoch, best_loss = load_checkpoint(dcrnn, optimizer, scheduler, latest_checkpoint)
        else:
            print(f"No checkpoint found in {args.checkpoint_dir}, starting from scratch")
    
    print(f"Starting training from epoch {start_epoch}")


    for e in range(start_epoch, args.epochs):
        train_loss_predict, train_loss_reconstruct = train(
            dcrnn,
            batch_g,
            train_loader,
            optimizer,
            scheduler,
            normalizer,
            loss_fn,
            device,
            node_position_batch0,
            args,
        )
        valid_loss_predict, valid_loss_reconstruct = eval(
            dcrnn, batch_g, valid_loader, normalizer, loss_fn, device, node_position_batch0, args
        )
        test_loss_predict, test_loss_reconstruct = eval(
            dcrnn, batch_g, test_loader, normalizer, loss_fn, device, node_position_batch0, args
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

        current_loss = valid_loss_predict + valid_loss_reconstruct
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss
        
        if (e + 1) % args.save_freq == 0 or is_best or e == args.epochs - 1:
            save_checkpoint(
                dcrnn, 
                optimizer, 
                scheduler, 
                e, 
                batch_cnt[0], 
                current_loss, 
                args.checkpoint_dir, 
                is_best=is_best
            )