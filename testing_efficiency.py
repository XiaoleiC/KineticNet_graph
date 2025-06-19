from torch.profiler import profile, record_function, ProfilerActivity
import torch
from model_Boltzmann import KineticForecastingFramework as GraphRNN
from dataloading import METR_LAGraphDataset, METR_LATrainDataset
import dgl
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

g = METR_LAGraphDataset()
batch_size = 4
batch_g = dgl.batch([g] * batch_size).to(device)

conv_params = {
    'map_feats': 64,
    'num_heads': 2
}

train_data = METR_LATrainDataset()
train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
x, y = next(iter(train_loader))
x = x.permute(1,0,2,3)
y = y.permute(1,0,2,3)
x = x.reshape(x.shape[0], -1, x.shape[3]).float()
y = y.reshape(y.shape[0], -1, y.shape[3]).float()
print(x.shape, y.shape)


x = x.to(device)
y = y.to(device)
print('model preparation finished...')

model = GraphRNN(
        d_features=2,
        d_features_source=2,
        Q_mesoscale=21,
        xi_velocities=torch.linspace(-10,10,21).to(device),
        min_macrovelocity=0,
        max_macrovelocity=70,
        num_layers_macro_to_meso=1,
        spatial_conv_type='gaan',
        conv_params=conv_params,
        collision_constraint='hard',
        dt=5/60,
        decay_steps=2000,
        device=device,
        num_layers_collision=5
    ).to(device)

print('model class finished...')

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=True) as prof:
    preds, recon = model(graph=batch_g, 
                         macro_features_sequence=x[:2].to(device),
                         num_pred_steps=2,
                         target_sequence=y[:2].to(device))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
