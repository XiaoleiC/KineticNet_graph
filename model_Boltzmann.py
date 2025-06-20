import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
from typing import Optional, Union
from dcrnn import DiffConv
from gaan import GatedGAT
from functools import partial
import time

class MacroToMesoEncoder(nn.Module):
    def __init__(self, d_features: int, Q_mesoscale: int, num_layers: int = 1, spatial_conv_type: str = 'gaan', 
                 conv_params: dict = None, is_SGRNN: bool = False):
        super(MacroToMesoEncoder, self).__init__()
        self.d_features = d_features
        self.Q_mesoscale = Q_mesoscale
        self.spatial_conv_type = spatial_conv_type
        self.conv_layers = nn.ModuleList()
        self.is_SGRNN = is_SGRNN
        if spatial_conv_type == 'diffconv':
            k = conv_params.get('k', 2)
            in_graph_list = conv_params.get('in_graph_list', [])
            out_graph_list = conv_params.get('out_graph_list', [])
        elif spatial_conv_type == 'gaan':
            map_feats = conv_params.get('map_feats', 64)
            num_heads = conv_params.get('num_heads', 2)
        else:
            raise ValueError(f"Unsupported spatial_conv_type: {spatial_conv_type}")

        for i in range(num_layers):
            in_dim = d_features if i == 0 else Q_mesoscale
            out_dim = Q_mesoscale
            
            if spatial_conv_type == 'diffconv':
                conv_layer = DiffConv(in_dim, out_dim, k, in_graph_list, out_graph_list)
            elif spatial_conv_type == 'gaan':
                conv_layer = GatedGAT(in_dim, out_dim, map_feats, num_heads)
            else:
                raise ValueError(f"Unsupported spatial_conv_type: {spatial_conv_type}")
            
            self.conv_layers.append(conv_layer)

    def forward(self, graph, macro_features):
        x = macro_features
        
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(graph, x)
            x = nn.functional.tanh(x) if i < len(self.conv_layers) - 1 else x
        if not self.is_SGRNN:
            x = nn.functional.relu(x)
        return x


class CollisionOperator(nn.Module):
    def __init__(self, Q_mesoscale: int, hidden_dim: int = 64, 
                 constraint_type: str = 'none', xi_velocities: torch.Tensor = None, num_layers: int = 5):
        super(CollisionOperator, self).__init__()
        self.Q_mesoscale = Q_mesoscale
        self.constraint_type = constraint_type  # 'none', 'soft', 'hard'
        self.mlp = nn.ModuleList()
        for k1 in range(num_layers):
            in_dim = Q_mesoscale if k1 == 0 else hidden_dim
            out_dim = hidden_dim if k1 < num_layers - 1 else Q_mesoscale
            self.mlp.append(nn.Linear(in_dim, out_dim))
        self.register_buffer('xi_velocities', xi_velocities)
        self.register_buffer('C_matrix', self._compute_C_matrix())

    def _compute_C_matrix(self):
        Q = self.Q_mesoscale
        w = torch.ones(Q, device=self.xi_velocities.device) / Q
        C = torch.zeros(2, Q, device=self.xi_velocities.device)
        C[0, :] = w.unsqueeze(0)
        C[1, :] = w.unsqueeze(0) * self.xi_velocities.unsqueeze(0)
        return C

    
    def _apply_hard_constraint(self, omega_raw):
        if self.C_matrix is None:
            return omega_raw
        
        C = self.C_matrix
        I = torch.eye(self.Q_mesoscale, device=omega_raw.device)
        
        CCT_inv = torch.inverse(C @ C.T + 1e-6 * torch.eye(2, device=omega_raw.device))
        projection = I - C.T @ CCT_inv @ C
        
        omega_constrained = omega_raw @ projection.T
        
        return omega_constrained
    
    def forward(self, f_distribution):
        x = f_distribution
        for k2, layer in enumerate(self.mlp):
            x = layer(x)
            x = nn.functional.relu(x) if k2 < len(self.mlp) - 1 else nn.functional.tanh(x)
        omega_raw = x
        
        if self.constraint_type == 'hard':
            omega = self._apply_hard_constraint(omega_raw)
        else:  # 'none'
            omega = omega_raw
        
        return omega

class SGraphRNN(nn.Module):
    def __init__(self, d_features: int, Q_mesoscale: int, num_layers: int = 1, hidden_dim: int = 64,
                 spatial_conv_type: str = 'gaan', conv_params: dict = None, is_SGRNN: bool = True, out_num_layers: int = 5, out_hidden_dim: int = 128):
        super(SGraphRNN, self).__init__()
        self.d_features = d_features
        self.Q_mesoscale = Q_mesoscale
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = out_hidden_dim
        
        self.encoder = MacroToMesoEncoder(d_features, Q_mesoscale, num_layers,
                                        spatial_conv_type, conv_params, is_SGRNN=is_SGRNN)
        
        self.gru_cell = nn.GRUCell(Q_mesoscale, hidden_dim)
        
        self.output_mlp = self._build_mlp(
            hidden_dim, Q_mesoscale,
            self.out_hidden_dim, out_num_layers
        )

    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dim, num_layers):
        layers = []
        for i in range(num_layers):
            in_d  = in_dim  if i == 0 else hidden_dim
            out_d = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def forward(self, graph, macro_features, hidden_state=None):
        N = macro_features.shape[0]
        device = macro_features.device
        
        if hidden_state is None:
            hidden_state = torch.zeros(N, self.hidden_dim, device=device)
        
        meso_input = self.encoder(graph, macro_features)  # [N, Q]
        
        hidden_flat = hidden_state.view(-1, self.hidden_dim)
        meso_flat = meso_input.view(-1, self.Q_mesoscale)
        
        new_hidden_flat = self.gru_cell(meso_flat, hidden_flat)
        new_hidden = new_hidden_flat.view(N, self.hidden_dim)
        
        source_term = self.output_mlp(new_hidden)  # [N, Q]
        
        return source_term, new_hidden


class BoltzmannUpdater(nn.Module):
    def __init__(self, Q_mesoscale: int, dt: float = 0.1, min_macrovelocity: int = 0, max_macrovelocity: int = 75, base_graph = None):
        super(BoltzmannUpdater, self).__init__()
        self.Q_mesoscale = Q_mesoscale
        self.dt = dt
        self.min_macrovelocity = min_macrovelocity
        self.max_macrovelocity = max_macrovelocity
        self.register_buffer('xi_velocities', torch.linspace(min_macrovelocity, max_macrovelocity, Q_mesoscale, device=base_graph.device))
        if base_graph is None:
            raise ValueError("base_graph must be provided for BoltzmannUpdater")
        self.graph = dgl.remove_self_loop(base_graph)
        self.reverse_graph = dgl.reverse(base_graph, copy_ndata=True, copy_edata=True)
        self.graph.ndata['out_degree'] = self.graph.out_degrees().float()
        self.graph.ndata['in_degree'] = self.graph.in_degrees().float()
        self.reverse_graph.ndata['out_degree'] = self.reverse_graph.out_degrees().float()
        self.reverse_graph.ndata['in_degree'] = self.reverse_graph.in_degrees().float()
        self.xi_broadcast = self.xi_velocities.unsqueeze(0)
    
    def compute_transport_term(self, f_distribution):
        g = self.graph
        reverse_g = self.reverse_graph
        g.ndata['f'] = f_distribution

        def msg_in(edges):
            f_diff = edges.dst["f"] - edges.src["f"]
            edge_w = edges.data["weight"].unsqueeze(1)
            q_ij = (1.0 / edges.src["out_degree"]).unsqueeze(1)
            xi_exp = self.xi_broadcast.expand(f_diff.shape[0], -1)
            return {"transport_all": q_ij * xi_exp * f_diff * edge_w}
        
        def msg_out(edges):
            f_diff = edges.src["f"] - edges.dst["f"]
            edge_w = edges.data["weight"].unsqueeze(1)
            q_ij = (1.0 / edges.dst["in_degree"]).unsqueeze(1)
            xi_exp = self.xi_broadcast.expand(f_diff.shape[0], -1)
            return {"transport_all": q_ij * xi_exp * f_diff * edge_w}
        
        def red(nodes):
            return {"transport_sum_all": nodes.mailbox["transport_all"].sum(dim=1)}
        
        g.update_all(msg_in, red)
        inflow = g.ndata.pop('transport_sum_all')

        reverse_g.ndata['f'] = f_distribution
        reverse_g.update_all(msg_out, red)
        outflow = reverse_g.ndata.pop('transport_sum_all')

        return outflow - inflow
    
    def forward(self, f_distribution, collision_term, source_term):
        f_distribution = torch.clamp(f_distribution, min=0.0)
        transport_term = self.compute_transport_term(f_distribution)
        
        f_new = f_distribution - self.dt * (transport_term - collision_term - source_term)
        f_new = torch.clamp(f_new, min=0.0)
        
        return f_new


class MesoToMacroDecoder(nn.Module):
    def __init__(self, Q_mesoscale: int, xi_velocities: torch.Tensor):
        super(MesoToMacroDecoder, self).__init__()
        self.Q_mesoscale = Q_mesoscale
        self.register_buffer('xi_velocities', xi_velocities)
        self.register_buffer('weights', torch.ones(Q_mesoscale) / Q_mesoscale)
    
    def forward(self, f_distribution):
        N = f_distribution.shape[0]
        macro_variables = []
        xi_broadcast = self.xi_velocities.unsqueeze(0)
        density = torch.sum(f_distribution * self.weights, dim=1, keepdim=True)
        macro_variables.append(density)
        
        momentum = torch.sum(f_distribution * self.weights.unsqueeze(0) * xi_broadcast, dim=1, keepdim=True)
        velocity = momentum / (density + 1e-8)
        macro_variables.append(velocity)
        macro_output = macro_variables[1]
        
        return macro_output



class KineticForecastingFramework(nn.Module):
    def __init__(self, d_features: int, d_features_source: int, Q_mesoscale: int, min_macrovelocity=0, max_macrovelocity=70, num_layers_macro_to_meso: int = 1,
                 spatial_conv_type: str = 'gaan', conv_params: dict = None,
                 collision_constraint: str = 'none', dt: float = 0.1, decay_steps: int = 2000, device: Optional[Union[str, torch.device]] = 'cpu', num_layers_collision: int = 6, 
                 hidden_dim_collision: int = 64, base_graph = None, source_mlp_num_layers: int = 5, source_mlp_hidden_dim: int = 128):
        super(KineticForecastingFramework, self).__init__()
        
        self.d_features = d_features
        self.d_features_source = d_features_source
        self.Q_mesoscale = Q_mesoscale
        self.decay_steps = decay_steps
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.register_buffer('xi_velocities', torch.linspace(min_macrovelocity, max_macrovelocity, Q_mesoscale, device=device))

        if Q_mesoscale != self.xi_velocities.shape[0]:
            raise ValueError(f"Q_mesoscale ({Q_mesoscale}) must match xi_velocities length ({self.xi_velocities.shape[0]})")
        
        self.macro_to_meso = MacroToMesoEncoder(d_features=d_features, Q_mesoscale=Q_mesoscale, num_layers=num_layers_macro_to_meso,
                                               spatial_conv_type=spatial_conv_type, conv_params=conv_params, is_SGRNN=False)
         
        if base_graph is None:
            raise ValueError("base_graph must be provided for KineticForecastingFramework")
        self.boltzmann_updater = BoltzmannUpdater(Q_mesoscale, dt, min_macrovelocity=min_macrovelocity, max_macrovelocity=max_macrovelocity, base_graph=base_graph)
        
        self.source_rnn = SGraphRNN(d_features_source, Q_mesoscale, num_layers = num_layers_macro_to_meso,
                                   spatial_conv_type=spatial_conv_type,
                                   conv_params=conv_params, is_SGRNN=True, out_num_layers=source_mlp_num_layers, out_hidden_dim=source_mlp_hidden_dim)
        
        self.collision_op = CollisionOperator(Q_mesoscale, hidden_dim_collision, constraint_type=collision_constraint,
                                            xi_velocities=self.xi_velocities, num_layers=num_layers_collision)
        
        self.meso_to_macro = MesoToMacroDecoder(Q_mesoscale, torch.linspace(min_macrovelocity, max_macrovelocity, Q_mesoscale, device=self.device))
        self.source_hidden = None  # Hidden state for source term RNN

    def compute_teacher_forcing_threshold(self, batch_cnt):
        return self.decay_steps / (
            self.decay_steps + np.exp(batch_cnt / self.decay_steps) 
        )

    def forward(self, graph, macro_features_sequence, num_pred_steps: int = 1, 
                target_sequence: torch.Tensor = None, batch_cnt: int = 0):
        T = macro_features_sequence.shape[0]
        macro_features_sequence = macro_features_sequence.to(self.device)
        target_sequence = target_sequence.to(self.device)
        predictions = []
        reconstruction_outputs = []
        
        teacher_forcing_threshold = self.compute_teacher_forcing_threshold(batch_cnt)
        
        for t in range(T):
            current_macro = macro_features_sequence[t]
            source_term, self.source_hidden = self.source_rnn(graph, current_macro, self.source_hidden)
            f_current = self.macro_to_meso(graph, current_macro)
            macro_reconstructed = self.meso_to_macro(f_current)
            reconstruction_outputs.append(macro_reconstructed)
        
        collision_term = self.collision_op(f_current)
        # collision_term = torch.zeros_like(f_current)
        f_next = self.boltzmann_updater(f_current, collision_term, source_term)
        macro_next = self.meso_to_macro(f_next)
        predictions.append(macro_next)

        for step in range(num_pred_steps-1):
            if (self.training and target_sequence is not None and 
                np.random.random() < teacher_forcing_threshold):
                f_current = self.macro_to_meso(graph, target_sequence[step])
                source_input = target_sequence[step]
            else:
                f_current = f_next
                source_input = torch.cat([macro_next,target_sequence[step][...,1:]],dim=-1)
            source_term, self.source_hidden = self.source_rnn(graph, source_input, 
                                                              self.source_hidden)
            collision_term = self.collision_op(f_current)
            # collision_term = torch.zeros_like(f_current)

            f_next = self.boltzmann_updater(f_current, collision_term, source_term)
            
            macro_next = self.meso_to_macro(f_next)
            
            predictions.append(macro_next)
            
        self.source_hidden = None
        predictions = torch.stack(predictions, dim=0)
        
        reconstruction_outputs = torch.stack(reconstruction_outputs, dim=0)
        return predictions, reconstruction_outputs
