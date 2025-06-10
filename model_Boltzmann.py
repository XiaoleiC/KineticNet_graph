import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from boltzmann import VelocitySet, CollisionOperator, FlowWeightCalculator, BoltzmannUpdater, MomentDecoder


class DiffusionGraphConv(nn.Module):
    """
    Diffusion graph convolution layer.
    """
    def __init__(self, in_feats, out_feats, num_hops):
        super(DiffusionGraphConv, self).__init__()
        self.num_hops = num_hops
        self.in_feats = in_feats
        self.out_feats = out_feats
        
        # Learnable weights for each hop
        self.linear = nn.Linear(in_feats * (num_hops + 1), out_feats)
        
    def forward(self, graph, feat):
        """
        Args:
            graph: DGL graph
            feat: Input features [N, in_feats]
        Returns:
            Output features [N, out_feats]
        """
        graph = graph.local_var()
        
        # Normalize adjacency matrix
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        graph.ndata['norm'] = norm.unsqueeze(1)
        
        feat_list = [feat]  # 0-hop (self)
        
        # Multi-hop diffusion
        graph.ndata['h'] = feat
        for _ in range(self.num_hops):
            graph.update_all(fn.u_mul_e('h', 'norm', 'm'), fn.sum('m', 'h_new'))
            graph.ndata['h'] = graph.ndata['h_new'] * graph.ndata['norm']
            feat_list.append(graph.ndata['h'])
        
        # Concatenate all hops
        feat_concat = torch.cat(feat_list, dim=-1)  # [N, in_feats * (num_hops + 1)]
        
        return self.linear(feat_concat)


class StackedEncoder(nn.Module):
    """
    Modified encoder that lifts macroscale variables to mesoscale (Q dimensions).
    """
    def __init__(self, input_dim, velocity_set, hidden_dim=64, num_layers=2, num_hops=2):
        super(StackedEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.Q = velocity_set.Q  # Number of velocity components
        
        # Graph convolution layers
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(DiffusionGraphConv(input_dim, hidden_dim, num_hops))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(DiffusionGraphConv(hidden_dim, hidden_dim, num_hops))
        
        # Last layer: hidden_dim -> Q (mesoscale)
        if num_layers > 1:
            self.layers.append(DiffusionGraphConv(hidden_dim, self.Q, num_hops))
        else:
            # If only one layer, direct mapping
            self.layers[0] = DiffusionGraphConv(input_dim, self.Q, num_hops)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, graph, inputs):
        """
        Args:
            graph: DGL graph
            inputs: Input features [batch_size, N, input_dim]
        Returns:
            Mesoscale distribution [batch_size, N, Q]
        """
        batch_size, num_nodes, _ = inputs.shape
        
        # Reshape for processing: [batch_size * N, input_dim]
        x = inputs.view(-1, self.input_dim)
        
        # Create batched graph
        if batch_size > 1:
            batched_graph = dgl.batch([graph] * batch_size)
        else:
            batched_graph = graph
        
        # Apply graph convolution layers
        for i, layer in enumerate(self.layers):
            x = layer(batched_graph, x)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        # Ensure non-negativity (physical constraint for distribution)
        x = F.softplus(x)
        
        # Reshape back: [batch_size, N, Q]
        x = x.view(batch_size, num_nodes, self.Q)
        
        return x


class BoltzmannCell(nn.Module):
    """
    Boltzmann-based cell replacing traditional GraphGRUCell.
    """
    def __init__(self, velocity_set, collision_hidden_dim=64):
        super(BoltzmannCell, self).__init__()
        self.velocity_set = velocity_set
        self.Q = velocity_set.Q
        
        # Boltzmann components
        self.collision_op = CollisionOperator(velocity_set, collision_hidden_dim)
        self.flow_calculator = FlowWeightCalculator(self.Q)  # feature_dim = Q
        self.boltzmann_updater = BoltzmannUpdater(velocity_set, self.collision_op, self.flow_calculator)
                
    def forward(self, graph, inputs, dt=0.1):
        """
        Single time step update using Boltzmann equation.
        
        Args:
            graph: DGL graph with edge weights (distances)
            inputs: Mesoscale distribution [batch_size, N, Q]
            dt: Time step
        Returns:
            Updated distribution [batch_size, N, Q]
        """
        batch_size, num_nodes, _ = inputs.shape
        
        if batch_size > 1:
            # Process each batch element separately for now
            # TODO: Optimize for batched processing
            outputs = []
            for b in range(batch_size):
                # Extract single batch
                f_single = inputs[b]  # [N, Q]
                
                # Use mesoscale distribution directly as node features
                node_features = f_single  # [N, Q] - no encoding needed
                
                # Apply Boltzmann update
                f_updated = self.boltzmann_updater(graph, f_single, node_features, dt)
                outputs.append(f_updated)
            
            return torch.stack(outputs, dim=0)  # [batch_size, N, Q]
        else:
            # Single batch element
            f_single = inputs.squeeze(0)  # [N, Q]
            node_features = f_single  # [N, Q] - use distribution directly
            f_updated = self.boltzmann_updater(graph, f_single, node_features, dt)
            return f_updated.unsqueeze(0)  # [1, N, Q]


class BoltzmannTrafficFlow(nn.Module):
    """
    Main model class for traffic flow prediction using Boltzmann equation.
    Modified from TrafficFlowRNN to use Boltzmann-based temporal updates.
    """
    def __init__(self, adj_mat, input_dim, velocity_set, hidden_dim=64, 
                 encoder_layers=2, dt=0.1, device='cpu'):
        super(BoltzmannTrafficFlow, self).__init__()
        
        self.adj_mat = adj_mat
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.velocity_set = velocity_set
        self.Q = velocity_set.Q
        self.dt = dt
        self.device = device
        
        # Build graph from adjacency matrix
        self.graph = self._build_graph(adj_mat)
        
        # Model components
        self.encoder = StackedEncoder(
            input_dim=input_dim,
            velocity_set=velocity_set,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers
        )
        
        self.boltzmann_cell = BoltzmannCell(
            velocity_set=velocity_set,
            collision_hidden_dim=hidden_dim,
            flow_feature_dim=hidden_dim
        )
        
        self.decoder = MomentDecoder(velocity_set)
        
    def _build_graph(self, adj_mat):
        """Build DGL graph from adjacency matrix."""
        # Find edges from adjacency matrix
        edges = torch.nonzero(adj_mat, as_tuple=True)
        src, dst = edges
        
        # Create DGL graph
        graph = dgl.graph((src, dst), device=self.device)
        
        # Set edge weights (distances)
        # Use adjacency matrix values as distances
        edge_weights = adj_mat[src, dst]
        graph.edata['weight'] = edge_weights.float()
        
        return graph
        
    def forward(self, inputs, targets=None):
        """
        Forward pass for multi-step prediction.
        
        Args:
            inputs: Historical data [batch_size, input_len, num_nodes, input_dim]
            targets: Target data [batch_size, output_len, num_nodes, input_dim] (optional)
            
        Returns:
            predictions: [batch_size, output_len, num_nodes, input_dim]
            auxiliary_loss: Variance loss from decoder
        """
        batch_size, input_len, num_nodes, input_dim = inputs.shape
        
        # Use last time step as initial condition
        current_state = inputs[:, -1, :, :]  # [batch_size, num_nodes, input_dim]
        
        # Encode to mesoscale
        current_meso = self.encoder(self.graph, current_state)  # [batch_size, num_nodes, Q]
        
        predictions = []
        total_variance_loss = 0.0
        
        # Predict multiple steps (typically 12 steps)
        output_len = targets.shape[1] if targets is not None else 12
        
        for t in range(output_len):
            # Boltzmann temporal update
            next_meso = self.boltzmann_cell(self.graph, current_meso, self.dt)
            
            # Decode to macroscale
            density, velocity, variance_loss = self.decoder(next_meso)
            
            # Combine density and velocity into prediction
            # Assuming input_dim = 2 (density + velocity)
            if input_dim == 2:
                prediction = torch.stack([density, velocity], dim=-1)  # [batch_size, num_nodes, 2]
            else:
                # If more features, just use density for now
                prediction = density.unsqueeze(-1)  # [batch_size, num_nodes, 1]
                if input_dim > 1:
                    # Pad with zeros or extend as needed
                    padding = torch.zeros(batch_size, num_nodes, input_dim - 1, device=density.device)
                    prediction = torch.cat([prediction, padding], dim=-1)
            
            predictions.append(prediction)
            total_variance_loss += variance_loss
            
            # Update current state for next iteration
            current_meso = next_meso
        
        predictions = torch.stack(predictions, dim=1)  # [batch_size, output_len, num_nodes, input_dim]
        avg_variance_loss = total_variance_loss / output_len
        
        return predictions, avg_variance_loss
    
    def get_param_groups(self):
        """
        Get parameter groups for different learning rates.
        """
        # Encoder parameters (graph convolution)
        encoder_params = list(self.encoder.parameters())
        
        # Boltzmann parameters (collision operator, flow calculator)
        boltzmann_params = list(self.boltzmann_cell.parameters())
        
        # Decoder parameters (minimal, mostly explicit calculations)
        decoder_params = list(self.decoder.parameters())
        
        return [
            {"params": encoder_params, "lr": 1e-3, "name": "encoder"},
            {"params": boltzmann_params, "lr": 1e-3, "name": "boltzmann"},
            {"params": decoder_params, "lr": 1e-4, "name": "decoder"}
        ]


# Factory function to create model with predefined settings
def create_boltzmann_traffic_model(adj_mat, input_dim=2, num_velocities=20, max_velocity=2.0,
                                  hidden_dim=64, device='cpu'):
    """
    Factory function to create Boltzmann traffic flow model.
    
    Args:
        adj_mat: Adjacency matrix [N, N]
        input_dim: Number of input features (e.g., 2 for density + velocity)
        num_velocities: Number of velocity components (Q)
        max_velocity: Maximum velocity for traffic flow
        hidden_dim: Hidden dimension for neural networks
        device: Computing device
        
    Returns:
        BoltzmannTrafficFlow model
    """
    # Create velocity set
    velocity_set = VelocitySet(
        num_velocities=num_velocities,
        max_velocity=max_velocity,
        device=device
    )
    
    # Create model
    model = BoltzmannTrafficFlow(
        adj_mat=adj_mat,
        input_dim=input_dim,
        velocity_set=velocity_set,
        hidden_dim=hidden_dim,
        device=device
    )
    
    return model.to(device)


# Example usage
if __name__ == "__main__":
    # Test the model
    device = 'cpu'
    num_nodes = 10
    
    # Create random adjacency matrix
    adj_mat = torch.rand(num_nodes, num_nodes)
    adj_mat = (adj_mat > 0.7).float()  # Sparse connectivity
    adj_mat.fill_diagonal_(0)  # No self loops
    
    # Create model
    model = create_boltzmann_traffic_model(
        adj_mat=adj_mat,
        input_dim=2,
        num_velocities=20,
        device=device
    )
    
    # Test data
    batch_size = 4
    input_len = 12
    output_len = 12
    
    inputs = torch.rand(batch_size, input_len, num_nodes, 2)
    targets = torch.rand(batch_size, output_len, num_nodes, 2)
    
    # Forward pass
    predictions, variance_loss = model(inputs, targets)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Variance loss: {variance_loss.item()}")
    print("Model test passed!")
    
    # Check parameter groups
    param_groups = model.get_param_groups()
    for group in param_groups:
        print(f"{group['name']}: {len(group['params'])} parameter groups")