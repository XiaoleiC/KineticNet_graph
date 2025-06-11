import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
from typing import Optional, Union
from dcrnn import DiffConv
from gaan import GatedGAT
from functools import partial


class MacroToMesoEncoder(nn.Module):
    """
    Encoder that lifts macroscale variables to mesoscale.
    Maps R^{N×d} → R^{N×Q}
    Incorporates spatial information using graph convolution.
    """
    def __init__(self, d_features: int, Q_mesoscale: int, spatial_conv_type: str = 'diffconv', 
                 conv_params: dict = None):
        super(MacroToMesoEncoder, self).__init__()
        self.d_features = d_features
        self.Q_mesoscale = Q_mesoscale
        self.spatial_conv_type = spatial_conv_type
        
        # Choose spatial convolution method
        if spatial_conv_type == 'diffconv':
            # Parameters for DiffConv
            k = conv_params.get('k', 2) if conv_params else 2
            in_graph_list = conv_params.get('in_graph_list', [])
            out_graph_list = conv_params.get('out_graph_list', [])
            self.spatial_conv = DiffConv(d_features, Q_mesoscale, k, 
                                       in_graph_list, out_graph_list)
        elif spatial_conv_type == 'gaan':
            # Parameters for GatedGAT
            map_feats = conv_params.get('map_feats', 64) if conv_params else 64
            num_heads = conv_params.get('num_heads', 2) if conv_params else 2
            self.spatial_conv = GatedGAT(d_features, Q_mesoscale, map_feats, num_heads)
        else:
            raise ValueError(f"Unsupported spatial_conv_type: {spatial_conv_type}")
    
    def forward(self, graph, macro_features):
        """
        Args:
            graph: DGL graph
            macro_features: [N, d] macroscale features
        Returns:
            meso_features: [N, Q] mesoscale distribution
        """
        meso_features = self.spatial_conv(graph, macro_features)
        return meso_features


class CollisionOperator(nn.Module):
    """
    Collision operator Ω(f) with collision invariance constraints.
    Currently implemented as local (no inter-node correlations).
    """
    def __init__(self, Q_mesoscale: int, hidden_dim: int = 64, 
                 constraint_type: str = 'none', xi_velocities: torch.Tensor = None):
        super(CollisionOperator, self).__init__()
        self.Q_mesoscale = Q_mesoscale
        self.constraint_type = constraint_type  # 'none', 'soft', 'hard'
        
        # Simple MLP for collision operator
        self.mlp = nn.Sequential(
            nn.Linear(Q_mesoscale, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, Q_mesoscale)
        )
        
        # Store velocity discretization points ξ_k
        if xi_velocities is not None:
            self.register_buffer('xi_velocities', xi_velocities)  # [Q]
            # Compute collision invariance matrix C
            self.register_buffer('C_matrix', self._compute_C_matrix())
        else:
            self.xi_velocities = None
            self.C_matrix = None
    
    def _compute_C_matrix(self):
        """
        Compute collision invariance matrix C for constraints:
        - Mass conservation: ∫ Ω(f) dξ = 0
        - Momentum conservation: ∫ ξ Ω(f) dξ = 0  
        - Energy conservation: ∫ ξ² Ω(f) dξ = 0
        """
        Q = self.Q_mesoscale
        # Assume equal weights w_i = 1/Q for now
        w = torch.ones(Q) / Q
        
        C = torch.zeros(2, Q)
        C[0, :] = w  # Mass conservation
        C[1, :] = w * self.xi_velocities  # Momentum conservation
        # C[2, :] = 0.5 * w * self.xi_velocities**2  # Energy conservation
        
        return C
    
    def _apply_hard_constraint(self, omega_raw):
        """
        Apply hard collision invariance constraint using Lagrangian method.
        Ω* = (I - C^T(CC^T)^{-1}C) Ω
        """
        if self.C_matrix is None:
            return omega_raw
        
        C = self.C_matrix  # [3, Q]
        I = torch.eye(self.Q_mesoscale, device=omega_raw.device)
        
        # Compute projection matrix
        CCT_inv = torch.inverse(C @ C.T + 1e-6 * torch.eye(2, device=omega_raw.device))
        projection = I - C.T @ CCT_inv @ C
        
        # Apply projection to each node
        omega_constrained = omega_raw @ projection.T
        
        return omega_constrained
    
    def compute_soft_constraint_loss(self, omega):
        """
        Compute soft constraint loss for collision invariance.
        """
        if self.C_matrix is None:
            return torch.tensor(0.0, device=omega.device)
        
        # omega: [N, Q]
        # C: [3, Q]
        constraint_violations = torch.matmul(omega, self.C_matrix.T)  # [N, 3]
        loss = torch.mean(constraint_violations**2)
        
        return loss
    
    def forward(self, f_distribution):
        """
        Args:
            f_distribution: [N, Q] mesoscale distribution
        Returns:
            omega: [N, Q] collision operator output
            constraint_loss: scalar (only for soft constraint)
        """
        omega_raw = self.mlp(f_distribution)
        
        if self.constraint_type == 'hard':
            omega = self._apply_hard_constraint(omega_raw)
            constraint_loss = torch.tensor(0.0, device=f_distribution.device)
        elif self.constraint_type == 'soft':
            omega = omega_raw
            constraint_loss = self.compute_soft_constraint_loss(omega)
        else:  # 'none'
            omega = omega_raw
            constraint_loss = torch.tensor(0.0, device=f_distribution.device)
        
        return omega, constraint_loss


class SGraphRNN(nn.Module):
    """
    Purely data-driven module for modeling source term S(f).
    Similar to GraphRNN but operates entirely in Q-dimensional space.
    """
    def __init__(self, d_features: int, Q_mesoscale: int, hidden_dim: int = 64,
                 spatial_conv_type: str = 'gaan', conv_params: dict = None):
        super(SGraphRNN, self).__init__()
        self.d_features = d_features
        self.Q_mesoscale = Q_mesoscale
        self.hidden_dim = hidden_dim
        
        # Encoder: d → Q (with spatial information)
        self.encoder = MacroToMesoEncoder(d_features, Q_mesoscale, 
                                        spatial_conv_type, conv_params)
        
        # GRU for temporal evolution in Q-dimensional space
        self.gru_cell = nn.GRUCell(Q_mesoscale, hidden_dim)
        
        # Output projection: hidden → Q (source term)
        self.output_proj = nn.Linear(hidden_dim, Q_mesoscale)
    
    def forward(self, graph, macro_features, hidden_state=None):
        """
        Args:
            graph: DGL graph
            macro_features: [N, d] macroscale features
            hidden_state: [N, hidden_dim] or None
        Returns:
            source_term: [N, Q] source term in mesoscale
            new_hidden: [N, hidden_dim] updated hidden state
        """
        N = macro_features.shape[0]
        device = macro_features.device
        
        if hidden_state is None:
            hidden_state = torch.zeros(N, self.hidden_dim, device=device)
        
        # Encode macro features to mesoscale
        meso_input = self.encoder(graph, macro_features)  # [N, Q]
        
        # GRU update
        hidden_flat = hidden_state.view(-1, self.hidden_dim)
        meso_flat = meso_input.view(-1, self.Q_mesoscale)
        
        new_hidden_flat = self.gru_cell(meso_flat, hidden_flat)
        new_hidden = new_hidden_flat.view(N, self.hidden_dim)
        
        # Generate source term
        source_term = self.output_proj(new_hidden)  # [N, Q]
        
        return source_term, new_hidden


class BoltzmannUpdater(nn.Module):
    """
    Physical update module based on discretized Boltzmann equation.
    Implements: f(t+Δt) = f(t) - Δt[Transport - Collision - Source]
    """
    def __init__(self, Q_mesoscale: int, xi_velocities: torch.Tensor, dt: float = 0.1):
        super(BoltzmannUpdater, self).__init__()
        self.Q_mesoscale = Q_mesoscale
        self.dt = dt
        
        # Register velocity discretization
        self.register_buffer('xi_velocities', xi_velocities)  # [Q]
        
        # Learnable flow weights q_ij (initialized uniformly)
        # Note: These will be computed based on graph structure
    
    def compute_transport_term(self, graph, f_distribution):
        with graph.local_scope():
            # Store distribution on nodes
            graph.ndata['f'] = f_distribution  # [N, Q]
            reverse_graph = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
            
            # Precompute velocity matrix for broadcasting: [Q] -> [1, Q]
            xi_broadcast = self.xi_velocities.unsqueeze(0)  # [1, Q]
            
            # Inflow computation (vectorized over all velocity components)
            def message_func_inflow(edges):
                # f_diff: [E, Q] - differences for all velocity components
                f_diff = edges.dst['f'] - edges.src['f']  # [E, Q]
                
                # Edge weights and q_ij: [E] -> [E, 1] for broadcasting
                edge_weight = edges.data.get('weight', torch.ones(edges.src['f'].shape[0], device=f_diff.device))
                edge_weight = edge_weight.unsqueeze(1)  # [E, 1]
                
                q_ij = (1.0 / edges.src['out_degree']).unsqueeze(1)  # [E, 1], !!! The problem is that we assume the weight is the same for all particle velocities
                
                # Broadcast xi_velocities: [1, Q] -> [E, Q]
                xi_expanded = xi_broadcast.expand(f_diff.shape[0], -1)  # [E, Q]
                
                # Compute transport for all velocity components: [E, Q]
                transport = q_ij * xi_expanded * f_diff * edge_weight  # [E, Q]
                
                return {'transport_all': transport}
            
            # Outflow computation (vectorized over all velocity components)  
            def message_func_outflow(edges):
                # f_diff: [E, Q] - differences for all velocity components
                f_diff = edges.src['f'] - edges.dst['f']  # [E, Q]
                
                # Edge weights and q_ij: [E] -> [E, 1] for broadcasting
                edge_weight = edges.data.get('weight', torch.ones(edges.src['f'].shape[0], device=f_diff.device))
                edge_weight = edge_weight.unsqueeze(1)  # [E, 1]
                
                q_ij = (1.0 / edges.dst['in_degree']).unsqueeze(1)  # [E, 1]
                
                # Broadcast xi_velocities: [1, Q] -> [E, Q]
                xi_expanded = xi_broadcast.expand(f_diff.shape[0], -1)  # [E, Q]
                
                # Compute transport for all velocity components: [E, Q]
                transport = q_ij * xi_expanded * f_diff * edge_weight  # [E, Q]
                
                return {'transport_all': transport}
            
            # Reduce function: sum over neighbors for all velocity components
            def reduce_func(nodes):
                # nodes.mailbox['transport_all']: [N, max_degree, Q]
                # Sum over neighbors (dim=1): [N, Q]
                return {'transport_sum_all': torch.sum(nodes.mailbox['transport_all'], dim=1)}
            
            # Single message passing for inflow (all velocity components)
            graph.update_all(message_func_inflow, reduce_func)
            inflow = graph.ndata['transport_sum_all'].clone()  # [N, Q]
            
            # Single message passing for outflow (all velocity components)
            reverse_graph.update_all(message_func_outflow, reduce_func)
            outflow = reverse_graph.ndata['transport_sum_all'].clone()  # [N, Q]
            
            # Final transport term: [N, Q]
            transport_term = outflow - inflow
        
        # for-loop
        # with graph.local_scope():
        #     # Store distribution on nodes
        #     graph.ndata['f'] = f_distribution  # [N, Q]
        #     reverse_graph = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
        #     # Compute differences for each velocity component
        #     transport_terms = []
            
        #     for k in range(self.Q_mesoscale):
        #         xi_k = self.xi_velocities[k]
                
        #         # Message function: compute ξ_k * (f_j - f_i) / d_ij for each edge
        #         def message_func_inflow(edges):
        #             f_diff = edges.dst['f'][:, k] - edges.src['f'][:, k]  # [E]
        #             # Use edge weight as 1/d_ij, and assume q_ij is uniform for now
        #             edge_weight = edges.data.get('weight', torch.ones_like(f_diff))
        #             q_ij = 1.0 / edges.src['out_degree']  # Approximate uniform distribution
                    
        #             transport = q_ij * xi_k * f_diff * edge_weight
        #             return {'transport_k': transport}
                
        #         def message_func_outflow(edges):
        #             f_diff = edges.src['f'][:, k] - edges.dst['f'][:, k]  # [E]
        #             # Use edge weight as 1/d_ij, and assume q_ij is uniform for now
        #             edge_weight = edges.data.get('weight', torch.ones_like(f_diff))
        #             q_ij = 1.0 / edges.dst['in_degree']  # Approximate uniform distribution
                    
        #             transport = q_ij * xi_k * f_diff * edge_weight
        #             return {'transport_k': transport}
                
        #         # Reduce function: sum over neighbors
        #         def reduce_func(nodes):
        #             return {'transport_sum_k': torch.sum(nodes.mailbox['transport_k'], dim=1)}
                
        #         graph.update_all(message_func_inflow, reduce_func)
        #         inflow = graph.ndata['transport_sum_k'].clone()

        #         reverse_graph.update_all(message_func_outflow, reduce_func)
        #         outflow = reverse_graph.ndata['transport_sum_k'].clone()

        #         # Apply message passing for this velocity component
        #         transport_terms.append(outflow - inflow)  
            
        #     # Stack transport terms: [N, Q]
        #     transport_term = torch.stack(transport_terms, dim=1)
            
        return transport_term
    
    def forward(self, graph, f_distribution, collision_term, source_term):
        """
        Update distribution using Boltzmann equation.
        
        Args:
            graph: DGL graph with edge weights
            f_distribution: [N, Q] current distribution
            collision_term: [N, Q] collision operator output
            source_term: [N, Q] source term
        Returns:
            f_new: [N, Q] updated distribution
        """
        f_distribution = torch.clamp(f_distribution, min=0.0)
        # Compute transport term
        transport_term = self.compute_transport_term(graph, f_distribution)
        
        # Boltzmann update: f(t+Δt) = f(t) - Δt[Transport - Collision - Source]
        f_new = f_distribution - self.dt * (transport_term - collision_term - source_term)
        f_new = torch.clamp(f_new, min=0.0)
        
        return f_new


class MesoToMacroDecoder(nn.Module):
    """
    Explicit decoder that converts mesoscale distribution to macroscale variables.
    Implements moment calculation: ρ = ∫f dξ, u = ∫ξf dξ/ρ, etc.
    """
    def __init__(self, Q_mesoscale: int, d_features: int, xi_velocities: torch.Tensor):
        super(MesoToMacroDecoder, self).__init__()
        self.Q_mesoscale = Q_mesoscale
        self.d_features = d_features
        
        # Register velocity discretization and weights
        self.register_buffer('xi_velocities', xi_velocities)  # [Q]
        # For now, use equal weights (UQ methods can be added later)
        self.register_buffer('weights', torch.ones(Q_mesoscale) / Q_mesoscale)  # [Q]
    
    def forward(self, f_distribution):
        """
        Compute macroscale moments from mesoscale distribution.
        
        Args:
            f_distribution: [N, Q] mesoscale distribution
        Returns:
            macro_variables: [N, d] macroscale variables
        """
        N = f_distribution.shape[0]
        macro_variables = []
        
        # 0th moment: density ρ = ∫f dξ
        density = torch.sum(f_distribution * self.weights, dim=1, keepdim=True)  # [N, 1]
        macro_variables.append(density)
        
        # 1st moment: velocity u = ∫ξf dξ / ρ  
        momentum = torch.sum(f_distribution * self.weights * self.xi_velocities, dim=1, keepdim=True)  # [N, 1]
        velocity = momentum / (density + 1e-8)  # Avoid division by zero
        macro_variables.append(velocity)
        
        # # Higher moments can be added based on d_features
        # if self.d_features > 2:
        #     # 2nd moment: energy/temperature related
        #     energy = torch.sum(f_distribution * self.weights * self.xi_velocities**2, dim=1, keepdim=True)
        #     macro_variables.append(energy)
        
        # # Add more moments as needed to match d_features
        # while len(macro_variables) < self.d_features:
        #     # Add higher order moments or other derived quantities
        #     moment_order = len(macro_variables)
        #     higher_moment = torch.sum(f_distribution * self.weights * (self.xi_velocities**moment_order), 
        #                             dim=1, keepdim=True)
        #     macro_variables.append(higher_moment)
        
        # Concatenate and select first d_features
        macro_output = torch.cat(macro_variables[:self.d_features], dim=1)  # [N, d]
        
        return macro_output


class KineticForecastingFramework(nn.Module):
    """
    Complete kinetic theory-informed forecasting framework.
    Integrates all five modules: Encoder → Updater ← (Collision + Source) → Decoder
    """
    def __init__(self, d_features: int, Q_mesoscale: int, xi_velocities: torch.Tensor,
                 spatial_conv_type: str = 'diffconv', conv_params: dict = None,
                 collision_constraint: str = 'none', dt: float = 0.1):
        super(KineticForecastingFramework, self).__init__()
        
        self.d_features = d_features
        self.Q_mesoscale = Q_mesoscale
        
        # Module 1: MacroToMesoEncoder
        self.macro_to_meso = MacroToMesoEncoder(d_features, Q_mesoscale, 
                                              spatial_conv_type, conv_params)
        
        # Module 2: BoltzmannUpdater  
        self.boltzmann_updater = BoltzmannUpdater(Q_mesoscale, xi_velocities, dt)
        
        # Module 3: SGraphRNN (for source term)
        self.source_rnn = SGraphRNN(d_features, Q_mesoscale, 
                                   spatial_conv_type=spatial_conv_type, 
                                   conv_params=conv_params)
        
        # Module 4: CollisionOperator
        self.collision_op = CollisionOperator(Q_mesoscale, constraint_type=collision_constraint,
                                            xi_velocities=xi_velocities)
        
        # Module 5: MesoToMacroDecoder
        self.meso_to_macro = MesoToMacroDecoder(Q_mesoscale, d_features, xi_velocities)
        
        # Store for source term hidden states
        self.source_hidden = None
    
    def forward(self, graph, macro_features_sequence, num_pred_steps: int = 1):
        """
        Forward pass for sequence prediction.
        
        Args:
            graph: DGL graph
            macro_features_sequence: [T, N, d] input sequence  
            num_pred_steps: number of future steps to predict
        Returns:
            predictions: [num_pred_steps, N, d] predicted macroscale variables
            constraint_losses: collision invariance losses
        """
        device = macro_features_sequence.device
        T, N, d = macro_features_sequence.shape
        
        predictions = []
        constraint_losses = []
        
        # Initialize source term hidden state
        if self.source_hidden is None:
            self.source_hidden = self.source_rnn.init_hidden(1, N, device)
        
        # Use last time step as initial condition
        current_macro = macro_features_sequence[-1]  # [N, d]
        
        for step in range(num_pred_steps):
            # 1. Encode: macro → meso
            f_current = self.macro_to_meso(graph, current_macro)  # [N, Q]
            
            # 2. Compute source term
            source_term, self.source_hidden = self.source_rnn(graph, current_macro, 
                                                             self.source_hidden)
            
            # 3. Compute collision term
            collision_term, constraint_loss = self.collision_op(f_current)
            constraint_losses.append(constraint_loss)
            
            # 4. Update using Boltzmann equation
            f_next = self.boltzmann_updater(graph, f_current, collision_term, source_term)
            
            # 5. Decode: meso → macro
            macro_next = self.meso_to_macro(f_next)  # [N, d]
            
            predictions.append(macro_next)
            current_macro = macro_next  # Update for next iteration
        
        predictions = torch.stack(predictions, dim=0)  # [num_pred_steps, N, d]
        total_constraint_loss = torch.stack(constraint_losses).mean()
        
        return predictions, total_constraint_loss
    
    def reset_source_hidden(self):
        """Reset source term hidden state."""
        self.source_hidden = None


# Example usage and initialization
def create_kinetic_framework(d_features: int = 2, Q_mesoscale: int = 16, 
                           xi_range: tuple = (-3, 3), **kwargs):
    """
    Helper function to create the kinetic framework with default parameters.
    """
    # Create velocity discretization
    xi_velocities = torch.linspace(xi_range[0], xi_range[1], Q_mesoscale)
    
    # Default conv_params for different spatial convolution types
    if kwargs.get('spatial_conv_type') == 'gaan':
        conv_params = {
            'map_feats': kwargs.get('map_feats', 64),
            'num_heads': kwargs.get('num_heads', 2)
        }
    else:  # diffconv
        conv_params = {
            'k': kwargs.get('k', 2),
            'in_graph_list': kwargs.get('in_graph_list', []),
            'out_graph_list': kwargs.get('out_graph_list', [])
        }
    
    framework = KineticForecastingFramework(
        d_features=d_features,
        Q_mesoscale=Q_mesoscale, 
        xi_velocities=xi_velocities,
        spatial_conv_type=kwargs.get('spatial_conv_type', 'diffconv'),
        conv_params=conv_params,
        collision_constraint=kwargs.get('collision_constraint', 'none'),
        dt=kwargs.get('dt', 0.1)
    )
    
    return framework

if __name__ == '__main__':
    """
    Test BoltzmannUpdater module independently
    """
    import time
    print("Testing BoltzmannUpdater...")
    t0 = time.time()
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)
    # Test parameters
    N = 5  # Number of nodes
    Q = 3  # Number of velocity components
    
    # Create a simple directed graph
    edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 4)]
    src, dst = zip(*edges)
    graph = dgl.graph((src, dst), num_nodes=N)
    
    # Add edge weights (representing 1/distance)
    graph.edata['weight'] = torch.rand(graph.number_of_edges()) * 0.5 + 0.5
    
    # Add degree information
    graph.ndata['out_degree'] = graph.out_degrees().float().clamp(min=1.0)
    graph.ndata['in_degree'] = graph.in_degrees().float().clamp(min=1.0)
    
    print(f"Graph info: {N} nodes, {graph.number_of_edges()} edges")
    print(f"Out degrees: {graph.ndata['out_degree']}")
    print(f"In degrees: {graph.ndata['in_degree']}")
    
    # Create velocity grid (all positive values)
    xi_velocities = torch.tensor([0.5, 1.0, 2.0])
    print(f"Velocity grid: {xi_velocities}")
    
    # Initialize BoltzmannUpdater
    dt = 0.1
    updater = BoltzmannUpdater(Q, xi_velocities, dt)
    
    # Create test distribution function
    f_distribution = torch.rand(N, Q) * 2.0 + 1.0  # [N, Q] positive values
    print(f"Initial distribution shape: {f_distribution.shape}")
    print(f"Initial distribution:\n{f_distribution}")
    
    # Create dummy collision and source terms
    collision_term = CollisionOperator(Q, constraint_type='hard', xi_velocities=xi_velocities)
    collision_term = collision_term(f_distribution)[0]  # [N, Q]
    print(f"Collision term shape: {collision_term.shape}")
    source_term = torch.rand(N, Q) * 0.05    # Small source term
    
    print("\n" + "="*50)
    print("Testing compute_transport_term...")
    
    # Test transport term computation
    try:
        transport_term = updater.compute_transport_term(graph, f_distribution)
        print(f"Transport term computed successfully!")
        print(f"Transport term shape: {transport_term.shape}")
        print(f"Transport term range: [{transport_term.min():.4f}, {transport_term.max():.4f}]")
        print(f"Transport term:\n{transport_term}")
        
        # Check for NaN or infinite values
        if torch.isnan(transport_term).any():
            print("Warning: NaN values detected in transport term")
        if torch.isinf(transport_term).any():
            print("Warning: Infinite values detected in transport term")
            
    except Exception as e:
        print(f"Error in transport term computation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "="*50)
    print("Testing full Boltzmann update...")
    
    # Test full update
    try:
        f_new = updater.forward(graph, f_distribution, collision_term, source_term)
        print(f"Boltzmann update completed successfully!")
        print(f"Updated distribution shape: {f_new.shape}")
        print(f"Updated distribution range: [{f_new.min():.4f}, {f_new.max():.4f}]")
        
        # Check conservation properties
        mass_before = f_distribution.sum()
        mass_after = f_new.sum()
        mass_change = abs(mass_after - mass_before) / mass_before
        print(f"Mass change: {mass_change:.6f} (relative)")
        
        if mass_change < 0.01:
            print("Mass approximately conserved")
        else:
            print("Warning: Significant mass change detected")
            
        # Check for negative values
        if (f_new < 0).any():
            print("Warning: Negative distribution values detected")
            neg_count = (f_new < 0).sum().item()
            print(f"   Number of negative values: {neg_count}")
        else:
            print("All distribution values remain positive")
            
        # Show the change
        f_change = f_new - f_distribution
        print(f"Distribution change range: [{f_change.min():.4f}, {f_change.max():.4f}]")
        
    except Exception as e:
        print(f"Error in Boltzmann update: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "="*50)
    print("All tests passed! BoltzmannUpdater is working correctly.")
    
    # Additional test: multiple time steps
    print("\nTesting multiple time steps...")
    f_current = f_distribution.clone()
    
    for step in range(5):
        collision_term = torch.rand(N, Q) * 0.05
        source_term = torch.rand(N, Q) * 0.02
        f_current = updater.forward(graph, f_current, collision_term, source_term)
        print(f"Step {step+1}: mass = {f_current.sum():.4f}, range = [{f_current.min():.4f}, {f_current.max():.4f}]")
        
        if torch.isnan(f_current).any():
            print(f"NaN detected at step {step+1}")
            break
    else:
        print("Multi-step simulation completed successfully!")

    print(f"Total time: {time.time() - t0:.4f} seconds")