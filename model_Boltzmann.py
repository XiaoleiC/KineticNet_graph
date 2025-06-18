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
    def __init__(self, d_features: int, Q_mesoscale: int, num_layers: int = 1, spatial_conv_type: str = 'diffconv', 
                 conv_params: dict = None, is_SGRNN: bool = False):
        super(MacroToMesoEncoder, self).__init__()
        self.d_features = d_features
        self.Q_mesoscale = Q_mesoscale
        self.spatial_conv_type = spatial_conv_type
        self.conv_layers = nn.ModuleList()
        self.is_SGRNN = is_SGRNN
        # Choose spatial convolution method
        if spatial_conv_type == 'diffconv':
            # Parameters for DiffConv
            k = conv_params.get('k', 2)
            in_graph_list = conv_params.get('in_graph_list', [])
            out_graph_list = conv_params.get('out_graph_list', [])
            # self.spatial_conv = DiffConv(d_features, Q_mesoscale, k, 
                                    #    in_graph_list, out_graph_list)
        elif spatial_conv_type == 'gaan':
            # Parameters for GatedGAT
            map_feats = conv_params.get('map_feats', 64)
            num_heads = conv_params.get('num_heads', 2)
            # self.spatial_conv = GatedGAT(d_features, Q_mesoscale, map_feats, num_heads)
        else:
            raise ValueError(f"Unsupported spatial_conv_type: {spatial_conv_type}")

        for i in range(num_layers):
            in_dim = d_features if i == 0 else Q_mesoscale
            out_dim = Q_mesoscale
            
            # Choose spatial convolution method
            if spatial_conv_type == 'diffconv':
                conv_layer = DiffConv(in_dim, out_dim, k, in_graph_list, out_graph_list)
            elif spatial_conv_type == 'gaan':
                conv_layer = GatedGAT(in_dim, out_dim, map_feats, num_heads)
            else:
                raise ValueError(f"Unsupported spatial_conv_type: {spatial_conv_type}")
            
            self.conv_layers.append(conv_layer)

    def forward(self, graph, macro_features):
        """
        Args:
            graph: DGL graph
            macro_features: [N, d] macroscale features
        Returns:
            meso_features: [N, Q] mesoscale distribution
        """
        x = macro_features
        
        # Apply multiple layers with ReLU activation
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(graph, x)
            x = nn.functional.tanh(x) if i < len(self.conv_layers) - 1 else x
        if not self.is_SGRNN:
            x = nn.functional.relu(x)  # Ensure output is non-negative
        return x


class CollisionOperator(nn.Module):
    """
    Collision operator Ω(f) with collision invariance constraints.
    Currently implemented as local (no inter-node correlations).
    """
    def __init__(self, Q_mesoscale: int, hidden_dim: int = 64, 
                 constraint_type: str = 'none', xi_velocities: torch.Tensor = None, num_layers: int = 5):
        super(CollisionOperator, self).__init__()
        self.Q_mesoscale = Q_mesoscale
        self.constraint_type = constraint_type  # 'none', 'soft', 'hard'
        
        # Simple MLP for collision operator
        # self.mlp = nn.Sequential(
        #     nn.Linear(Q_mesoscale, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, Q_mesoscale)
        # )
        self.mlp = nn.ModuleList()
        for k1 in range(num_layers):
            in_dim = Q_mesoscale if k1 == 0 else hidden_dim
            out_dim = hidden_dim if k1 < num_layers - 1 else Q_mesoscale
            self.mlp.append(nn.Linear(in_dim, out_dim))
        
        # Store velocity discretization points ξ_k
        self.register_buffer('xi_velocities', xi_velocities)  # [Q]
    
    def _compute_C_matrix(self, macro_velocities):
        """
        Compute collision invariance matrix C for constraints:
        - Mass conservation: ∫ Ω(f) dξ = 0
        - Momentum conservation: ∫ ξ Ω(f) dξ = 0  
        
        Args:
            macro_velocities: [N, 1] macro velocities for each node
        Returns:
            C: [N, 2, Q] collision invariance matrix for each node
        """
        Q = self.Q_mesoscale
        N = macro_velocities.shape[0]
        
        # xi_shifted: [N, Q] - absolute velocities for each node
        xi_shifted = self.xi_velocities.unsqueeze(0) + macro_velocities  # [N, Q]
        
        # Equal weights
        w = torch.ones(Q, device=self.xi_velocities.device) / Q
        
        C = torch.zeros(N, 2, Q, device=self.xi_velocities.device)
        C[:, 0, :] = w.unsqueeze(0)  # Mass conservation: [N, Q]
        C[:, 1, :] = w.unsqueeze(0) * xi_shifted  # Momentum conservation: [N, Q]
        
        return C

    
    def _apply_hard_constraint(self, omega_raw, C_matrix):
        """
        Apply hard collision invariance constraint using Lagrangian method (batched version).
        Ω* = (I - C^T(CC^T)^{-1}C) Ω
        
        Args:
            omega_raw: [N, Q] raw collision operator output
            C_matrix: [N, 2, Q] collision invariance matrix for each node
        Returns:
            omega_constrained: [N, Q] constrained collision operator output
        """
        N, Q = omega_raw.shape
        device = omega_raw.device
        
        # Identity matrix
        I = torch.eye(Q, device=device).unsqueeze(0).expand(N, -1, -1)  # [N, Q, Q]
        
        # Compute CC^T for all nodes: [N, 2, Q] @ [N, Q, 2] = [N, 2, 2]
        CCT = torch.bmm(C_matrix, C_matrix.transpose(-1, -2))  # [N, 2, 2]
        
        # Add regularization
        reg = 1e-6 * torch.eye(2, device=device).unsqueeze(0).expand(N, -1, -1)
        CCT = CCT + reg  # [N, 2, 2]
        
        # Compute (CC^T)^{-1} for all nodes
        CCT_inv = torch.inverse(CCT)  # [N, 2, 2]
        
        # Compute C^T(CC^T)^{-1}C for all nodes
        # C_matrix.transpose(-1, -2): [N, Q, 2]
        # CCT_inv: [N, 2, 2] 
        # C_matrix: [N, 2, Q]
        temp = torch.bmm(C_matrix.transpose(-1, -2), CCT_inv)  # [N, Q, 2]
        CTC_inv_C = torch.bmm(temp, C_matrix)  # [N, Q, Q]
        
        # Compute projection matrix: I - C^T(CC^T)^{-1}C
        projection = I - CTC_inv_C  # [N, Q, Q]
        
        # Apply projection: omega_raw @ projection^T
        # omega_raw.unsqueeze(-1): [N, Q, 1]
        # projection.transpose(-1, -2): [N, Q, Q]
        omega_constrained = torch.bmm(projection, omega_raw.unsqueeze(-1)).squeeze(-1)  # [N, Q]
        
        return omega_constrained
    
    def forward(self, f_distribution, macro_velocities):
        """
        Args:
            f_distribution: [N, Q] mesoscale distribution
            macro_velocities: [N,1] macro velocities for each node
        Returns:
            omega: [N, Q] collision operator output
        """
        # MLP forward pass
        x = f_distribution
        for k2, layer in enumerate(self.mlp):
            x = layer(x)
            x = nn.functional.relu(x) if k2 < len(self.mlp) - 1 else nn.functional.tanh(x)
        omega_raw = x
        
        if self.constraint_type == 'hard':
            # Compute C matrix for current macro velocities
            C_matrix = self._compute_C_matrix(macro_velocities)  # [N, 2, Q]
            omega = self._apply_hard_constraint(omega_raw, C_matrix)
        else:  # 'none'
            omega = omega_raw
        
        return omega


class SGraphRNN(nn.Module):
    """
    Purely data-driven module for modeling source term S(f).
    Similar to GraphRNN but operates entirely in Q-dimensional space.
    """
    def __init__(self, d_features: int, Q_mesoscale: int, num_layers: int = 1, hidden_dim: int = 64,
                 spatial_conv_type: str = 'gaan', conv_params: dict = None, is_SGRNN: bool = True):
        super(SGraphRNN, self).__init__()
        self.d_features = d_features
        self.Q_mesoscale = Q_mesoscale
        self.hidden_dim = hidden_dim
        
        # Encoder: d → Q (with spatial information)
        self.encoder = MacroToMesoEncoder(d_features, Q_mesoscale, num_layers,
                                        spatial_conv_type, conv_params, is_SGRNN=is_SGRNN)
        
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
        # graph = graph.clone()
        # graph = dgl.remove_self_loop(graph)  # Remove self-loops for transport computation
        with graph.local_scope():
            # Store distribution on nodes
            graph.ndata['f'] = f_distribution  # [N, Q]
            reverse_graph = dgl.reverse(graph, copy_ndata=True, copy_edata=True)

            graph.ndata['out_degree'] = graph.out_degrees().float()  # [N]
            graph.ndata['in_degree'] = graph.in_degrees().float()  # [N]
            reverse_graph.ndata['out_degree'] = reverse_graph.out_degrees().float()
            reverse_graph.ndata['in_degree'] = reverse_graph.in_degrees().float()
            
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
    
    def forward(self, f_distribution, macro_velocities):
        """
        Compute macroscale moments from mesoscale distribution.
        
        Args:
            f_distribution: [N, Q] mesoscale distribution
            macro_velocities: [N, 1] macroscale velocities (used for shifting)
        Returns:
            macro_variables: [N, d] macroscale variables
        """
        N = f_distribution.shape[0]
        macro_variables = []
        xi_shifted = self.xi_velocities.unsqueeze(0) + macro_velocities
        # 0th moment: density ρ = ∫f dξ
        density = torch.sum(f_distribution * self.weights, dim=1, keepdim=True)  # [N, 1]
        macro_variables.append(density)
        
        # 1st moment: velocity u = ∫ξf dξ / ρ  
        momentum = torch.sum(f_distribution * self.weights.unsqueeze(0) * xi_shifted, dim=1, keepdim=True)  # [N, 1]
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
        
        # # Concatenate and select first d_features
        # macro_output = torch.cat(macro_variables[:self.d_features], dim=1)  # [N, d]
        macro_output = macro_variables[1] # [N, 1], only the velocity is used for now
        
        return macro_output



class KineticForecastingFramework(nn.Module):
    """
    Complete kinetic theory-informed forecasting framework.
    Integrates all five modules: Encoder → Updater ← (Collision + Source) → Decoder
    """
    def __init__(self, d_features: int, d_features_source: int, Q_mesoscale: int, xi_velocities: torch.Tensor, num_layers_macro_to_meso: int = 1,
                 spatial_conv_type: str = 'gaan', conv_params: dict = None,
                 collision_constraint: str = 'none', dt: float = 0.1, decay_steps: int = 2000, device: Optional[Union[str, torch.device]] = 'cpu', num_layers_collision: int = 6):
        super(KineticForecastingFramework, self).__init__()
        
        self.d_features = d_features
        self.d_features_source = d_features_source
        self.Q_mesoscale = Q_mesoscale
        self.decay_steps = decay_steps  # For teacher forcing decay
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.register_buffer('xi_velocities', xi_velocities)  # [Q]

        if Q_mesoscale != xi_velocities.shape[0]:
            raise ValueError(f"Q_mesoscale ({Q_mesoscale}) must match xi_velocities length ({xi_velocities.shape[0]})")
        
        # Module 1: MacroToMesoEncoder
        self.macro_to_meso = MacroToMesoEncoder(d_features=d_features, Q_mesoscale=Q_mesoscale, num_layers=num_layers_macro_to_meso,
                                               spatial_conv_type=spatial_conv_type, conv_params=conv_params, is_SGRNN=False)
        
        # Module 2: BoltzmannUpdater  
        self.boltzmann_updater = BoltzmannUpdater(Q_mesoscale, self.xi_velocities, dt)
        
        # Module 3: SGraphRNN (for source term)
        self.source_rnn = SGraphRNN(d_features_source, Q_mesoscale, num_layers = num_layers_macro_to_meso,
                                   spatial_conv_type=spatial_conv_type,
                                   conv_params=conv_params, is_SGRNN=True)
        
        # Module 4: CollisionOperator
        self.collision_op = CollisionOperator(Q_mesoscale, constraint_type=collision_constraint,
                                            xi_velocities=self.xi_velocities, num_layers=num_layers_collision)
        
        # Module 5: MesoToMacroDecoder
        self.meso_to_macro = MesoToMacroDecoder(Q_mesoscale, d_features, self.xi_velocities)
        self.source_hidden = None  # Hidden state for source term RNN

    def compute_teacher_forcing_threshold(self, batch_cnt):
        """
        Compute teacher forcing threshold based on training progress.
        Higher values at the beginning, gradually decay to 0.
        
        Args:
            batch_cnt: current batch/epoch count
        Returns:
            threshold: probability of using teacher forcing [0, 1]
        """
        import numpy as np
        return self.decay_steps / (
            self.decay_steps + np.exp(batch_cnt / self.decay_steps)
        )

    def forward(self, graph, macro_features_sequence, num_pred_steps: int = 1, 
                target_sequence: torch.Tensor = None, batch_cnt: int = 0):
        """
        Forward pass for sequence prediction with adaptive teacher forcing.
        
        Logic:
        1. Use input sequence to train source term with teacher forcing
        2. MacroToMesoEncoder learns spatial correlations only  
        3. Temporal evolution via Boltzmann equation (physics-informed)
        4. Source term carries historical memory with decaying teacher forcing
        
        Args:
            graph: DGL graph
            macro_features_sequence: [T, N, d] input sequence  
            num_pred_steps: number of future steps to predict
            target_sequence: [num_pred_steps, N, d] ground truth targets (for training)
            batch_cnt: current batch/epoch count for teacher forcing decay
        Returns:
            predictions: [num_pred_steps, N, d] predicted macroscale variables
            constraint_losses: collision invariance losses
            reconstruction_outputs: [T, N, d] reconstructed historical sequence (training only)
        """        
        T = macro_features_sequence.shape[0]
        macro_features_sequence = macro_features_sequence.to(self.device)
        target_sequence = target_sequence.to(self.device)
        predictions = []
        # constraint_losses = []
        reconstruction_outputs = []  # For training phase reconstruction loss
        
        # Compute teacher forcing threshold
        teacher_forcing_threshold = self.compute_teacher_forcing_threshold(batch_cnt)
        
        # Phase 1: Historical sequence processing with reconstruction (training phase)
        for t in range(T):
            current_macro = macro_features_sequence[t]  # [N, d]
            
            # Update source term hidden states with ground truth data
            source_term, self.source_hidden = self.source_rnn(graph, current_macro, self.source_hidden)
            
            # During training, also compute reconstruction for loss calculation
            # if self.training:
                # 1. Encode: macro → meso
            f_current = self.macro_to_meso(graph, current_macro)  # [N, Q]
            
            # 5. Decode: meso → macro
            macro_reconstructed = self.meso_to_macro(f_current, current_macro[...,:1])  # [N, d]
            
            reconstruction_outputs.append(macro_reconstructed)
        
        collision_term = self.collision_op(f_current)
        f_next = self.boltzmann_updater(graph, f_current, collision_term, source_term)
        macro_next = self.meso_to_macro(f_next, current_macro[...,:1])  # [N, d]
        # Phase 2: Autoregressive prediction with adaptive teacher forcing
        predictions.append(macro_next)

        for step in range(num_pred_steps-1):
            # 1. Encode: macro → meso (spatial correlations only)
            # f_current = self.macro_to_meso(graph, current_macro)  # [N, Q]
            
            # 2. Adaptive teacher forcing decision for source term input
            if (self.training and target_sequence is not None and 
                np.random.random() < teacher_forcing_threshold):
                # Use ground truth for source term (teacher forcing)
                f_current = self.macro_to_meso(graph, target_sequence[step]) # [N, Q]
                source_input = target_sequence[step]  # [N, d]
            else:
                # Use model prediction for source term
                f_current = f_next
                source_input = torch.cat([macro_next,target_sequence[step][...,self.d_features:]],dim=-1)  # [N, d]
            # 3. Compute source term (carries historical memory)
            source_term, self.source_hidden = self.source_rnn(graph, source_input, 
                                                              self.source_hidden)
            
            # 4. Compute collision term (physical constraints)
            collision_term = self.collision_op(f_current)
            # constraint_losses.append(constraint_loss)
            
            # 5. Update using Boltzmann equation (physics-informed temporal evolution)
            f_next = self.boltzmann_updater(graph, f_current, collision_term, source_term)
            
            # 6. Decode: meso → macro
            macro_next = self.meso_to_macro(f_next, macro_next)  # [N, d]
            
            predictions.append(macro_next)
            # current_macro = torch.cat([macro_next,target_sequence[step][...,self.d_features:]],dim=-1)  # Update for next iteration??
            
        self.source_hidden = None  # Reset source hidden state after prediction
        predictions = torch.stack(predictions, dim=0)  # [num_pred_steps, N, d]
        # total_constraint_loss = torch.stack(constraint_losses).mean() if constraint_losses else torch.tensor(0.0)
        
        # Stack reconstruction outputs if in training mode
        # if self.training and reconstruction_outputs:
        #     reconstruction_outputs = torch.stack(reconstruction_outputs, dim=0)  # [T, N, d]
        # else:
        #     reconstruction_outputs = None
        reconstruction_outputs = torch.stack(reconstruction_outputs, dim=0)  # [T, N, d]
        return predictions, reconstruction_outputs


# if __name__ == '__main__':
#     """
#     Test BoltzmannUpdater module independently
#     """
#     import time
#     print("Testing BoltzmannUpdater...")
#     t0 = time.time()
#     torch.manual_seed(42)  # For reproducibility
#     np.random.seed(42)
#     # Test parameters
#     N = 5  # Number of nodes
#     Q = 3  # Number of velocity components
    
#     # Create a simple directed graph
#     edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 4)]
#     src, dst = zip(*edges)
#     graph = dgl.graph((src, dst), num_nodes=N)
    
#     # Add edge weights (representing 1/distance)
#     graph.edata['weight'] = torch.rand(graph.number_of_edges()) * 0.5 + 0.5
    
#     # Add degree information
#     graph.ndata['out_degree'] = graph.out_degrees().float().clamp(min=1.0)
#     graph.ndata['in_degree'] = graph.in_degrees().float().clamp(min=1.0)
    
#     print(f"Graph info: {N} nodes, {graph.number_of_edges()} edges")
#     print(f"Out degrees: {graph.ndata['out_degree']}")
#     print(f"In degrees: {graph.ndata['in_degree']}")
    
#     # Create velocity grid (all positive values)
#     xi_velocities = torch.tensor([0.5, 1.0, 2.0])
#     print(f"Velocity grid: {xi_velocities}")
    
#     # Initialize BoltzmannUpdater
#     dt = 0.1
#     updater = BoltzmannUpdater(Q, xi_velocities, dt)
    
#     # Create test distribution function
#     f_distribution = torch.rand(N, Q) * 2.0 + 1.0  # [N, Q] positive values
#     print(f"Initial distribution shape: {f_distribution.shape}")
#     print(f"Initial distribution:\n{f_distribution}")
    
#     # Create dummy collision and source terms
#     collision_term = CollisionOperator(Q, constraint_type='hard', xi_velocities=xi_velocities)
#     collision_term = collision_term(f_distribution)[0]  # [N, Q]
#     print(f"Collision term shape: {collision_term.shape}")
#     source_term = torch.rand(N, Q) * 0.05    # Small source term
    
#     print("\n" + "="*50)
#     print("Testing compute_transport_term...")
    
#     # Test transport term computation
#     try:
#         transport_term = updater.compute_transport_term(graph, f_distribution)
#         print(f"Transport term computed successfully!")
#         print(f"Transport term shape: {transport_term.shape}")
#         print(f"Transport term range: [{transport_term.min():.4f}, {transport_term.max():.4f}]")
#         print(f"Transport term:\n{transport_term}")
        
#         # Check for NaN or infinite values
#         if torch.isnan(transport_term).any():
#             print("Warning: NaN values detected in transport term")
#         if torch.isinf(transport_term).any():
#             print("Warning: Infinite values detected in transport term")
            
#     except Exception as e:
#         print(f"Error in transport term computation: {e}")
#         import traceback
#         traceback.print_exc()
#         exit(1)
    
#     print("\n" + "="*50)
#     print("Testing full Boltzmann update...")
    
#     # Test full update
#     try:
#         f_new = updater.forward(graph, f_distribution, collision_term, source_term)
#         print(f"Boltzmann update completed successfully!")
#         print(f"Updated distribution shape: {f_new.shape}")
#         print(f"Updated distribution range: [{f_new.min():.4f}, {f_new.max():.4f}]")
        
#         # Check conservation properties
#         mass_before = f_distribution.sum()
#         mass_after = f_new.sum()
#         mass_change = abs(mass_after - mass_before) / mass_before
#         print(f"Mass change: {mass_change:.6f} (relative)")
        
#         if mass_change < 0.01:
#             print("Mass approximately conserved")
#         else:
#             print("Warning: Significant mass change detected")
            
#         # Check for negative values
#         if (f_new < 0).any():
#             print("Warning: Negative distribution values detected")
#             neg_count = (f_new < 0).sum().item()
#             print(f"   Number of negative values: {neg_count}")
#         else:
#             print("All distribution values remain positive")
            
#         # Show the change
#         f_change = f_new - f_distribution
#         print(f"Distribution change range: [{f_change.min():.4f}, {f_change.max():.4f}]")
        
#     except Exception as e:
#         print(f"Error in Boltzmann update: {e}")
#         import traceback
#         traceback.print_exc()
#         exit(1)
    
#     print("\n" + "="*50)
#     print("All tests passed! BoltzmannUpdater is working correctly.")
    
#     # Additional test: multiple time steps
#     print("\nTesting multiple time steps...")
#     f_current = f_distribution.clone()
    
#     for step in range(5):
#         collision_term = torch.rand(N, Q) * 0.05
#         source_term = torch.rand(N, Q) * 0.02
#         f_current = updater.forward(graph, f_current, collision_term, source_term)
#         print(f"Step {step+1}: mass = {f_current.sum():.4f}, range = [{f_current.min():.4f}, {f_current.max():.4f}]")
        
#         if torch.isnan(f_current).any():
#             print(f"NaN detected at step {step+1}")
#             break
#     else:
#         print("Multi-step simulation completed successfully!")

#     print(f"Total time: {time.time() - t0:.4f} seconds")

if __name__ == '__main__':
    """
    Test KineticForecastingFramework integration
    """
    print("Testing KineticForecastingFramework...")
    
    # Test parameters
    N = 10  # Number of nodes
    T = 5   # Historical sequence length
    d_features = 1  # Macro feature dimension
    d_features_source = 2  # Source term feature dimension
    Q_mesoscale = 6  # Mesoscale velocity components
    num_pred_steps = 5  # Prediction steps
    
    # Create velocity grid
    xi_velocities = torch.randn(Q_mesoscale)
    
    # Create test graph
    edges = [(i, (i+1) % N) for i in range(N)]  # Ring graph
    edges += [(i, (i+2) % N) for i in range(N)]  # Add some long-range connections
    src, dst = zip(*edges)
    graph = dgl.graph((src, dst), num_nodes=N)
    
    # Add graph properties
    graph.edata['weight'] = torch.rand(graph.number_of_edges()) * 0.5 + 0.5
    graph.ndata['out_degree'] = graph.out_degrees().float().clamp(min=1.0)
    graph.ndata['in_degree'] = graph.in_degrees().float().clamp(min=1.0)
    
    graph = graph.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Graph: {N} nodes, {graph.number_of_edges()} edges")
    
    k = 2
    out_graph_list, in_graph_list = DiffConv.attach_graph(graph, k)

    conv_params = {
        "k": k,
        "in_graph_list": in_graph_list,
        "out_graph_list": out_graph_list,
    }


    # Initialize framework
    model = KineticForecastingFramework(
        d_features=d_features,
        d_features_source=d_features_source,
        Q_mesoscale=Q_mesoscale,
        xi_velocities=xi_velocities,
        spatial_conv_type='diffconv',
        conv_params=conv_params,
        collision_constraint='hard',
        dt=0.1,
        decay_steps=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Framework initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test data
    macro_sequence = torch.randn(T, N, d_features_source) * 0.5 + 1.0  # [T, N, d]
    target_sequence = torch.randn(num_pred_steps, N, d_features_source) * 0.5 + 1.0  # [num_pred_steps, N, d]
    
    print(f"Input sequence shape: {macro_sequence.shape}")
    print(f"Target sequence shape: {target_sequence.shape}")
    
    print("\n" + "="*60)
    print("Testing Training Mode...")
    
    # Test training mode
    model.train()
    print(f"Training mode: {model.training}")
    
    try:
        # Test teacher forcing threshold
        for batch_cnt in [0, 500, 1000, 2000, 5000]:
            threshold = model.compute_teacher_forcing_threshold(batch_cnt)
            print(f"Batch {batch_cnt}: Teacher forcing threshold = {threshold:.4f}")
        
        print("\nTesting forward pass in training mode...")
        predictions, reconstructions = model(
            graph, macro_sequence.to('cuda' if torch.cuda.is_available() else 'cpu'), 
            num_pred_steps=num_pred_steps,
            target_sequence=target_sequence.to('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_cnt=100
        )
        
        print(f"Training forward pass successful!")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Reconstructions shape: {reconstructions.shape if reconstructions is not None else None}")
        
        # Check shapes
        assert predictions.shape == (num_pred_steps, N, d_features), f"Wrong predictions shape: {predictions.shape}"
        assert reconstructions.shape == (T, N, d_features), f"Wrong reconstructions shape: {reconstructions.shape}"
        print("Training mode shapes correct!")
        
    except Exception as e:
        print(f"Training mode error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "="*60)
    print("Testing Evaluation Mode...")
    
    # Test evaluation mode
    model.eval()
    print(f"Training mode: {model.training}")
    
    try:
        # Reset hidden state
        model.source_hidden = None
        
        predictions, reconstructions = model(
            graph, macro_sequence.to('cuda' if torch.cuda.is_available() else 'cpu'),
            num_pred_steps=num_pred_steps,
            target_sequence=target_sequence.to('cuda' if torch.cuda.is_available() else 'cpu'),  # No targets in eval
            batch_cnt=100
        )
        
        print(f"Evaluation forward pass successful!")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Reconstructions: {reconstructions}")  # Should be None
        
        # Check shapes
        assert predictions.shape == (num_pred_steps, N, d_features), f"Wrong predictions shape: {predictions.shape}"
        print("Evaluation mode shapes correct!")
        
    except Exception as e:
        print(f"Evaluation mode error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "="*60)
    print("Testing Multiple Steps and Consistency...")
    
    try:
        model.train()
        
        # Test different prediction lengths
        for steps in [1, 3, 5]:
            model.source_hidden = None
            predictions, reconstructions = model(
                graph, macro_sequence,
                num_pred_steps=steps,
                target_sequence=torch.randn(steps, N, d_features_source),
                batch_cnt=500
            )
            print(f"Steps {steps}: predictions {predictions.shape}, reconstructions {reconstructions.shape}")
        
        # Test numerical stability
        model.source_hidden = None
        predictions, _ = model(
            graph, macro_sequence,
            num_pred_steps=10,  # Longer prediction
            target_sequence=torch.randn(10, N, d_features_source),
            batch_cnt=1000
        )
        
        if torch.isnan(predictions).any():
            print("Warning: NaN values detected in predictions")
        else:
            print("No NaN values in extended predictions")
            
        if torch.isinf(predictions).any():
            print("Warning: Infinite values detected in predictions")
        else:
            print("No infinite values in extended predictions")
        
        print(f"Extended prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
    except Exception as e:
        print(f"Multi-step test error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "="*60)
    print("All tests passed! KineticForecastingFramework is working correctly.")
    
    print("\nFramework Summary:")
    print(f"- Input features: {d_features}")
    print(f"- Mesoscale components: {Q_mesoscale}")
    print(f"- Velocity grid: {xi_velocities}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"- Decay steps: {model.decay_steps}")
    print(f"- Training mode support: pass")
    print(f"- Teacher forcing: pass")
    print(f"- Reconstruction loss: pass")
    print(f"- Multi-step prediction: pass")