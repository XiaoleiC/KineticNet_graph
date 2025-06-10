import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from typing import Tuple, Optional


class VelocitySet:
    """
    Manages discrete velocity set for Boltzmann equation on graphs.
    For traffic flow, velocities are non-negative only.
    """
    
    def __init__(self, num_velocities: int = 20, max_velocity: float = 20.0, device: str = 'cpu', weights: Optional[torch.Tensor] = None):
        self.num_velocities = num_velocities
        self.max_velocity = max_velocity
        self.device = device
        
        # Generate non-negative velocity set for traffic flow
        # Using uniform spacing from 0 to max_velocity
        self.velocities = torch.linspace(0, max_velocity, num_velocities, device=device)
        
        # Fixed importance sampling weights (uniform for now)
        # Could be replaced with Gauss-Hermite quadrature weights
        if weights is not None:
            assert weights.shape[0] == num_velocities, "Weights must match number of velocities"
            self.weights = weights.to(device)
        else:
            # Uniform weights
            self.weights = torch.ones(num_velocities, device=device) / num_velocities
        
    def get_velocities(self) -> torch.Tensor:
        """Returns velocity set {ξ_k}"""
        return self.velocities
    
    def get_weights(self) -> torch.Tensor:
        """Returns importance sampling weights {w_i}"""
        return self.weights
    
    @property
    def Q(self) -> int:
        """Number of velocity components"""
        return self.num_velocities



class CollisionOperator(nn.Module):
    """
    Collision operator with spatial locality consideration.
    Implements Ω(f) with soft collision invariance constraints.
    """
    
    def __init__(self, velocity_set: VelocitySet, hidden_dim: int = 64, 
                 constraint_weight: float = 1.0):
        super(CollisionOperator, self).__init__()
        self.velocity_set = velocity_set
        self.Q = velocity_set.Q
        self.constraint_weight = constraint_weight
        
        # MLP for collision operator considering spatial locality
        # Input: [f(x_i, ξ_1), ..., f(x_i, ξ_Q), neighbor_info]
        self.collision_net = nn.Sequential(
            nn.Linear(self.Q + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.Q)
        )
        
        # Network to encode spatial neighborhood information
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.Q, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Collision invariance matrix C
        self.register_buffer('collision_matrix', self._build_collision_matrix())
        
    def _build_collision_matrix(self) -> torch.Tensor:
        """
        Build collision invariance matrix C for mass, momentum conservation.
        C = [[w_1, w_2, ..., w_Q],
             [w_1*ξ_1, w_2*ξ_2, ..., w_Q*ξ_Q]]
        """
        velocities = self.velocity_set.get_velocities()
        weights = self.velocity_set.get_weights()
        
        # Mass conservation row
        mass_row = weights.unsqueeze(0)  # [1, Q]
        
        # Momentum conservation row  
        momentum_row = (weights * velocities).unsqueeze(0)  # [1, Q]
        
        # Combine into collision matrix [2, Q]
        C = torch.cat([mass_row, momentum_row], dim=0)
        return C
    
    def forward(self, f_node: torch.Tensor, neighbor_f: torch.Tensor, apply_constraint = False) -> torch.Tensor:
        """
        Forward pass of collision operator.
        
        Args:
            f_node: Distribution at current node [batch_size, Q]
            neighbor_f: Average distribution from neighbors [batch_size, Q]
            
        Returns:
            Collision term Ω(f) [batch_size, Q]
        """
        batch_size = f_node.size(0)
        
        # Encode spatial neighborhood information
        spatial_info = self.spatial_encoder(neighbor_f)  # [batch_size, hidden_dim]
        
        # Concatenate node distribution and spatial info
        collision_input = torch.cat([f_node, spatial_info], dim=-1)  # [batch_size, Q + hidden_dim]
        
        # Compute raw collision term
        omega_raw = self.collision_net(collision_input)  # [batch_size, Q]
        if apply_constraint:
            # Apply soft collision invariance constraints
            omega_constrained = self._apply_soft_constraints(omega_raw)
        else:
            omega_constrained = omega_raw

        return omega_constrained
    
    def _apply_soft_constraints(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Apply soft collision invariance constraints using penalty method.
        """
        batch_size = omega.size(0)
        
        # Compute constraint violations
        # constraints = C @ omega^T  # [2, batch_size]
        constraints = torch.matmul(self.collision_matrix, omega.transpose(0, 1))  # [2, batch_size]
        
        # Penalty term: minimize ||C @ omega||^2
        constraint_loss = torch.sum(constraints ** 2, dim=0)  # [batch_size]
        
        # Apply penalty as a correction term (soft constraint)
        # This is a simple heuristic; more sophisticated methods could be used
        penalty_correction = self.constraint_weight * constraint_loss.unsqueeze(-1)  # [batch_size, 1]
        
        # Reduce collision term magnitude when constraints are violated
        omega_constrained = omega * torch.exp(-penalty_correction)
        
        return omega_constrained
    
    def _apply_hard_constraints(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Apply hard collision invariance constraints using projection method.
        Ω* = (I - C^T(CC^T)^{-1}C) Ω
        """
        batch_size = omega.size(0)
        C = self.collision_matrix  # [2, Q]
        
        # Compute projection matrix P = I - C^T(CC^T)^{-1}C
        # notes: CC^T is [2, 2], C^T(CC^T)^{-1}C is [Q, Q]
        CCT = torch.matmul(C, C.transpose(0, 1))  # [2, 2]
        CCT_inv = torch.inverse(CCT + 1e-8 * torch.eye(2, device=C.device))  # for numerical stability
        
        projection_matrix = torch.eye(self.Q, device=C.device) - \
                        torch.matmul(C.transpose(0, 1), torch.matmul(CCT_inv, C))  # [Q, Q]
        
        # Apply projection: Ω* = P @ Ω^T, then transpose back
        omega_constrained = torch.matmul(omega, projection_matrix.transpose(0, 1))  # [batch_size, Q]
        
        return omega_constrained


class FlowWeightCalculator(nn.Module):
    """
    Calculates flow weights q_{ij} based on graph structure and current state.
    """
    
    def __init__(self, feature_dim: int = 64):
        super(FlowWeightCalculator, self).__init__()
        self.feature_dim = feature_dim
        
        # Network to compute flow weights from edge features
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, feature_dim),  # +1 for distance
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # Ensure weights are in [0, 1]
        )
    
    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor) -> torch.Tensor:
        """
        Compute flow weights q_{ij} for all edges.
        
        Args:
            graph: DGL graph with edge weights as distances
            node_features: Node features [N, feature_dim]
            
        Returns:
            Flow weights [num_edges]
        """
        src_nodes, dst_nodes = graph.edges()
        
        # Get source and destination node features
        src_features = node_features[src_nodes]  # [num_edges, feature_dim]
        dst_features = node_features[dst_nodes]  # [num_edges, feature_dim]
        
        # Get edge distances (weights)
        edge_distances = graph.edata['weight'].unsqueeze(-1)  # [num_edges, 1]
        
        # Concatenate features
        edge_input = torch.cat([src_features, dst_features, edge_distances], dim=-1)
        
        # Compute raw weights
        raw_weights = self.weight_net(edge_input).squeeze(-1)  # [num_edges]
        
        # Normalize weights for each source node (sum to 1)
        graph.edata['raw_weight'] = raw_weights
        graph.update_all(fn.copy_e('raw_weight', 'm'), fn.sum('m', 'weight_sum'))
        
        # Avoid division by zero
        weight_sums = graph.ndata['weight_sum'][src_nodes]
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        
        normalized_weights = raw_weights / weight_sums
        
        return normalized_weights


class BoltzmannUpdater(nn.Module):
    """
    Implements temporal updating using Boltzmann equation on graphs.
    """
    
    def __init__(self, velocity_set: VelocitySet, collision_op: CollisionOperator,
                 flow_calculator: FlowWeightCalculator):
        super(BoltzmannUpdater, self).__init__()
        self.velocity_set = velocity_set
        self.collision_op = collision_op
        self.flow_calculator = flow_calculator
        self.Q = velocity_set.Q
        
    def forward(self, graph: dgl.DGLGraph, f: torch.Tensor, node_features: torch.Tensor,
                dt: float = 0.1) -> torch.Tensor:
        """
        Update distribution function using Boltzmann equation.
        
        Args:
            graph: DGL graph with edge weights as distances d_{ij}
            f: Distribution function [N, Q] where N is number of nodes
            node_features: Node features for flow weight calculation [N, feature_dim]
            dt: Time step
            
        Returns:
            Updated distribution function [N, Q]
        """
        N = graph.num_nodes()
        velocities = self.velocity_set.get_velocities()  # [Q]
        
        # Compute flow weights q_{ij}
        flow_weights = self.flow_calculator(graph, node_features)  # [num_edges]
        graph.edata['flow_weight'] = flow_weights
        
        # Initialize transport term
        transport_term = torch.zeros_like(f)  # [N, Q]
        
        # Compute transport term for each velocity component
        for k in range(self.Q):
            velocity_k = velocities[k]
            f_k = f[:, k]  # [N]
            
            # Set node data
            graph.ndata['f_k'] = f_k
            
            # Compute neighbor contribution: sum over neighbors
            def message_func(edges):
                # Transport: q_{ij} * ξ_k * (f_j - f_i) / d_{ij}
                src_f = edges.src['f_k']
                dst_f = edges.dst['f_k']
                flow_weight = edges.data['flow_weight']
                distance = edges.data['weight']
                
                # Note: in directed graph, flow is from src to dst
                transport = flow_weight * velocity_k * (dst_f - src_f) / torch.clamp(distance, min=1e-8)
                return {'transport': transport}
            
            def reduce_func(nodes):
                return {'transport_sum': torch.sum(nodes.mailbox['transport'], dim=1)}
            
            graph.update_all(message_func, reduce_func)
            transport_term[:, k] = graph.ndata['transport_sum']
        
        # Compute collision term (considering spatial locality)
        # Average neighbor distributions for spatial encoding
        graph.ndata['f_full'] = f
        graph.update_all(fn.copy_u('f_full', 'm'), fn.mean('m', 'neighbor_f'))
        neighbor_f = graph.ndata['neighbor_f']  # [N, Q]
        
        collision_term = self.collision_op(f, neighbor_f)  # [N, Q]
        
        # Boltzmann equation update: f^{n+1} = f^n - dt * [transport - collision]
        f_new = f - dt * (transport_term - collision_term)
        
        # Ensure non-negativity (physical constraint)
        f_new = torch.clamp(f_new, min=0.0)
        
        return f_new


class MomentDecoder(nn.Module):
    """
    Explicit decoder that recovers macroscale variables using moment calculations.
    """
    
    def __init__(self, velocity_set: VelocitySet, variance_weight: float = 0.1):
        super(MomentDecoder, self).__init__()
        self.velocity_set = velocity_set
        self.variance_weight = variance_weight
        
    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute macroscale variables from mesoscale distribution.
        
        Args:
            f: Distribution function [N, Q]
            
        Returns:
            density: ρ [N]
            velocity: u [N] 
            variance_loss: For weight optimization [1]
        """
        velocities = self.velocity_set.get_velocities()  # [Q]
        weights = self.velocity_set.get_weights()  # [Q]
        
        # Density: ρ = Σ w_i * f_i
        density = torch.sum(weights * f, dim=-1)  # [N]
        
        # Velocity: u = Σ w_i * ξ_i * f_i / ρ
        weighted_momentum = torch.sum(weights * velocities * f, dim=-1)  # [N]
        velocity = weighted_momentum / torch.clamp(density, min=1e-8)  # [N]
        
        # Variance loss for importance sampling weight optimization
        # This is a placeholder - could implement more sophisticated variance estimation
        f_normalized = f / torch.clamp(density.unsqueeze(-1), min=1e-8)  # [N, Q]
        variance_loss = torch.var(f_normalized) * self.variance_weight
        
        return density, velocity, variance_loss


# Example usage and testing
if __name__ == "__main__":
    # Test the components
    device = 'cpu'
    
    # Create velocity set
    velocity_set = VelocitySet(num_velocities=20, max_velocity=2.0, device=device)
    
    # Create components
    collision_op = CollisionOperator(velocity_set, hidden_dim=64)
    flow_calculator = FlowWeightCalculator(feature_dim=64)
    updater = BoltzmannUpdater(velocity_set, collision_op, flow_calculator)
    decoder = MomentDecoder(velocity_set)
    
    # Create a simple test graph
    N = 10  # number of nodes
    edges = [(i, (i+1) % N) for i in range(N)]  # ring graph
    src, dst = zip(*edges)
    
    g = dgl.graph((src, dst))
    g.edata['weight'] = torch.ones(g.num_edges())  # unit distances
    
    # Test data
    f = torch.rand(N, velocity_set.Q) + 0.1  # [N, Q]
    node_features = torch.rand(N, 64)  # [N, 64]
    
    # Test forward pass
    f_new = updater(g, f, node_features, dt=0.1)
    density, velocity, var_loss = decoder(f_new)
    
    print(f"Input shape: {f.shape}")
    print(f"Output shape: {f_new.shape}")
    print(f"Density shape: {density.shape}")
    print(f"Velocity shape: {velocity.shape}")
    print(f"Variance loss: {var_loss.item()}")
    print("Boltzmann components test passed!")