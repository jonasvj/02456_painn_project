import math
import torch
import torch.nn as nn
from typing import Union, Tuple
from torch_geometric.nn import radius_graph


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with Reparameterization Trick.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

        # KL divergence for this layer (calculated in forward)
        self.kl_div = 0.0

    def reset_parameters(self):
        # Initialize mu like standard linear layer
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)

        # Initialize rho to give small initial variance (sigma = log(1+exp(rho)))
        # rho = -3 => sigma ~= 0.05
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1. Compute sigma = log(1 + exp(rho))
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # 2. Sample epsilon ~ N(0, 1)
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)

        # 3. Reparameterize: w = mu + sigma * epsilon
        weight = self.weight_mu + weight_sigma * weight_epsilon
        bias = self.bias_mu + bias_sigma * bias_epsilon

        # 4. Compute output
        output = nn.functional.linear(input, weight, bias)

        # 5. Compute KL Divergence (Analytical: Gaussian q(w) || Gaussian p(w)=N(0,1))
        # KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        # Note: We sum over all weights and biases in this layer
        
        # Weight KL
        kl_weight = 0.5 * torch.sum(
            weight_sigma**2 + self.weight_mu**2 - 1 - torch.log(weight_sigma**2 + 1e-8)
        )
        # Bias KL
        kl_bias = 0.5 * torch.sum(
            bias_sigma**2 + self.bias_mu**2 - 1 - torch.log(bias_sigma**2 + 1e-8)
        )
        
        self.kl_div = kl_weight + kl_bias

        return output


def build_readout_network(
    num_in_features: int,
    num_out_features: int = 1,
    num_layers: int = 2,
    activation: nn.Module = nn.SiLU
):
    """
    Build readout network using Bayesian Linear layers.

    Args:
        num_in_features: Number of input features.
        num_out_features: Number of output features (targets).
        num_layers: Number of layers in the network.
        activation: Activation function as a nn.Module.
    
    Returns:
        The readout network as a nn.Module.
    """
    # Number of neurons in each layer
    num_neurons = [
        num_in_features,
        *[
            max(num_out_features, num_in_features // 2**(i + 1))
            for i in range(num_layers-1)
        ],
        num_out_features,
    ]

    # Build network
    readout_network = nn.Sequential()
    for i, (n_in, n_out) in enumerate(zip(num_neurons[:-1], num_neurons[1:])):
        readout_network.append(BayesianLinear(n_in, n_out))
        if i < num_layers - 1:
            readout_network.append(activation())

    return readout_network


class SinusoidalRBFLayer(nn.Module):
    """
    Sinusoidal Radial Basis Function.
    """
    def __init__(self, num_basis: int = 20, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            num_basis: Number of radial basis functions to use.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_dist = cutoff_dist     

        self.register_buffer(
            'freqs',
            math.pi * torch.arange(1, self.num_basis + 1) / self.cutoff_dist
        )


    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal radial basis functions for a tensor of distances.

        Args:
            distances: torch.Tensor of distances (any size).
        
        Returns:
            A torch.Tensor of radial basis functions with size [*, num_basis]
                where * is the size of the input (the distances).
        """
        distances = distances.unsqueeze(-1)
        return torch.sin(self.freqs * distances) / distances


class CosineCutoff(nn.Module):
    """
    Cosine cutoff function.
    """
    def __init__(self, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.cutoff_dist = cutoff_dist


    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Applies cosine cutoff function to input.

        Args:
            distances: torch.Tensor of distances (any size).
        
        Returns:
            torch.Tensor of distances that has been cut with the cosine cutoff
            function.
        """
        return torch.where(
            distances < self.cutoff_dist,
            0.5 * (torch.cos(distances * math.pi / self.cutoff_dist) + 1),
            0
        )

class PaiNNMessageBlock(nn.Module):
    """
    Message block in PaiNN.
    """
    def __init__(
        self,
        num_features: int = 128,
        num_rbf_features: int = 20
    ) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_rbf_features: Number of radial basis functions to represent
                distances.
        """
        super().__init__()
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features

        self.scalar_network = nn.Sequential(
            nn.Linear(
                in_features=self.num_features,
                out_features=self.num_features
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self.num_features,
                out_features=3*self.num_features
            )
        )
        self.rbf_network = nn.Linear(
            in_features=self.num_rbf_features,
            out_features=3*self.num_features
        )


    def forward(
        self,
        idx_i: Union[torch.IntTensor, torch.LongTensor],
        idx_j: Union[torch.IntTensor, torch.LongTensor],
        rel_dir: torch.Tensor,
        rel_dist_cut: torch.Tensor,
        rbf_features: torch.Tensor,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PaiNN message block.
        
        Args:
            idx_i: torch.Tensor of size [num_edges] with node indexes.
            idx_j: torch.Tensor of size [num_edges] with neighbor indexes.
            rel_dir: torch.Tensor of size [num_edges, 3] with directions
                between nodes.
            rel_dist_cut: torch.Tensor of size [num_edges] with cosine cutted
                distances between nodes.
            rbf_features: torch.Tensor of size [num_edges, num_rbf_features].
            scalar_features: torch.Tensor of size [num_nodes, num_features] 
                with scalar features of each node.
            vector_features: torch.Tensor of size [num_nodes, num_features, 3]
                with vector features of each node.

        Returns:
            A tuple with scalar features and vector features, i.e., tensors
            with sizes [num_nodes, num_features] and
            [num_nodes, num_features, 3], respectively.
        """
        # Propagate scalar features
        phi = self.scalar_network(scalar_features)                                  # [num_nodes, 3*num_features]

        # Propagate rbf features 
        W = self.rbf_network(rbf_features) * rel_dist_cut.unsqueeze(-1)             # [num_edges, 3*num_features]

        # Multiply phi and W and split
        phi_W = phi[idx_j] * W                                                      # [num_edges, 3*num_features]
        phi_W_vv, phi_W_ss, phi_W_vs = torch.split(                                 # [num_edges, num_features]
            phi_W, self.num_features, dim=-1
        )

        # Compute scalar residuals
        scalar_residuals = torch.zeros_like(scalar_features)                        # [num_nodes, num_features]
        scalar_residuals.index_add_(dim=0, index=idx_i, source=phi_W_ss)            # [num_nodes, num_features]

        # Compute vector residuals
        vector_residuals = torch.zeros_like(vector_features)                        # [num_nodes, num_features, 3]
        vector_residuals_per_edge = (                                               # [num_edges, num_features, 3]
            vector_features[idx_j] * phi_W_vv.unsqueeze(-1)
            + phi_W_vs.unsqueeze(-1) * rel_dir.unsqueeze(-2)
        )
        vector_residuals.index_add_(                                                # [num_nodes, num_features, 3]
            dim=0,
            index=idx_i,
            source=vector_residuals_per_edge
        )

        scalar_features = scalar_features + scalar_residuals                        # [num_nodes, num_features]
        vector_features = vector_features + vector_residuals                        # [num_nodes, num_features, 3]

        return scalar_features, vector_features


class PaiNNUpdateBlock(nn.Module):
    """
    Update block in PaiNN.
    """
    def __init__(self, num_features: int = 128) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
        """
        super().__init__()
        self.num_features = num_features

        self.U = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_features,
            bias=False
        )
        self.V = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_features,
            bias=False
        )
        self.scalar_vector_network = nn.Sequential(
            nn.Linear(
                in_features=2*self.num_features,
                out_features=self.num_features
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self.num_features,
                out_features=3*self.num_features
            )
        )


    def forward(
        self,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PaiNN update block.

        Args:
            scalar_features: torch.Tensor of size [num_nodes, num_features] 
                with scalar features of each node.
            vector_features: torch.Tensor of size [num_nodes, num_features, 3]
                with vector features of each node.
        
        Returns:
            A tuple with scalar features and vector features, i.e., tensors
            with sizes [num_nodes, num_features] and
            [num_nodes, num_features, 3], respectively.
        """
        U_vector_features = self.U(vector_features.movedim(-2, -1)).movedim(-2, -1) # [num_nodes, num_features, 3]
        V_vector_features = self.V(vector_features.movedim(-2, -1)).movedim(-2, -1) # [num_nodes, num_features, 3]

        a = self.scalar_vector_network(                                             # [num_nodes, 3*num_features]
            torch.cat([
                torch.linalg.vector_norm(V_vector_features, dim=-1),
                scalar_features
            ], dim=-1)
        )
        a_vv, a_sv, a_ss = torch.split(a, self.num_features, dim=-1)                # [num_nodes, num_features]

        vector_residuals = U_vector_features * a_vv.unsqueeze(-1)                   # [num_nodes, num_features, 3]
        scalar_residuals = (
            a_ss + a_sv * torch.sum(U_vector_features * V_vector_features, dim=-1)
        )
        scalar_features = scalar_features + scalar_residuals                        # [num_nodes, num_features]
        vector_features = vector_features + vector_residuals                        # [num_nodes, num_features, 3]

        return scalar_features, vector_features


class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist

        self.atom_embedding = nn.Embedding(
            num_embeddings=self.num_unique_atoms + 1,
            embedding_dim=num_features,
            padding_idx=0
        )
        self.cosine_cut = CosineCutoff(
            cutoff_dist=self.cutoff_dist
        )
        self.radial_basis = SinusoidalRBFLayer(
            num_basis=self.num_rbf_features,
            cutoff_dist=self.cutoff_dist
        )
        self.message_blocks = nn.ModuleList()
        self.update_blocks = nn.ModuleList()
        for _ in range(self.num_message_passing_layers):
            self.message_blocks.append(
                PaiNNMessageBlock(
                    num_features=self.num_features,
                    num_rbf_features=self.num_rbf_features,
                )
            )
            self.update_blocks.append(
                PaiNNUpdateBlock(num_features=self.num_features)
            )
        
        self.readout_network = build_readout_network(
            num_in_features=self.num_features,
            num_out_features=self.num_outputs,
            num_layers=2,
            activation=nn.SiLU
        )


    @property
    def kl_divergence(self):
        """
        Computes the KL divergence of the model (sum of KL of all Bayesian layers).
        """
        kl = 0.0
        for layer in self.readout_network:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_div
        return kl


    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        scalar_features = self.atom_embedding(atoms)                                # [num_nodes, num_features]
        vector_features = torch.zeros(                                              # [num_nodes, num_features, 3]
            scalar_features.size() + (3,), 
            dtype=scalar_features.dtype,
            device=scalar_features.device,
        )

        _, num_nodes_per_graph = torch.unique(graph_indexes, return_counts=True)
        idx_i, idx_j = radius_graph(
            x=atom_positions,
            r=self.cutoff_dist,
            batch=graph_indexes,
            loop=False,
            max_num_neighbors=torch.max(num_nodes_per_graph),
            flow='target_to_source',
            batch_size=len(num_nodes_per_graph),
        )
        rel_pos = atom_positions[idx_j] - atom_positions[idx_i]                     # [num_possible_edges, 3]
        rel_dist = torch.linalg.vector_norm(rel_pos, dim=1)                         # [num_possible_edges]

        # Relative directions, cosine cutted distances, and rbf features
        rel_dir = rel_pos / rel_dist.unsqueeze(-1)                                  # [num_edges, 3]
        rel_dist_cut = self.cosine_cut(rel_dist)                                    # [num_edges]
        rbf_features = self.radial_basis(rel_dist)                                  # [num_edges, num_rbf_features]

        for message, update in zip(self.message_blocks, self.update_blocks):
            scalar_features, vector_features = message(                             # ([num_nodes, num_features],
                idx_i,                                                              #  [num_nodes, num_features, 3])
                idx_j,
                rel_dir,
                rel_dist_cut,
                rbf_features,
                scalar_features,
                vector_features,
            )
            scalar_features, vector_features = update(                              # ([num_nodes, num_features],
                scalar_features,                                                    #  [num_nodes, num_features, 3])
                vector_features
            )

        atomic_contributions = self.readout_network(scalar_features)                # [num_nodes, num_outputs]

        return atomic_contributions