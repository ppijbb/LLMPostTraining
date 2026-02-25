"""
Continuous Thought Machine (CTM) implementation from SakanaAI
https://github.com/SakanaAI/continuous-thought-machines

Paper: https://arxiv.org/abs/2505.05522
Blog: https://sakana.ai/ctm/
Interactive Website: https://pub.sakana.ai/ctm/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.models as models

# ==================== CONSTANTS ====================

VALID_NEURON_SELECT_TYPES = ['first-last', 'random', 'random-pairing']

VALID_BACKBONE_TYPES = [
    f'resnet{depth}-{i}' for depth in [18, 34, 50, 101, 152] for i in range(1, 5)
] + ['shallow-wide', 'parity_backbone']

VALID_POSITIONAL_EMBEDDING_TYPES = [
    'learnable-fourier', 'multi-learnable-fourier',
    'custom-rotational', 'custom-rotational-1d'
]

# ==================== UTILITY FUNCTIONS ====================

def compute_normalized_entropy(logits, reduction='mean'):
    """
    Calculates the normalized entropy of a PyTorch tensor of logits along the 
    final dimension.
    """
    # Apply softmax to get probabilities
    preds = F.softmax(logits, dim=-1)
    # Calculate the log probabilities
    log_preds = torch.log_softmax(logits, dim=-1)
    # Calculate the entropy
    entropy = -torch.sum(preds * log_preds, dim=-1)
    # Calculate the maximum possible entropy
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    # Normalize the entropy
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)
    return normalized_entropy

def add_coord_dim(x, scaled=True):
    """
    Adds a final dimension to the tensor representing 2D coordinates.
    """
    B, H, W = x.shape
    # Create coordinate grids
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)  # Shape (H, W)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)  # Shape (H, W)
    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)
    # Stack coordinates and expand dimensions
    coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape (H, W, 2)
    coords = coords.unsqueeze(0)  # Shape (1, 1, H, W, 2)
    coords = coords.repeat(B, 1, 1, 1)  # Shape (B, D, H, W, 2)
    return coords

# ==================== BASIC MODULES ====================

class Identity(nn.Module):
    """Identity Module - returns input unchanged."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Squeeze(nn.Module):
    """Squeeze Module - removes a specified dimension of size 1."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

# ==================== CTM CORE MODULES ====================

class SuperLinear(nn.Module):
    """
    SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM.
    
    This layer applies N independent linear transformations to corresponding
    slices of the input tensor along the neuron dimension.
    """
    def __init__(self, in_dims, out_dims, N, T=1.0, do_norm=False, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.in_dims = in_dims
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm

        # Initialize weights and biases
        # w1 shape: (memory_length, out_dims, d_model)
        self.register_parameter('w1', nn.Parameter(
            torch.zeros(in_dims, out_dims, N).uniform_(-math.sqrt(1/(in_dims + out_dims)), math.sqrt(1/(in_dims + out_dims)))
        ))
        # b1 shape: (1, N, out_dims)
        self.register_parameter('b1', nn.Parameter(
            torch.zeros(1, N, out_dims).uniform_(-math.sqrt(1/(in_dims + out_dims)), math.sqrt(1/(in_dims + out_dims)))
        ))
        
        # Learnable temperature/scaling factor
        self.register_parameter('T', nn.Parameter(torch.tensor(T), requires_grad=True))

    def forward(self, x):
        # x shape: (batch_size, n_neurons, history_length)
        x = self.dropout(x)
        x = self.layernorm(x)
        
        # Apply N independent linear transformations
        # torch.einsum('bni,iog->bno', x, self.w1) performs parallel matrix multiplications
        out = torch.einsum('bni,iog->bno', x, self.w1) + self.b1
        
        # Apply learnable temperature scaling
        out = out * self.T
        
        # Squeeze last dimension if output is 1D
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
            
        return out

class SynapseUNET(nn.Module):
    """
    UNET-style architecture for the Synapse Model.
    
    This module implements the connections between neurons in the CTM's latent
    space using a U-Net structure with skip connections.
    """
    def __init__(self, out_dims, depth, minimum_width=16, dropout=0.0):
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth

        # Define UNET structure based on depth
        widths = np.linspace(out_dims, minimum_width, depth)

        # Initial projection layer
        self.first_projection = nn.Sequential(
            nn.LazyLinear(int(widths[0])),
            nn.LayerNorm(int(widths[0])),
            nn.SiLU()
        )

        # Downward and upward paths
        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()
        num_blocks = len(widths) - 1

        for i in range(num_blocks):
            # Down block: widths[i] -> widths[i+1]
            self.down_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i]), int(widths[i+1])),
                nn.LayerNorm(int(widths[i+1])),
                nn.SiLU()
            ))
            # Up block: widths[i+1] -> widths[i]
            self.up_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i+1]), int(widths[i])),
                nn.LayerNorm(int(widths[i])),
                nn.SiLU()
            ))
            # Skip connection LayerNorm
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        # Initial projection
        out_first = self.first_projection(x)

        # Downward path, storing outputs for skip connections
        outs_down = [out_first]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))

        # Upward path, starting from the bottleneck output
        outs_up = outs_down[-1]
        num_blocks = len(self.up_projections)

        for i in range(num_blocks):
            up_layer_idx = num_blocks - 1 - i
            out_up = self.up_projections[up_layer_idx](outs_up)
            
            # Add skip connection
            skip_idx = up_layer_idx
            skip_connection = outs_down[skip_idx]
            outs_up = self.skip_lns[skip_idx](out_up + skip_connection)

        return outs_up

# ==================== BACKBONE MODULES ====================

class ParityBackbone(nn.Module):
    """Simple embedding backbone for parity tasks."""
    def __init__(self, n_embeddings, d_embedding):
        super().__init__()
        self.embedding = nn.Embedding(n_embeddings, d_embedding)

    def forward(self, x):
        return self.embedding(x)

class ShallowWide(nn.Module):
    """Shallow wide convolutional backbone."""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LazyConv2d(512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(2048, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)

# ==================== POSITIONAL EMBEDDINGS ====================

class LearnableFourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding.
    From: https://arxiv.org/pdf/2106.02795.pdf
    """
    def __init__(self, d_model, G=1, M=2, F_dim=256, H_dim=128, gamma=1/2.5):
        super().__init__()
        self.d_model = d_model
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.gamma = gamma
        
        # Learnable Fourier features
        self.B = nn.Parameter(torch.randn(M, F_dim // 2) * gamma)
        
        # MLP to map Fourier features to output dimension
        self.mlp = nn.Sequential(
            nn.Linear(F_dim, H_dim),
            nn.ReLU(),
            nn.Linear(H_dim, d_model)
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Create coordinate grid
        coords = add_coord_dim(torch.zeros(B, H, W, device=x.device), scaled=True)  # (B, H, W, 2)
        coords = coords.reshape(B, -1, 2)  # (B, H*W, 2)
        
        # Apply Fourier features
        proj = 2 * np.pi * coords @ self.B.T  # (B, H*W, F_dim//2)
        fourier_features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, H*W, F_dim)
        
        # Apply MLP
        pos_emb = self.mlp(fourier_features)  # (B, H*W, d_model)
        pos_emb = pos_emb.reshape(B, H, W, self.d_model).permute(0, 3, 1, 2)  # (B, d_model, H, W)
        
        return pos_emb

class MultiLearnableFourierPositionalEncoding(nn.Module):
    """Multiple Learnable Fourier Positional Encodings with different scales."""
    def __init__(self, d_model, G=1, M=2, F_dim=256, H_dim=128, gamma_range=[1.0, 0.1], N=10):
        super().__init__()
        self.N = N
        
        # Create N encodings with different gamma values
        gammas = np.linspace(gamma_range[0], gamma_range[1], N)
        self.encodings = nn.ModuleList([
            LearnableFourierPositionalEncoding(d_model // N, G, M, F_dim, H_dim, gamma)
            for gamma in gammas
        ])

    def forward(self, x):
        # Apply each encoding and concatenate
        pos_embs = [enc(x) for enc in self.encodings]
        return torch.cat(pos_embs, dim=1)

class CustomRotationalEmbedding(nn.Module):
    """Simple sinusoidal rotational embedding for 2D coordinates."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create coordinate grids
        y_pos = torch.arange(H, device=x.device).float().unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=x.device).float().unsqueeze(0).repeat(H, 1)
        
        # Normalize coordinates
        y_pos = y_pos / (H - 1) * 2 - 1
        x_pos = x_pos / (W - 1) * 2 - 1
        
        # Create positional embeddings
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        pos_emb = torch.zeros(H, W, self.d_model, device=x.device)
        pos_emb[:, :, 0::4] = torch.sin(x_pos.unsqueeze(-1) * div_term[::2])
        pos_emb[:, :, 1::4] = torch.cos(x_pos.unsqueeze(-1) * div_term[::2])
        pos_emb[:, :, 2::4] = torch.sin(y_pos.unsqueeze(-1) * div_term[::2])
        pos_emb[:, :, 3::4] = torch.cos(y_pos.unsqueeze(-1) * div_term[::2])
        
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos_emb

class CustomRotationalEmbedding1D(nn.Module):
    """1D rotational embedding."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # Simple 1D positional encoding
        seq_len = x.shape[-1]
        pos = torch.arange(seq_len, device=x.device).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        pos_emb = torch.zeros(seq_len, self.d_model, device=x.device)
        pos_emb[:, 0::2] = torch.sin(pos.unsqueeze(-1) * div_term)
        pos_emb[:, 1::2] = torch.cos(pos.unsqueeze(-1) * div_term)
        
        return pos_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)

# ==================== RESNET BACKBONE ====================

def prepare_resnet_backbone(backbone_type):
    """Prepare ResNet backbone based on type specification."""
    # Parse backbone type (e.g., 'resnet18-2')
    parts = backbone_type.split('-')
    resnet_name = parts[0]  # e.g., 'resnet18'
    scale = int(parts[1])   # e.g., 2
    
    # Get the appropriate ResNet model
    if resnet_name == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif resnet_name == 'resnet34':
        model = models.resnet34(pretrained=False)
    elif resnet_name == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif resnet_name == 'resnet101':
        model = models.resnet101(pretrained=False)
    elif resnet_name == 'resnet152':
        model = models.resnet152(pretrained=False)
    else:
        raise ValueError(f"Unsupported ResNet type: {resnet_name}")
    
    # Create feature extractor based on scale
    layers = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1]
    if scale >= 2:
        layers.append(model.layer2)
    if scale >= 3:
        layers.append(model.layer3)
    if scale >= 4:
        layers.append(model.layer4)
    
    return nn.Sequential(*layers)

# ==================== MAIN CTM MODEL ====================

class ContinuousThoughtMachine(nn.Module):
    """
    Continuous Thought Machine (CTM).

    Technical report: https://arxiv.org/abs/2505.05522
    Interactive Website: https://pub.sakana.ai/ctm/
    Blog: https://sakana.ai/ctm/

    Thought takes time and reasoning is a process. 
    
    The CTM consists of three main ideas:
    1. The use of internal recurrence, enabling a dimension over which a concept analogous to thought can occur. 
    2. Neuron-level models, that compute post-activations by applying private (i.e., on a per-neuron basis) MLP 
       models to a history of incoming pre-activations.
    3. Synchronisation as representation, where the neural activity over time is tracked and used to compute how 
       pairs of neurons synchronise with one another over time. This measure of synchronisation is the representation 
       with which the CTM takes action and makes predictions.
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 ):
        super(ContinuousThoughtMachine, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.heads = heads
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None
        
        # --- Core CTM Modules ---
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm)

        #  --- Start States ---
        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model)))))
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length)))))

        # --- Synchronisation ---
        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)
        
        for synch_type, size in (('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)):
            print(f"Synch representation size {synch_type}: {size}")
        if self.synch_representation_size_action:  # if not zero
            self.set_synchronisation_parameters('action', self.n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters('out', self.n_synch_out, n_random_pairing_self)

        # --- Output Processing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """
        Computes synchronisation to be used as a vector representation.
        """
        if synch_type == 'action':
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out':
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        
        if self.neuron_select_type in ('first-last', 'random'):
            if self.neuron_select_type == 'first-last':
                if synch_type == 'action':
                    selected_left = selected_right = activated_state[:, -n_synch:]
                elif synch_type == 'out':
                    selected_left = selected_right = activated_state[:, :n_synch]
            else:
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]
            
            # Compute outer product of selected neurons
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            # Take upper triangle
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]
            
        elif self.neuron_select_type == 'random-pairing':
            # For random-pairing, compute sync between specific pairs
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError("Invalid neuron selection type")
        
        # Compute synchronisation recurrently
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        """Compute the key-value features from the input data using the backbone."""
        # Handle different input types based on backbone
        if self.backbone_type == 'none':
            # For simple feature vectors, no backbone processing needed
            # x shape: (B, input_dim) -> we need (B, seq_len, d_input)
            if len(x.shape) == 2:
                # Convert 2D input to sequence format
                kv = x.unsqueeze(1)  # (B, 1, input_dim)
                if self.kv_proj is not None:
                    kv = self.kv_proj(kv)
                return kv
            else:
                # If already in sequence format, just project
                if self.kv_proj is not None:
                    return self.kv_proj(x)
                return x
        else:
            # For image inputs with backbones
            initial_rgb = self.initial_rgb(x)
            self.kv_features = self.backbone(initial_rgb)
            pos_emb = self.positional_embedding(self.kv_features)
            
            # Check if we have spatial dimensions to flatten
            if len(self.kv_features.shape) > 2:
                combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
            else:
                # Handle case where backbone output is already flattened
                combined_features = (self.kv_features + pos_emb)
                if len(combined_features.shape) == 2:
                    combined_features = combined_features.unsqueeze(1)
            
            if self.kv_proj is not None:
                kv = self.kv_proj(combined_features)
            else:
                kv = combined_features
            return kv

    def compute_certainty(self, current_prediction):
        """Compute the certainty of the current prediction."""
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """Setup initial RGB conversion layer."""
        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1)
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        """Get the dimensionality of the backbone output."""
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type: 
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else: raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else: raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        """Set the backbone module based on the specified type."""
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """Get the positional embedding module."""
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0  # Default no-op
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
        """Get neuron level models (NLMs)."""
        if deep_nlms:
            return nn.Sequential(
                SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                            do_norm=do_layernorm_nlm, dropout=dropout),
                nn.GLU(),
                SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                            do_norm=do_layernorm_nlm, dropout=dropout),
                nn.GLU(),
                Squeeze(-1)
            )
        else:
            return nn.Sequential(
                SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                            do_norm=do_layernorm_nlm, dropout=dropout),
                nn.GLU(),
                Squeeze(-1)
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        """Get the synapse model."""
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
        """Set synchronisation parameters for neuron selection."""
        assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
        left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
        synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
        self.register_buffer(f'{synch_type}_neuron_indices_left', left)
        self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        """Initialize the left and right neuron indices based on the neuron selection type."""
        if self.neuron_select_type=='first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

        device = self.start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def get_neuron_select_type(self):
        """Get neuron selection types for output and action."""
        print(f"Using neuron select type: {self.neuron_select_type}")
        if self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return neuron_select_type_out, neuron_select_type_action

    def verify_args(self):
        """Verify the validity of the input arguments."""
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"
        
        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        
        if self.neuron_select_type == 'first-last':
            assert self.d_model >= (self.n_synch_out + self.n_synch_action), \
                "d_model must be >= n_synch_out + n_synch_action for neuron subsets"

        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):
        """Calculate the size of the synchronisation representation."""
        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size

    def forward(self, x, track=False):
        """Forward pass of the CTM."""
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)  # Shape: (B, H)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        # --- Initialise Recurrent Synch Values  ---
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out


# ==================== USAGE EXAMPLE ====================

def create_simple_ctm(out_dims=10, iterations=5):
    """
    Create a simple CTM for demonstration purposes.
    Uses lazy modules which require forward pass for initialization.
    This version works with simple feature vectors (not images).
    """
    return ContinuousThoughtMachine(
        iterations=iterations,
        d_model=256,
        d_input=128,  # This should match input feature dimension
        heads=8,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=2,
        memory_length=4,
        deep_nlms=True,
        memory_hidden_dims=64,
        do_layernorm_nlm=False,
        backbone_type='none',  # No backbone for simple feature vectors
        positional_embedding_type='none',  # No positional embedding needed
        out_dims=out_dims,
        prediction_reshaper=[out_dims],
        dropout=0.1,
        neuron_select_type='random-pairing',
        n_random_pairing_self=4
    )

# ==================== USAGE EXAMPLE (NON-LAZY VERSION) ====================

def create_simple_ctm_non_lazy(out_dims=10, iterations=5, input_dim=128):
    """
    Create a simple CTM for demonstration purposes without lazy modules.
    This version specifies exact dimensions to avoid lazy initialization issues.
    """
    d_model = 256
    d_input = 128
    n_synch_out = 32
    n_synch_action = 32
    neuron_select_type = 'random-pairing'
    
    # Calculate synchronisation representation sizes
    if neuron_select_type == 'random-pairing':
        synch_size_action = n_synch_action
        synch_size_out = n_synch_out
    elif neuron_select_type in ('first-last', 'random'):
        synch_size_action = (n_synch_action * (n_synch_action + 1)) // 2
        synch_size_out = (n_synch_out * (n_synch_out + 1)) // 2
    
    class NonLazyCTM(ContinuousThoughtMachine):
        def __init__(self, *args, **kwargs):
            # Override some lazy components with fixed-size ones
            super().__init__(*args, **kwargs)
            
            # Replace lazy projections with correct dimensions
            if self.heads and self.heads > 0:
                self.kv_proj = nn.Sequential(
                    nn.Linear(d_input, d_input), 
                    nn.LayerNorm(d_input)
                )
                # q_proj should take synchronisation_action size as input
                self.q_proj = nn.Linear(synch_size_action, d_input)
            
            # Replace lazy linear in output projector with correct dimension
            self.output_projector = nn.Sequential(nn.Linear(synch_size_out, out_dims))
            
            # Replace lazy synapses if depth is 1
            if kwargs.get('synapse_depth', 1) == 1:
                expected_input_size = d_input + d_model  # attention output + activated state
                self.synapses = nn.Sequential(
                    nn.Dropout(kwargs.get('dropout', 0)),
                    nn.Linear(expected_input_size, d_model * 2),
                    nn.GLU(),
                    nn.LayerNorm(d_model)
                )
    
    return NonLazyCTM(
        iterations=iterations,
        d_model=d_model,
        d_input=d_input,
        heads=8,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        synapse_depth=1,  # Use simple synapse to avoid LazyLinear in UNet
        memory_length=4,
        deep_nlms=True,
        memory_hidden_dims=64,
        do_layernorm_nlm=False,
        backbone_type='none',  # No backbone to avoid LazyConv2d
        positional_embedding_type='none',
        out_dims=out_dims,
        prediction_reshaper=[out_dims],
        dropout=0.1,
        neuron_select_type=neuron_select_type,
        n_random_pairing_self=4
    )

if __name__ == "__main__":
    # Example usage
    model = create_simple_ctm_non_lazy()
    
    # Create dummy input
    batch_size = 4
    input_dim = 128
    dummy_input = torch.randn(batch_size, input_dim)
    
    # Forward pass
    predictions, certainties, _ = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Certainties shape: {certainties.shape}")
    print("CTM model created successfully!")
