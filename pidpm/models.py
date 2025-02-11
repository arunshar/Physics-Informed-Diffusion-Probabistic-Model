import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryModel(nn.Module):
    """
    Neural network model for trajectory data.
    This model predicts the added noise (epsilon) given a noisy trajectory and time step.
    """
    def __init__(self, input_dim, hidden_dim=128):
        """
        Initialize the trajectory model.
        Args:
            input_dim: Dimensionality of the input trajectory (flattened length).
            hidden_dim: Hidden layer size.
        """
        super(TrajectoryModel, self).__init__()
        self.input_dim = input_dim
        # Simple time embedding: linear layer to embed time step into hidden_dim
        self.time_embed = nn.Linear(1, hidden_dim)
        # Neural network layers to process trajectory data
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

        # (Optional) If using sequence models, you could add an RNN/Transformer here

    def forward(self, x, t):
        """
        Forward pass of the model.
        Args:
            x: Noisy trajectory input (tensor of shape [batch, input_dim]).
            t: Diffusion step (can be int, or tensor of shape [batch] or [batch,1]).
        Returns:
            Predicted noise (tensor of shape [batch, input_dim]) for input x at time t.
        """
        # Ensure t is a tensor of shape [batch, 1]
        if isinstance(t, int):
            # If single int, create a batch tensor filled with t
            batch_size = x.shape[0] if x.ndim > 0 else 1
            t = torch.tensor([t] * batch_size, dtype=torch.float32, device=x.device)
        t = t.to(x.device, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(1)  # shape [batch, 1]
        # Normalize time step to [0,1] (assuming t ranges 0 to T-1, and input_dim roughly correlates with T for normalization)
        t_norm = t / max(1.0, float(self.input_dim))
        # Compute time embedding
        time_feat = F.relu(self.time_embed(t_norm))
        # Process trajectory data through fully connected layers
        h = F.relu(self.fc1(x))
        # Add time embedding into hidden representation
        if time_feat.shape[1] != h.shape[1]:
            # If time_feat dimension does not match hidden dimension, repeat or truncate accordingly
            time_feat = time_feat.repeat(1, h.shape[1] // time_feat.shape[1])
        h = h + time_feat
        h = F.relu(self.fc2(h))
        # Output layer (predict noise same shape as input)
        out = self.fc3(h)
        return out

    # Additional architectures (e.g., RNN-based) could be added here.