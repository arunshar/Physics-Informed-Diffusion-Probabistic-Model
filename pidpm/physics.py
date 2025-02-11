import torch

def compute_physics_loss(trajectory):
    """
    Compute a physics-informed loss for a given trajectory or batch of trajectories.
    This example uses a smoothness constraint (minimizing jerk: third derivative of position).
    Args:
        trajectory: Trajectory tensor. Shape can be (time, dim), (batch, time, dim), or (batch, length) for 1D trajectories.
    Returns:
        A torch scalar representing the physics loss (higher if trajectory violates physical smoothness).
    """
    # Ensure trajectory is a tensor on correct device
    traj = trajectory
    if not torch.is_tensor(traj):
        traj = torch.tensor(traj, dtype=torch.float32)
    # If a single trajectory is provided without batch dimension, add batch dim
    if traj.dim() == 2:
        # Shape (time, dim) -> add batch dimension
        traj = traj.unsqueeze(0)
    elif traj.dim() == 1:
        # Shape (length,) -> single 1D trajectory, add batch and feature dims
        traj = traj.unsqueeze(0).unsqueeze(-1)
    # Now traj shape is (batch, time, dim) or (batch, time, 1)
    # Compute first, second, and third differences along time axis
    diff1 = traj[:, 1:, :] - traj[:, :-1, :]
    diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
    diff3 = diff2[:, 1:, :] - diff2[:, :-1, :]
    # Physics loss: mean squared third difference (jerk)
    loss = torch.mean(diff3 ** 2)
    return loss