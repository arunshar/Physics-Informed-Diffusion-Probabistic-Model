import torch
import numpy as np

def load_dataset(file_path):
    """
    Load trajectory dataset from a CSV file.
    The dataset is expected to be structured as one trajectory per row, with trajectory points flattened.
    If the number of columns suggests 2D or 3D trajectories, the data is reshaped accordingly (not normalized).
    Args:
        file_path: Path to CSV file containing trajectories.
    Returns:
        data: Torch tensor of shape (num_samples, sequence_length * dim) or (num_samples, sequence_length, dim).
    """
    # Load using numpy (assuming numeric data and comma delimiter)
    data = np.loadtxt(file_path, delimiter=',')
    # If the data is one-dimensional (single trajectory), reshape for consistency
    if data.ndim == 1:
        data = data.reshape(1, -1)
    N, M = data.shape
    # Heuristic: if M is divisible by 2 or 3, assume that as spatial dimensions
    traj_dim = None
    if M % 3 == 0:
        traj_dim = 3
    elif M % 2 == 0:
        traj_dim = 2
    if traj_dim is not None and traj_dim != 1:
        T = M // traj_dim
        if T * traj_dim == M:
            data = data.reshape(N, T, traj_dim)
    # Convert to torch tensor
    tensor_data = torch.tensor(data, dtype=torch.float32)
    return tensor_data

def save_trajectories(trajs, file_path):
    """
    Save trajectories to a CSV file.
    Each trajectory will be flattened into one row.
    Args:
        trajs: Torch tensor or NumPy array of trajectories (shape [N, ...]).
        file_path: Path to save the CSV file.
    """
    # Move to CPU and convert to numpy for saving
    if torch.is_tensor(trajs):
        arr = trajs.detach().cpu().numpy()
    else:
        arr = np.array(trajs)
    # If trajectories are in shape (N, T, D), flatten to (N, T*D)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    np.savetxt(file_path, arr, delimiter=',')

def save_model(model, file_path):
    """
    Save the model's state dictionary to a file.
    Args:
        model: Trained model (nn.Module).
        file_path: Path to save the model (e.g., 'results/model.pth').
    """
    torch.save(model.state_dict(), file_path)

def load_model(model_class, file_path, **kwargs):
    """
    Load model state from file and return the model instance.
    Args:
        model_class: The class of the model to instantiate.
        file_path: Path to the saved state dict file.
        **kwargs: Any additional args needed to initialize the model.
    Returns:
        model: Model instance with loaded weights.
    """
    model = model_class(**kwargs)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model