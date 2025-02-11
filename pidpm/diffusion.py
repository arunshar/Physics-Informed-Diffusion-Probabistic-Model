import torch
import numpy as np

class DiffusionModel:
    """Diffusion process for trajectories with forward (noising) and reverse (denoising) steps."""
    def __init__(self, model, num_timesteps=100, beta_start=1e-4, beta_end=0.02, device=None):
        """
        Initialize the diffusion model.
        Args:
            model: Neural network model (e.g. from pidpm.models) used to predict noise.
            num_timesteps: Number of diffusion steps (T).
            beta_start: Starting value of beta for variance schedule.
            beta_end: Final value of beta for variance schedule.
            device: Torch device to use (CPU or CUDA). If None, use model's device or CPU.
        """
        self.model = model
        self.num_timesteps = num_timesteps
        # Create linear beta schedule from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        # Precompute alphas and their cumulative products
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        # If device specified (or model has one), move tensors to device
        if device is None:
            device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)

    def q_sample(self, x0, t):
        """
        Diffuse the data (trajectory) `x0` to time step `t` by adding Gaussian noise.
        Args:
            x0: Original data (trajectory) at time 0, as a torch Tensor.
            t: Diffusion step (int or Tensor of shape (batch,) indicating step for each sample).
        Returns:
            x_t: Noised trajectory at time step t.
        """
        # Ensure t is a tensor on correct device
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long, device=self.device)
        t = t.to(self.device)
        # If x0 is multi-dimensional (e.g. batch of trajectories), ensure t has matching shape
        if t.dim() == 0:
            t = t.unsqueeze(0)  # shape (1,)
        # Get cumulative alpha values for each t
        alpha_cum = self.alpha_cumprod[t]  # shape (len(t),)
        # Make sure alpha_cum is shaped for broadcasting with x0
        while alpha_cum.dim() < x0.dim():
            alpha_cum = alpha_cum.unsqueeze(-1)
        # Sample normal noise
        noise = torch.randn_like(x0)
        # Compute x_t = sqrt(alpha_cum) * x0 + sqrt(1 - alpha_cum) * noise
        return torch.sqrt(alpha_cum) * x0 + torch.sqrt(1 - alpha_cum) * noise

    def predict_epsilon(self, x_t, t):
        """
        Use the neural network model to predict the noise added to get x_t (i.e., epsilon).
        Args:
            x_t: Noisy data at time step t.
            t: Diffusion step (int or Tensor) corresponding to x_t.
        Returns:
            Predicted noise (epsilon) for x_t.
        """
        return self.model(x_t, t)

    def p_sample_step(self, x_t, t):
        """
        Perform one reverse diffusion step: predict x_{t-1} from x_t.
        Args:
            x_t: Current noised data at step t.
            t: Current timestep (int).
        Returns:
            x_{t-1}: Predicted data at previous step (t-1).
        """
        # Predict noise using the model
        eps_pred = self.predict_epsilon(x_t, torch.tensor([t], device=self.device))
        eps_pred = eps_pred.to(self.device)
        # Obtain parameters for reverse distribution
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cum_t = self.alpha_cumprod[t]
        if t > 0:
            alpha_cum_prev = self.alpha_cumprod[t-1]
        else:
            alpha_cum_prev = torch.tensor(1.0, device=self.device)  # alpha_cumprod[-1] = 1 for t = 0
        # Estimate x0 from x_t and predicted noise epsilon
        x0_pred = (x_t - torch.sqrt(1 - alpha_cum_t) * eps_pred) / torch.sqrt(alpha_cum_t)
        # Compute mean of p(x_{t-1} | x_t)
        mean = torch.sqrt(alpha_cum_prev) * x0_pred + torch.sqrt(1 - alpha_cum_prev) * eps_pred
        if t > 0:
            # Add random noise for sampling during intermediate steps
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * noise
        else:
            # At t=0, return mean (final output)
            return mean

    def sample(self, num_samples=1, trajectory_shape=None):
        """
        Generate new trajectories by simulating the reverse diffusion process.
        Args:
            num_samples: Number of trajectories to generate.
            trajectory_shape: Shape of each trajectory (if None, infer from model or training data).
        Returns:
            A torch Tensor containing generated trajectories with shape (num_samples, *trajectory_shape).
        """
        # Determine shape of each data sample
        if trajectory_shape is None:
            # Try to infer from model attributes
            if hasattr(self.model, 'input_dim'):
                # Model expects flattened trajectory of length input_dim
                sample_shape = (num_samples, self.model.input_dim)
            elif hasattr(self.model, 'data_shape'):
                # If model has data_shape attribute (e.g., (time, features))
                sample_shape = (num_samples,) + tuple(self.model.data_shape)
            else:
                raise ValueError("trajectory_shape must be provided if not inferable from model.")
        else:
            sample_shape = (num_samples,) + tuple(trajectory_shape)
        # Start from x_T ~ N(0, I)
        x_t = torch.randn(sample_shape, device=self.device)
        # Iteratively sample from p(x_{t-1} | x_t)
        for t in range(self.num_timesteps - 1, -1, -1):
            x_t = self.p_sample_step(x_t, t)
        # After loop, x_t is x_0 (generated trajectory)
        return x_t

    def compute_loss(self, x0, physics_func=None, physics_weight=0.1):
        """
        Compute the training loss for a batch of trajectories `x0`.
        Includes diffusion model loss (prediction error) and optional physics-informed loss.
        Args:
            x0: Original trajectories batch (shape: [batch, ...]).
            physics_func: Optional function to compute physics loss given a trajectory.
            physics_weight: Weight factor for the physics loss term.
        Returns:
            A torch scalar tensor representing the total loss.
        """
        # Ensure input is on correct device
        x0 = x0.to(self.device)
        batch_size = x0.shape[0] if x0.dim() > 0 else 1
        # Sample random timesteps for each trajectory in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        # Sample noise to add
        noise = torch.randn_like(x0)
        # Get corresponding alpha_cumprod for each t
        alpha_cum = self.alpha_cumprod[t]
        while alpha_cum.dim() < x0.dim():
            alpha_cum = alpha_cum.unsqueeze(-1)
        # Diffuse x0 to x_t
        x_t = torch.sqrt(alpha_cum) * x0 + torch.sqrt(1 - alpha_cum) * noise
        # Predict noise at x_t
        pred_noise = self.model(x_t, t)
        # Diffusion loss: MSE between the predicted noise and the true noise
        diff_loss = torch.mean((pred_noise - noise) ** 2)
        # Physics-informed loss (if provided)
        if physics_func is not None:
            # Predict x0 from x_t and pred_noise
            x0_pred = (x_t - torch.sqrt(1 - alpha_cum) * pred_noise) / torch.sqrt(alpha_cum)
            phys_loss = physics_func(x0_pred)
            # If physics_func returns per-sample loss, average it
            if hasattr(phys_loss, "dim") and phys_loss.dim() > 0:
                phys_loss = torch.mean(phys_loss)
            total_loss = diff_loss + physics_weight * phys_loss
        else:
            total_loss = diff_loss
        return total_loss