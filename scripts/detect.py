import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
import torch
from pidpm.diffusion import DiffusionModel
from pidpm.models import TrajectoryModel
from pidpm import utils

def main():
    parser = argparse.ArgumentParser(description="Compute anomaly scores for trajectories using a trained Pi-DPM model.")
    parser.add_argument('--data', type=str, default='data/sample_dataset.csv', help='Path to input trajectories CSV')
    parser.add_argument('--model', type=str, default='results/model.pth', help='Path to trained model weights (.pth file)')
    parser.add_argument('--out', type=str, default='results/anomaly_scores.csv', help='Output CSV file for anomaly scores')
    args = parser.parse_args()

    # Load data and model
    data = utils.load_dataset(args.data)
    num_samples = data.shape[0]
    input_dim = data.shape[-1] if data.ndim == 2 else data.shape[1] * data.shape[2]
    # Flatten data if needed
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    # Initialize model and load weights
    model = TrajectoryModel(input_dim=input_dim, hidden_dim=128)
    model = utils.load_model(TrajectoryModel, args.model, input_dim=input_dim, hidden_dim=128)
    diffusion = DiffusionModel(model, num_timesteps=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    diffusion.device = device

    # Compute anomaly scores for each trajectory
    scores = []
    # We'll use a fixed mid diffusion step (t = T/2) for reconstruction
    t_eval = diffusion.num_timesteps // 2
    alpha_cum = diffusion.alpha_cumprod[t_eval]
    # Expand alpha_cum for broadcasting to data shape
    while alpha_cum.dim() < data.dim():
        alpha_cum = alpha_cum.unsqueeze(-1)
    for i in range(num_samples):
        x0 = data[i:i+1].to(device)  # shape (1, input_dim)
        # Forward diffuse to step t_eval
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_cum) * x0 + torch.sqrt(1 - alpha_cum) * noise
        # Predict noise and reconstruct x0
        pred_noise = model(x_t, torch.tensor([t_eval], device=device))
        x0_recon = (x_t - torch.sqrt(1 - alpha_cum) * pred_noise) / torch.sqrt(alpha_cum)
        # Compute MSE between original and reconstructed trajectory
        mse = torch.mean((x0_recon - x0) ** 2).item()
        scores.append(mse)

    # Save scores to CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write("trajectory_id,anomaly_score\n")
        for i, score in enumerate(scores):
            f.write(f"{i},{score:.6f}\n")
    print(f"Anomaly scores saved to {args.out}")

if __name__ == "__main__":
    main()