import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
import torch
from pidpm.diffusion import DiffusionModel
from pidpm.models import TrajectoryModel
from pidpm import utils

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic trajectories using a trained Pi-DPM model.")
    parser.add_argument('--model', type=str, default='results/model.pth', help='Path to trained model weights')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of trajectories to generate')
    parser.add_argument('--out', type=str, default='results/generated_trajectories.csv', help='Output CSV file for generated trajectories')
    args = parser.parse_args()

    # Load model weights (requires knowing the model architecture input_dim)
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    # Infer input_dim from saved weights (assuming TrajectoryModel architecture)
    input_dim = state_dict.get('fc3.bias').shape[0] if 'fc3.bias' in state_dict else None
    if input_dim is None:
        raise ValueError("Could not infer model input dimension from weights. Provide the correct model architecture.")
    model = TrajectoryModel(input_dim=input_dim, hidden_dim=128)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = DiffusionModel(model, num_timesteps=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    diffusion.device = device

    # Generate samples
    samples = diffusion.sample(num_samples=args.num_samples)
    samples = samples.to(torch.device('cpu'))
    # Save generated trajectories
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    utils.save_trajectories(samples, args.out)
    print(f"Generated {args.num_samples} trajectories saved to {args.out}")

if __name__ == "__main__":
    main()