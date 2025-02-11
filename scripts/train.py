import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
import torch
from pidpm.diffusion import DiffusionModel
from pidpm.models import TrajectoryModel
from pidpm.physics import compute_physics_loss
from pidpm import utils

def main():
    parser = argparse.ArgumentParser(description="Train the Pi-DPM model on a trajectory dataset.")
    parser.add_argument('--data', type=str, default='data/sample_dataset.csv', help='Path to training data CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--out', type=str, default='results/model.pth', help='Output path for the trained model')
    args = parser.parse_args()

    # Load dataset
    data = utils.load_dataset(args.data)
    num_samples = data.shape[0]
    input_dim = data.shape[-1] if data.ndim == 2 else data.shape[1] * data.shape[2]
    print(f"Loaded dataset from {args.data} with {num_samples} trajectories.")
    # If data is 3D (N, T, D), reshape to 2D (N, T*D) for the MLP model
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)

    # Initialize model and diffusion
    model = TrajectoryModel(input_dim=input_dim, hidden_dim=128)
    diffusion = DiffusionModel(model, num_timesteps=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    diffusion.device = device

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create DataLoader for batch training
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            x0 = batch[0].to(device)
            loss = diffusion.compute_loss(x0, physics_func=compute_physics_loss, physics_weight=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x0.size(0)
        epoch_loss /= num_samples
        print(f"Epoch {epoch}/{args.epochs}, Loss: {epoch_loss:.6f}")

    # Save trained model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    utils.save_model(model, args.out)
    print(f"Trained model saved to {args.out}")

if __name__ == "__main__":
    main()