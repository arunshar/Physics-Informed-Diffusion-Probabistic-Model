import pandas as pd
import numpy as np

# Define parameters for the synthetic dataset
num_samples = 100  # Number of trajectories
seq_length = 20  # Number of time steps per trajectory
trajectory_dim = 2  # 2D trajectories (x, y)

# Generate synthetic trajectories with smooth movement
np.random.seed(42)
trajectories = []
for _ in range(num_samples):
    x = np.cumsum(np.random.normal(0, 1, seq_length))  # Random walk for x
    y = np.cumsum(np.random.normal(0, 1, seq_length))  # Random walk for y
    trajectory = np.column_stack((x, y)).flatten()  # Flatten (x, y) pairs into a single row
    trajectories.append(trajectory)

# Convert to DataFrame
columns = [f"x{t//2}" if t % 2 == 0 else f"y{t//2}" for t in range(seq_length * trajectory_dim)]
df = pd.DataFrame(trajectories, columns=columns)

# Save as CSV
df.to_csv("sample_dataset.csv", index=False)

print("Sample dataset saved as 'sample_dataset.csv'")