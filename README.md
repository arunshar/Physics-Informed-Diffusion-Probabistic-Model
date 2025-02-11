# Physics-Informed Diffusion Probabistic Model
 Implementation of KDD 2025 Research Track Submission: "Physics-Informed Diffusion Probabilistic Model (Pi-DPM) for Anomaly Detection in Trajectories: A Summary of Results

## Overview
Pi-DPM is a physics-informed generative model designed for anomaly detection in trajectory data. It leverages the Denoising Diffusion Probabilistic Model (DDPM) framework to learn the normal patterns of object trajectories, then detects anomalies as deviations from these patterns. Unlike standard diffusion models, Pi-DPM integrates physics-based constraints into the generation process, ensuring that synthetic trajectories remain physically plausible and adhere to known motion laws. This combination of data-driven learning and physics-informed priors improves the alignment of generated trajectories with real-world dynamics and provides a natural regularization against overfitting.

## Key Features
- Physics-Informed Generation: Incorporates physical constraints (e.g., kinematic limits or conservation laws) into the diffusion model to produce realistic trajectory reconstructions and samples.

- Unsupervised Anomaly Detection: Learns the distribution of normal trajectories from unlabeled data and flags trajectories with high reconstruction error as anomalies, eliminating the need for labeled anomalies during training.

- Robust Diffusion Framework: Builds on the DDPM paradigm for stable training and high-fidelity trajectory generation, avoiding issues like mode collapse common in GANs/VAEs.

- Synthetic Trajectory Synthesis: Can generate new plausible trajectories that follow learned dynamics, useful for data augmentation or simulation scenarios.

## Requirements

The required packages with python environment is:

      python>=3.7
      torch>=1.7
      pandas
      numpy
      matplotlib
      pathlib
      shutil
      datetime
      colored
      math

## Installation Instructions

1. Clone the repository: 

Download the Pi-DPM repository to your local machine, either via git or as a ZIP download.

      git clone https://github.com/arunshar/Physics-Informed-Diffusion-Probabistic-Model.git
      cd Pi-DPM

2. Installation Dependencies: 

Pi-DPM is built with Python (>=3.8) and PyTorch. Install the required packages using pip (it is recommended to use a virtual environment):

      pip install -r requirements.txt

## Running Instructions

Once the environment is set up, you can use the provided scripts to train the model, detect anomalies on new trajectories, and generate synthetic trajectories. Below are the typical usage steps:

### Training the Model

Once the environment is set up, you can use the provided scripts to train the model, detect anomalies on new trajectories, and generate synthetic trajectories. Below are the typical usage steps:

      python scripts/train.py --data data/train_dataset.csv --epochs 100 --batch_size 64 --out checkpoints/							

In this example:

--data specifies the path to the training dataset of normal trajectories.

--epochs and --batch_size control the training iterations and batch size.

--out is the directory to save model checkpoints and training logs.


### Detecting Anomalies

Train Pi-DPM on your trajectory dataset by running the training script. This will learn the diffusion model on normal trajectory patterns:

		python scripts/detect.py --model checkpoints/pidpm_model.pth --data data/test_dataset.csv --threshold 0.1 --output results/anomalies.csv						

Here:

--model points to the trained Pi-DPM model checkpoint.

--data is the file or directory containing trajectories to evaluate.

The script will reconstruct or evaluate each trajectory and compute an anomaly score (e.g., based on reconstruction error). If a --threshold is provided, the script will label trajectories with a score above this threshold as anomalies.

--output (optional) can be used to save the anomaly detection results (e.g., anomaly scores or flags for each trajectory) to a CSV file.

### Generating Synthetic Trajectories

Pi-DPM can also generate new, synthetic trajectories that mimic the normal behavior learned from the training data. This can be useful for data augmentation or simulating trajectories under the learned model:

python scripts/generate.py --model checkpoints/pidpm_model.pth --num_samples 50 --out results/synthetic_trajectories.csv

      python scripts/generate.py --model checkpoints/pidpm_model.pth --num_samples 50 --out results/synthetic_trajectories.csv


### Code Structure

      Pi-DPM/ 
      ├── pidpm/                   # Core Python package for the Pi-DPM model
      │   ├── diffusion.py         # Diffusion model implementation (forward & reverse process)
      │   ├── physics.py           # Module for physics-informed constraints and losses
      │   ├── models.py            # Neural network architectures for trajectory encoding/decoding
      │   ├── utils.py             # Helper functions (data loading, metrics, etc.)
      ├── scripts/                 # Scripts for training, inference, and generation
      ├── data/                    # Directory for datasets (not included in repo, user-provided)
      ├── results/                 # Output directory for results (model checkpoints, anomaly outputs, synthetic data)
      ├── requirements.txt         # List of required Python packages
      ├── LICENSE                  # License file (MIT License)
      └── README.md                # Documentation (this file)																																