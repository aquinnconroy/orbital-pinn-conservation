# Conservation-Augmented Physics-Informed Neural Networks for N-Body Dynamics

Code and data for: *Predicting orbital trajectories by enforcing conservation laws in physics-informed neural networks* (Conroy & Farber, 2026).

## Results Summary

Across 92 hyperparameter configurations (28 baseline, 64 conservation-augmented), enforcing energy, momentum, and angular momentum conservation penalties as soft constraints in a feed-forward neural network:

- Reduced mean validation MSE by 35.6%
- Reduced MSE standard deviation by over 150x (from 943 to 6)
- Improved energy drift by ~10^6x, momentum drift by ~859x, angular momentum drift by ~1.2x10^5x
- Achieved 23x better trajectory accuracy at 500-step rollouts

## Repository Structure

```
├── train.py                  # Model training script (W&B sweep compatible)
├── generate_figures.py       # Generates all 8 paper figures
├── data/
│   ├── paper_runs_final.json         # 92-run experiment results
│   ├── rollout_length_comparison.csv # Multi-horizon rollout data (top-5 per group)
│   └── appendix_drift_data.json      # Leapfrog integrator drift simulation
├── configs/
│   ├── sweep_baseline.yaml               # Baseline (no conservation penalties)
│   ├── sweep_conservation.yaml           # All three conservation penalties
│   ├── sweep_ablation_energy_only.yaml   # Energy penalty only
│   ├── sweep_ablation_momentum_only.yaml # Momentum penalty only
│   └── sweep_ablation_angular_only.yaml  # Angular momentum penalty only
├── checkpoints/
│   └── model_*.pt            # Top-5 baseline + top-5 conservation checkpoints (Table 2 models)
└── figures/                  # Generated figures (created by generate_figures.py)
```

## Generating Figures

To reproduce all figures from the paper:

```bash
pip install -r requirements.txt
python generate_figures.py
```

Figures are saved to `figures/`. The script reads from `data/` and requires no GPU or network access.

## Training

To train models using Weights & Biases sweeps:

```bash
pip install -r requirements.txt
wandb sweep configs/sweep_conservation.yaml
wandb agent <sweep_id>
```

Training requires PyTorch with CUDA support. See `train.py` docstring for architecture details and hyperparameter options. Multiple sweeps were run with varying sub-ranges; the config files represent the overall search space described in the paper.

## System

- 6-body gravitational system in 3D
- Plummer-softened Newtonian gravity (G=1, epsilon=0.001)
- Leapfrog (Stormer-Verlet) integrator for data generation
- Feed-forward MLP predicting next-step velocities
- FP64 training for numerical stability of conservation quantities

## Citation

If you use this code, please cite:

```
Conroy, A. Q., & Farber, R. (2026). Predicting orbital trajectories by enforcing
conservation laws in physics-informed neural networks. Journal of Emerging Investigators.
```
