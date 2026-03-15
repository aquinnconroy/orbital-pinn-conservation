"""
N-Body Physics-Informed Neural Network (PINN) v3.6 - Training Script for W&B Sweeps

This script trains a neural network to predict N-body gravitational dynamics while
enforcing physical conservation laws (energy, momentum, angular momentum) as soft
constraints in the loss function.

FIX (v3.6a): Validation now uses UNWEIGHTED losses by default. Previously, temporal
weighting was applied during validation, which masked poor performance on later
timesteps. Now val/mse_raw and val/mse_weighted will be the same (both unweighted)
during validation, giving a true picture of model performance across all timesteps.
Temporal weighting still applies during training to help the model learn short-term
predictions first.

Architecture: Direct MLP with raw pos + vel + mass input (42 features for 6 bodies, 3D).

v3.6 Key Changes (Comprehensive Conservation Study):
    1. Removed Fourier Features: Simplified model uses raw inputs only.
    2. Fixed 3D Angular Momentum Bug: Explicit shape handling for mass multiplication.
    3. Per-Batch Conservation Losses: All conservation losses return (B,) shape for
       consistent temporal weighting with MSE.
    4. UNNORMALIZED Integrated Drift: All three conservation losses use the same
       unnormalized integrated drift formulation for fair comparison:
       - Energy: (E[t] - E[0])^2
       - Momentum: ||P[t] - P[0]||^2
       - Angular: ||L[t] - L[0]||^2 (or (L[t] - L[0])^2 for 2D)
       This allows studying relative importance of each conservation law.
    5. Configurable Drift Modes: Each conservation loss can use integrated drift
       (from initial state) or per-step drift (consecutive differences).
       - use_integrated_energy_drift: true=E[t]-E[0], false=E[t+1]-E[t]
       - use_integrated_momentum_drift: true=P[t]-P[0], false=P[t+1]-P[t]
       - use_integrated_angular_drift: true=L[t]-L[0], false=L[t+1]-L[t]
    6. FP64 Training Mode: Optional use_fp64_training flag for numerical stability
       (disabled by default, ~2-3x slower but more accurate gradients).
    7. Early Stopping: Tracks best model by validation MSE, saves checkpoint_best.pt,
       stops if no improvement for early_stopping_patience epochs.
    8. Enhanced Wandb Logging: Drift mode configs, early stopping events, norm updates.

v3.5 Features (retained):
    - Adaptive Temporal Weighting: w[t] = exp(-alpha * sum(L[0:t]))
    - FP64 validation physics computations
    - Dynamic gradient balancing via periodic norm updates
    - Configurable hidden layer architecture
    - Multiple LR scheduler options
    - Center-of-Mass Frame for Angular Momentum

Changelog:
    v3.6: Removed Fourier features, UNNORMALIZED integrated drift for all conservation laws,
           drift modes, early stopping, per-batch losses
    v3.5: Adaptive temporal weighting, integrated energy drift, CoM angular momentum

Usage:
    wandb sweep sweep_config_v3.6.yaml

Author: Quinn Conroy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    ExponentialLR,
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import os

# Enable TF32 for faster training on Ampere+ GPUs (RTX 30xx, 40xx, 50xx)
# Note: TF32 trades some precision for speed - disabled for physics computations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable cuDNN benchmark for auto-tuning convolution algorithms
torch.backends.cudnn.benchmark = True


# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def physics_acc(positions: torch.Tensor, masses: torch.Tensor, G: float, softening: float) -> torch.Tensor:
    """
    Compute gravitational acceleration for all bodies using direct N-body summation.

    Uses Plummer softening: a_i = G * sum_j(m_j * (r_j - r_i) / (|r_j - r_i|^2 + ε^2)^(3/2))

    Args:
        positions: Body positions, shape (B, N, D)
        masses: Body masses, shape (B, N, 1)
        G: Gravitational constant
        softening: Plummer softening length ε to prevent singularities

    Returns:
        accelerations: Shape (B, N, D)
    """
    B, N, D = positions.shape

    # Compute pairwise displacement vectors: r_ij = r_j - r_i
    r_i = positions.unsqueeze(2)  # (B, N, 1, D)
    r_j = positions.unsqueeze(1)  # (B, 1, N, D)
    rij = r_j - r_i               # (B, N, N, D)

    # Compute distances |r_ij|
    dist = torch.norm(rij, dim=-1, keepdim=True)  # (B, N, N, 1)

    # Compute 1/(|r_ij|^2 + ε^2)^(3/2) with Plummer softening
    inv_dist3 = 1.0 / (dist ** 2 + softening ** 2) ** 1.5  # (B, N, N, 1)

    # Compute contribution from each body j
    m_j = masses.unsqueeze(1)  # (B, 1, N, 1)
    contrib = G * m_j * rij * inv_dist3  # (B, N, N, D)

    # Zero out self-interaction
    mask = 1.0 - torch.eye(N, device=positions.device).view(1, N, N, 1)
    contrib = contrib * mask  # (B, N, N, D)

    return contrib.sum(dim=2)  # (B, N, D)


def leapfrog_step(pos, vel, mass, dt, G, softening):
    """
    Symplectic leapfrog (Verlet) integrator step.

    Args:
        pos: Current positions, shape (B, N, D)
        vel: Current velocities, shape (B, N, D)
        mass: Body masses, shape (B, N, 1)
        dt: Time step
        G: Gravitational constant
        softening: Softening parameter

    Returns:
        pos_next, vel_next, a_next: All shape (B, N, D)
    """
    a_t = physics_acc(pos, mass, G, softening)
    vel_half = vel + 0.5 * dt * a_t
    pos_next = pos + dt * vel_half
    a_next = physics_acc(pos_next, mass, G, softening)
    vel_next = vel_half + 0.5 * dt * a_next
    return pos_next, vel_next, a_next


def model_leapfrog_step(pos, vel, mass, dt, model):
    """
    Leapfrog-style integration using model-predicted velocities.

    The model predicts the full-step velocity vel[t+1], but for leapfrog
    consistency, we need to update position using a half-step velocity.
    We approximate: vel_half ≈ (vel[t] + vel[t+1]) / 2

    This ensures position updates match the leapfrog scheme used in ground truth,
    eliminating integration mismatch in the MSE loss.

    Args:
        pos: Current positions, shape (B, N, D)
        vel: Current velocities, shape (B, N, D)
        mass: Body masses, shape (B, N, 1)
        dt: Time step
        model: Neural network that predicts next velocity

    Returns:
        pos_next: Next positions (B, N, D)
        vel_next: Next velocities from model (B, N, D)
    """
    # Model predicts the full-step velocity at t+1
    vel_next_pred = model(pos, vel, mass)  # (B, N, D)

    # Approximate half-step velocity for position update
    # This matches leapfrog's use of vel_half for position integration
    vel_half = 0.5 * (vel + vel_next_pred)

    # Update position using half-step velocity (matches leapfrog)
    pos_next = pos + dt * vel_half

    return pos_next, vel_next_pred


# =============================================================================
# CONSERVATION QUANTITY COMPUTATIONS
# =============================================================================

def compute_kinetic_energy(vel, mass):
    """KE = 0.5 * sum_i(m_i * |v_i|^2), returns shape (B,)"""
    v_squared = (vel ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    ke_per_body = 0.5 * mass * v_squared  # (B, N, 1)
    return ke_per_body.sum(dim=(1, 2))  # (B,)


def compute_potential_energy(pos, mass, G, eps=1e-3):
    """PE = -G * sum_{i<j}(m_i * m_j / sqrt(r_ij^2 + ε^2)), Plummer softening, returns shape (B,)"""
    B, N, D = pos.shape

    r_i = pos.unsqueeze(2)  # (B, N, 1, D)
    r_j = pos.unsqueeze(1)  # (B, 1, N, D)
    rij = r_j - r_i         # (B, N, N, D)
    dist = torch.norm(rij, dim=-1)  # (B, N, N)

    # Upper triangle mask (i < j)
    mask = torch.triu(torch.ones(N, N, device=pos.device), diagonal=1)

    m_i = mass.squeeze(-1).unsqueeze(2)  # (B, N, 1)
    m_j = mass.squeeze(-1).unsqueeze(1)  # (B, 1, N)

    pe_pairs = -G * m_i * m_j / torch.sqrt(dist ** 2 + eps ** 2)  # (B, N, N)
    pe_pairs = pe_pairs * mask.unsqueeze(0)

    return pe_pairs.sum(dim=(1, 2))  # (B,)


def compute_total_energy(pos, vel, mass, G, eps=1e-3):
    """E = KE + PE, returns shape (B,)"""
    return compute_kinetic_energy(vel, mass) + compute_potential_energy(pos, mass, G, eps)


def compute_linear_momentum(vel, mass):
    """P = sum_i(m_i * v_i), returns shape (B, D)"""
    return (mass * vel).sum(dim=1)  # (B, D)


def compute_angular_momentum(pos, vel, mass):
    """
    L = sum_i(m_i * r_i x v_i) computed in center-of-mass frame.
    
    Returns (B,) for 2D, (B, 3) for 3D.
    
    v3.6 Fix: Explicit shape handling for 3D case to ensure proper broadcasting
    of mass (B, N, 1) with cross product (B, N, 3).
    """
    B, N, D = pos.shape
    
    # Compute center of mass
    total_mass = mass.sum(dim=1, keepdim=True)  # (B, 1, 1)
    pos_com = (mass * pos).sum(dim=1, keepdim=True) / total_mass  # (B, 1, D)
    vel_com = (mass * vel).sum(dim=1, keepdim=True) / total_mass  # (B, 1, D)
    
    # Positions and velocities relative to CoM
    pos_rel = pos - pos_com  # (B, N, D)
    vel_rel = vel - vel_com  # (B, N, D)

    if D == 2:
        # 2D: L = m * (x * vy - y * vx), scalar per body
        cross = pos_rel[..., 0] * vel_rel[..., 1] - pos_rel[..., 1] * vel_rel[..., 0]  # (B, N)
        L_per_body = mass.squeeze(-1) * cross  # (B, N)
        return L_per_body.sum(dim=1)  # (B,)
    else:
        # 3D: L = m * (r x v), vector per body
        # cross has shape (B, N, 3), mass has shape (B, N, 1)
        cross = torch.cross(pos_rel, vel_rel, dim=-1)  # (B, N, 3)
        # Explicit broadcast: mass (B, N, 1) * cross (B, N, 3) -> (B, N, 3)
        L_per_body = mass * cross  # (B, N, 3) - broadcasting works correctly
        return L_per_body.sum(dim=1)  # (B, 3)


# =============================================================================
# CONSERVATION LOSS FUNCTIONS (Per-Batch, Shape (B,))
# =============================================================================

def momentum_conservation_loss_per_batch(vel, mass, vel_next):
    """
    Per-batch momentum change: ||dP||^2 for each sample.
    
    Returns:
        loss: Shape (B,) - squared momentum change per batch element
    """
    P_t = compute_linear_momentum(vel, mass)  # (B, D)
    P_next = compute_linear_momentum(vel_next, mass)  # (B, D)
    dP = P_next - P_t  # (B, D)
    return (dP ** 2).sum(dim=-1)  # (B,) - ||dP||^2 per sample


def angular_momentum_conservation_loss_per_batch(pos, vel, mass, pos_next, vel_next):
    """
    Per-batch angular momentum change: ||dL||^2 for each sample.
    
    Returns:
        loss: Shape (B,) - squared angular momentum change per batch element
    """
    L_t = compute_angular_momentum(pos, vel, mass)  # (B,) for 2D, (B, 3) for 3D
    L_next = compute_angular_momentum(pos_next, vel_next, mass)
    dL = L_next - L_t
    
    # For 2D: dL is (B,) scalar
    # For 3D: dL is (B, 3) vector, need ||dL||^2
    if dL.dim() == 1:
        return dL ** 2  # (B,)
    else:
        return (dL ** 2).sum(dim=-1)  # (B,) - ||dL||^2


def compute_energy_drift_per_batch(E_next, E_ref, eps=1e-6):
    """
    Per-batch energy drift (NORMALIZED).

    v3.6b: Computes normalized squared energy difference for scale-invariance.
    All three conservation losses (energy, momentum, angular) use NORMALIZED
    drift for fair comparison across samples and consistent loss scales (~1.0).
        loss = ((E_next - E_ref) / sqrt(|E_ref|² + eps²))²

    Args:
        E_next: Energy at next timestep (usually predicted), shape (B,)
        E_ref: Reference energy (initial state), shape (B,)
        eps: Small value for numerical stability (default 1e-6)

    Returns:
        drift: Shape (B,) - normalized squared drift
    """
    dE = E_next - E_ref  # (B,)
    # Use sqrt form for smooth handling of near-zero reference values
    scale = torch.sqrt(E_ref ** 2 + eps ** 2)  # (B,)
    return (dE / scale) ** 2  # (B,)


def compute_momentum_drift_per_batch(P_next, P_ref, eps=1e-6):
    """
    Per-batch momentum drift (NORMALIZED).

    v3.6b: Computes normalized squared momentum difference for scale-invariance.
    Normalization enables fair comparison with energy and angular momentum losses.
        loss = (||P_next - P_ref|| / sqrt(||P_ref||² + eps²))²

    Args:
        P_next: Momentum at next timestep (usually predicted), shape (B, D)
        P_ref: Reference momentum (initial or previous true state), shape (B, D)
        eps: Small value for numerical stability (default 1e-6)

    Returns:
        drift: Shape (B,) - normalized squared drift
    """
    dP = P_next - P_ref  # (B, D)
    dP_mag = torch.norm(dP, dim=-1)  # (B,) - ||dP||
    P_ref_mag = torch.norm(P_ref, dim=-1)  # (B,) - ||P_ref||
    scale = torch.sqrt(P_ref_mag ** 2 + eps ** 2)  # (B,)
    return (dP_mag / scale) ** 2  # (B,)


def compute_angular_drift_per_batch(L_next, L_ref, eps=1e-6):
    """
    Per-batch angular momentum drift (NORMALIZED).

    v3.6b: Computes normalized squared angular momentum difference for scale-invariance.
    Normalization enables fair comparison with energy and momentum losses.
        loss = (||L_next - L_ref|| / sqrt(||L_ref||² + eps²))²

    Args:
        L_next: Angular momentum at next timestep (usually predicted), shape (B,) for 2D, (B, 3) for 3D
        L_ref: Reference angular momentum (initial or previous true state), same shape
        eps: Small value for numerical stability (default 1e-6)

    Returns:
        drift: Shape (B,) - normalized squared drift
    """
    dL = L_next - L_ref

    if dL.dim() == 1:
        # 2D: L is scalar
        scale = torch.sqrt(L_ref ** 2 + eps ** 2)  # (B,)
        return (dL / scale) ** 2  # (B,)
    else:
        # 3D: L is vector
        dL_mag = torch.norm(dL, dim=-1)  # (B,)
        L_ref_mag = torch.norm(L_ref, dim=-1)  # (B,)
        scale = torch.sqrt(L_ref_mag ** 2 + eps ** 2)  # (B,)
        return (dL_mag / scale) ** 2  # (B,)


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

class DirectOrbitMLP(nn.Module):
    """
    Direct MLP that predicts next velocities from current state.
    
    v3.6: Simplified architecture using raw pos + vel + mass inputs.
    No Fourier feature encoding.

    Input: Flattened [positions, velocities, masses] for all bodies
           = N*D (pos) + N*D (vel) + N (mass) = N*(2D+1) features
           For 6 bodies, 3D: 6*(2*3+1) = 42 input features
    Output: Next velocities for all bodies (N*D features)
    """

    def __init__(self, n_bodies=6, dim=3, hidden_sizes=[256, 256, 256, 128]):
        super().__init__()
        self.n_bodies = n_bodies
        self.dim = dim
        
        # Input: pos (N*D) + vel (N*D) + mass (N) = N*(2D+1)
        input_size = n_bodies * (2 * dim + 1)
        output_size = n_bodies * dim

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, pos, vel, mass):
        """
        Args:
            pos: (B, N, D), vel: (B, N, D), mass: (B, N, 1)
        Returns:
            vel_next: (B, N, D)
        """
        B = pos.shape[0]
        pos_flat = pos.view(B, -1)
        vel_flat = vel.view(B, -1)
        mass_flat = mass.view(B, -1)

        x = torch.cat([pos_flat, vel_flat, mass_flat], dim=-1)
        
        out = self.net(x)
        return out.view(B, self.n_bodies, self.dim)


# =============================================================================
# DATASET
# =============================================================================

class NBodyTrajDataset(Dataset):
    """Dataset of N-body trajectories generated with leapfrog integration."""

    def __init__(self, num_traj, steps, N, dt, dim, G, softening, device='cpu', seed=42,
                 rollout_steps=10):
        super().__init__()
        self.device = device
        self.rollout_steps = rollout_steps
        self.dt = dt
        self.num_traj = num_traj
        self.steps = steps
        self.starts_per_traj = steps - rollout_steps

        self.pos = torch.empty((num_traj, steps, N, dim), device=device)
        self.vel = torch.empty((num_traj, steps, N, dim), device=device)
        self.mass = torch.empty((num_traj, N, 1), device=device)

        g = torch.Generator(device=device).manual_seed(seed)

        for i in range(num_traj):
            pos = torch.randn((1, N, dim), generator=g, device=device) * 0.5
            vel = torch.randn((1, N, dim), generator=g, device=device) * 0.1
            mass = 0.5 + torch.rand((1, N, 1), generator=g, device=device)

            self.mass[i] = mass.squeeze(0)
            p, v = pos.clone(), vel.clone()

            for t in range(steps):
                self.pos[i, t] = p.squeeze(0)
                self.vel[i, t] = v.squeeze(0)
                p, v, _ = leapfrog_step(p, v, mass, dt, G, softening)

    def __len__(self):
        return self.num_traj * self.starts_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.starts_per_traj
        step_idx = idx % self.starts_per_traj
        end_idx = step_idx + self.rollout_steps + 1

        return {
            'pos_seq': self.pos[traj_idx, step_idx:end_idx],
            'vel_seq': self.vel[traj_idx, step_idx:end_idx],
            'mass': self.mass[traj_idx],
            'dt': self.dt
        }


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

def create_scheduler(optimizer, config):
    """
    Create a learning rate scheduler based on config.

    Args:
        optimizer: The optimizer to schedule
        config: wandb config with scheduler parameters

    Returns:
        scheduler: The LR scheduler (or None if scheduler_type is 'none')
        is_plateau: True if using ReduceLROnPlateau (needs val_loss for step)
    """
    scheduler_type = getattr(config, 'scheduler_type', 'cosine')
    epochs = config.epochs

    if scheduler_type == 'none':
        return None, False

    elif scheduler_type == 'cosine':
        # Cosine annealing from initial LR to eta_min
        eta_min = getattr(config, 'scheduler_eta_min', 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        return scheduler, False

    elif scheduler_type == 'step':
        # Reduce LR by gamma every step_size epochs
        step_size = getattr(config, 'scheduler_step_size', 10)
        gamma = getattr(config, 'scheduler_gamma', 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, False

    elif scheduler_type == 'plateau':
        # Reduce LR when validation loss plateaus
        factor = getattr(config, 'scheduler_factor', 0.5)
        patience = getattr(config, 'scheduler_patience', 5)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )
        return scheduler, True

    elif scheduler_type == 'exponential':
        # Exponential decay each epoch
        gamma = getattr(config, 'scheduler_gamma', 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        return scheduler, False

    else:
        print(f"Unknown scheduler type '{scheduler_type}', using no scheduler")
        return None, False


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def batch_to_device(batch, device, dtype=None):
    """Move all tensors in batch to specified device with non-blocking transfers.

    Args:
        batch: Dictionary with tensors and other values
        device: Target device (e.g., 'cuda', 'cpu')
        dtype: Optional dtype to cast floating-point tensors to (e.g., torch.float64)
    """
    result = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.is_floating_point() and dtype is not None:
                # Convert floating-point tensors to target dtype directly on target device
                # This avoids issues with AMP scaler expecting consistent dtypes
                result[k] = v.to(device, dtype=dtype, non_blocking=True)
            else:
                result[k] = v.to(device, non_blocking=True)
        else:
            result[k] = v
    return result


def compute_loss(batch, model, device, config, norm_constants=None, use_fp64_physics=False,
                 apply_temporal_weighting=True):
    """
    Compute total loss = MSE + weighted conservation penalties.

    v3.6 Key Changes:
    1. All conservation losses return per-batch (B,) shape for consistent temporal weighting.
    2. Configurable drift modes: integrated (from initial state) vs per-step (consecutive).
    3. Supports FP64 training mode for numerical stability.

    Args:
        batch: Dictionary with 'pos_seq', 'vel_seq', 'mass', 'dt'
        model: The neural network
        device: Compute device
        config: wandb config with weights, physics parameters, and drift mode flags
        norm_constants: Dict of normalization values for each loss (optional)
        use_fp64_physics: Use FP64 for physics computations (for validation or FP64 mode)
        apply_temporal_weighting: If False, skip temporal weighting (useful for validation to
            get true unweighted performance across all timesteps)

    Returns:
        losses: Dictionary with 'total', 'mse', and optionally 'energy', 'momentum', 'angular_momentum'
    """
    pos_seq = batch['pos_seq'].to(device)  # (B, T, N, D)
    vel_seq = batch['vel_seq'].to(device)  # (B, T, N, D)
    mass = batch['mass'].to(device)        # (B, N, 1)
    dt = batch['dt'] if isinstance(batch['dt'], float) else batch['dt'][0].item()

    B, T, N, D = pos_seq.shape
    rollout_steps = T - 1

    # Choose precision for physics computations
    physics_dtype = torch.float64 if use_fp64_physics else torch.float32
    
    # Get adaptive temporal weighting parameters
    temporal_decay_alpha = getattr(config, 'temporal_decay_alpha', 1.0)
    apply_temporal_to_conservation = getattr(config, 'apply_temporal_to_conservation', True)
    
    # Get drift mode flags (True = integrated from initial, False = per-step)
    use_integrated_energy_drift = getattr(config, 'use_integrated_energy_drift', True)
    use_integrated_momentum_drift = getattr(config, 'use_integrated_momentum_drift', True)
    use_integrated_angular_drift = getattr(config, 'use_integrated_angular_drift', True)

    # Model's rollout trajectory
    pos_rollout = pos_seq[:, 0]  # (B, N, D)
    vel_rollout = vel_seq[:, 0]  # (B, N, D)

    # Store per-step losses for adaptive weighting - all have shape (B,) per step
    mse_losses_per_step = []        # List of (B,) tensors
    energy_losses_per_step = []     # List of (B,) tensors
    momentum_losses_per_step = []   # List of (B,) tensors
    angular_losses_per_step = []    # List of (B,) tensors
    
    # For integrated drift modes - store initial quantities
    E_initial = None
    P_initial = None
    L_initial = None
    
    # Also track previous values for per-step mode
    E_prev = None
    P_prev = None
    L_prev = None

    # First pass: Compute all per-step losses
    for t in range(rollout_steps):
        # Model-based leapfrog step
        pos_rollout_next, vel_pred = model_leapfrog_step(
            pos_rollout, vel_rollout, mass, dt, model
        )

        # Ground truth references
        pos_gt_next = pos_seq[:, t + 1]     # (B, N, D)
        vel_gt_next = vel_seq[:, t + 1]     # (B, N, D)

        # MSE loss (per step, per batch) - shape (B,)
        mse_pos = nn.functional.mse_loss(pos_rollout_next, pos_gt_next, reduction='none').mean(dim=(1, 2))  # (B,)
        mse_vel = nn.functional.mse_loss(vel_pred, vel_gt_next, reduction='none').mean(dim=(1, 2))  # (B,)
        mse_step = (mse_pos + mse_vel) / 2  # (B,)
        mse_losses_per_step.append(mse_step)

        # Conservation losses computed with specified precision
        with torch.cuda.amp.autocast(enabled=False):
            pos_rollout_phys = pos_rollout.to(physics_dtype)
            vel_rollout_phys = vel_rollout.to(physics_dtype)
            pos_next_phys = pos_rollout_next.to(physics_dtype)
            vel_pred_phys = vel_pred.to(physics_dtype)
            mass_phys = mass.to(physics_dtype)
            
            # Energy loss
            if config.weight_energy > 0:
                E_current = compute_total_energy(pos_rollout_phys, vel_rollout_phys,
                                                 mass_phys, config.G, eps=config.softening)  # (B,) - True current
                E_pred_next = compute_total_energy(pos_next_phys, vel_pred_phys,
                                             mass_phys, config.G, eps=config.softening)  # (B,) - Predicted next

                if t == 0:
                    E_initial = E_current.clone()  # Store TRUE initial energy

                if use_integrated_energy_drift:
                    # Integrated: drift from initial state (E_pred[t] vs E_initial[0])
                    # Compares PREDICTED energy at each step against TRUE initial energy
                    # This measures cumulative drift: as model improves, drift decreases
                    energy_loss = compute_energy_drift_per_batch(E_pred_next, E_initial)  # (B,)
                else:
                    # Per-step: consecutive difference in TRUE states
                    # Measures how well true physics conserves energy (for reference)
                    energy_loss = compute_energy_drift_per_batch(E_pred_next, E_current)  # (B,)

                energy_losses_per_step.append(energy_loss.float())  # (B,)
                
            # Momentum loss
            if config.weight_momentum > 0:
                P_current = compute_linear_momentum(vel_rollout_phys, mass_phys)  # (B, D) - True current
                P_pred_next = compute_linear_momentum(vel_pred_phys, mass_phys)  # (B, D) - Predicted next

                if t == 0:
                    P_initial = P_current.clone()  # Store TRUE initial momentum

                if use_integrated_momentum_drift:
                    # Integrated: drift from initial state (P_pred[t] vs P_initial[0])
                    # Compares PREDICTED momentum at each step against TRUE initial momentum
                    # This measures cumulative drift: as model improves, drift decreases
                    momentum_loss = compute_momentum_drift_per_batch(P_pred_next, P_initial)  # (B,)
                else:
                    # Per-step: consecutive difference in TRUE states
                    # Measures how well true physics conserves momentum (for reference)
                    P_gt_next = compute_linear_momentum(vel_gt_next.to(physics_dtype), mass_phys)  # (B, D)
                    momentum_loss = compute_momentum_drift_per_batch(P_gt_next, P_current)  # (B,)

                momentum_losses_per_step.append(momentum_loss.float())  # (B,)
                
            # Angular momentum loss
            if config.weight_angular_momentum > 0:
                L_current = compute_angular_momentum(pos_rollout_phys, vel_rollout_phys, mass_phys)  # (B,) or (B, 3) - True current
                L_pred_next = compute_angular_momentum(pos_next_phys, vel_pred_phys, mass_phys)  # (B,) or (B, 3) - Predicted next

                if t == 0:
                    L_initial = L_current.clone()  # Store TRUE initial angular momentum

                if use_integrated_angular_drift:
                    # Integrated: drift from initial state (L_pred[t] vs L_initial[0])
                    # Compares PREDICTED angular momentum at each step against TRUE initial angular momentum
                    # This measures cumulative drift: as model improves, drift decreases
                    angular_loss = compute_angular_drift_per_batch(L_pred_next, L_initial)  # (B,)
                else:
                    # Per-step: consecutive difference in TRUE states
                    # Measures how well true physics conserves angular momentum (for reference)
                    L_gt_next = compute_angular_momentum(pos_gt_next.to(physics_dtype), vel_gt_next.to(physics_dtype), mass_phys)
                    angular_loss = compute_angular_drift_per_batch(L_gt_next, L_current)  # (B,)

                angular_losses_per_step.append(angular_loss.float())  # (B,)

        # Update for next iteration (autoregressive)
        pos_rollout = pos_rollout_next
        vel_rollout = vel_pred

    # Second pass: Compute adaptive temporal weights (only if enabled)
    # Stack MSE losses: (rollout_steps, B)
    mse_tensor = torch.stack(mse_losses_per_step, dim=0)  # (T, B)

    if apply_temporal_weighting:
        # Compute cumulative loss sum up to each step (detached to avoid second-order gradients)
        cumsum_losses = torch.cumsum(mse_tensor.detach(), dim=0)  # (T, B)

        # Compute weights: w[t] = exp(-alpha * sum(L[0:t]))
        # For t=0, cumsum is L[0], for t=1, cumsum is L[0]+L[1], etc.
        # We want w[t] based on losses BEFORE step t, so shift cumsum
        cumsum_before = torch.cat([
            torch.zeros(1, B, device=device),  # w[0] has no prior losses
            cumsum_losses[:-1]  # w[t] uses cumsum up to t-1
        ], dim=0)  # (T, B)

        # Compute raw weights and clamp to avoid underflow
        raw_weights = torch.exp(-temporal_decay_alpha * cumsum_before)  # (T, B)
        weights = torch.clamp(raw_weights, min=1e-8, max=1.0)  # (T, B)

        # Normalize weights so they sum to rollout_steps (maintains scale)
        weight_sum = weights.sum(dim=0, keepdim=True)  # (1, B)
        weights = weights * (rollout_steps / (weight_sum + 1e-9))  # (T, B)

        # Apply weights to MSE
        weighted_mse = (mse_tensor * weights).sum(dim=0).mean()  # Sum over time, mean over batch
    else:
        # No temporal weighting - use uniform weights
        weighted_mse = mse_tensor.mean()  # Simple mean over time and batch
        weights = None  # Signal that no temporal weighting was applied

    # Build loss dictionary
    losses = {'mse': mse_tensor.mean().item()}  # Log unweighted MSE for monitoring
    losses['mse_weighted'] = weighted_mse

    # Check if dynamic normalization is enabled
    use_dynamic_norm = getattr(config, 'use_dynamic_normalization', False)

    # Normalize MSE if dynamic normalization is enabled AND we're using temporal weighting
    # IMPORTANT: Skip normalization when temporal weighting is disabled (e.g., validation)
    # because norm_constants are computed from temporally-weighted losses, not raw MSE
    if norm_constants is not None and use_dynamic_norm and apply_temporal_weighting:
        mse_normalized = weighted_mse / norm_constants['mse']
    else:
        mse_normalized = weighted_mse

    total_loss = mse_normalized  # MSE always has implicit weight of 1

    # Add conservation losses (all now have consistent (T, B) shape)
    if config.weight_energy > 0 and energy_losses_per_step:
        energy_tensor = torch.stack(energy_losses_per_step, dim=0)  # (T, B)

        if apply_temporal_weighting and apply_temporal_to_conservation and weights is not None:
            weighted_energy = (energy_tensor * weights).sum(dim=0).mean()
        else:
            weighted_energy = energy_tensor.mean()

        losses['energy'] = weighted_energy
        # Normalize only if dynamic normalization is enabled
        if norm_constants is not None and use_dynamic_norm:
            e_normalized = weighted_energy / norm_constants['energy']
        else:
            e_normalized = weighted_energy
        total_loss = total_loss + config.weight_energy * e_normalized

    if config.weight_momentum > 0 and momentum_losses_per_step:
        # v3.6a: No temporal weighting for momentum (matching v3.5 behavior)
        # Momentum should be conserved at ALL timesteps equally
        momentum_tensor = torch.stack(momentum_losses_per_step, dim=0)  # (T, B)
        weighted_momentum = momentum_tensor.mean()  # Simple mean over time and batch

        losses['momentum'] = weighted_momentum
        # Normalize only if dynamic normalization is enabled
        if norm_constants is not None and use_dynamic_norm:
            p_normalized = weighted_momentum / norm_constants['momentum']
        else:
            p_normalized = weighted_momentum
        total_loss = total_loss + config.weight_momentum * p_normalized

    if config.weight_angular_momentum > 0 and angular_losses_per_step:
        # v3.6a: No temporal weighting for angular momentum (matching v3.5 behavior)
        # Angular momentum should be conserved at ALL timesteps equally
        angular_tensor = torch.stack(angular_losses_per_step, dim=0)  # (T, B)
        weighted_angular = angular_tensor.mean()  # Simple mean over time and batch

        losses['angular_momentum'] = weighted_angular
        # Normalize only if dynamic normalization is enabled
        if norm_constants is not None and use_dynamic_norm:
            l_normalized = weighted_angular / norm_constants['angular_momentum']
        else:
            l_normalized = weighted_angular
        total_loss = total_loss + config.weight_angular_momentum * l_normalized

    losses['total'] = total_loss

    # Log average temporal weight for monitoring (only if temporal weighting was applied)
    if weights is not None:
        losses['avg_temporal_weight'] = weights.mean().item()
    else:
        losses['avg_temporal_weight'] = 1.0  # Uniform weights if no temporal weighting
    
    return losses


@torch.no_grad()
def compute_initial_loss_scales(model, loaders, device, config, num_batches_per_loader=10, use_fp64_training=False):
    """
    Compute initial loss values for normalization using median for robustness.

    Args:
        model: Untrained neural network
        loaders: List of DataLoaders (e.g., [train_loader, val_loader])
        device: Compute device
        config: wandb config
        num_batches_per_loader: Number of batches to sample from each loader
        use_fp64_training: Whether the model is in FP64 mode (affects batch dtype)

    Returns:
        norm_constants: Dict mapping loss names to their median initial values
    """
    model.eval()
    # Collect all values for median computation
    # Note: We use 'weighted' versions since that's what actually gets normalized during training
    all_values = {'mse': [], 'energy': [], 'momentum': [], 'angular_momentum': []}
    batch_dtype = torch.float64 if use_fp64_training else None

    for loader in loaders:
        for i, batch in enumerate(loader):
            if i >= num_batches_per_loader:
                break
            batch = batch_to_device(batch, device, dtype=batch_dtype)

            # Compute weighted losses (no normalization) - use temporal weighting as in training
            # CRITICAL: We must use mse_weighted since that's what gets normalized during training
            losses = compute_loss(batch, model, device, config, norm_constants=None,
                                  apply_temporal_weighting=True)

            # Use the weighted versions for normalization constants
            if 'mse_weighted' in losses:
                all_values['mse'].append(losses['mse_weighted'].item() if isinstance(losses['mse_weighted'], torch.Tensor) else losses['mse_weighted'])
            if 'energy' in losses:
                all_values['energy'].append(losses['energy'].item() if isinstance(losses['energy'], torch.Tensor) else losses['energy'])
            if 'momentum' in losses:
                all_values['momentum'].append(losses['momentum'].item() if isinstance(losses['momentum'], torch.Tensor) else losses['momentum'])
            if 'angular_momentum' in losses:
                all_values['angular_momentum'].append(losses['angular_momentum'].item() if isinstance(losses['angular_momentum'], torch.Tensor) else losses['angular_momentum'])

    # Compute median with robust minimum clamp
    norm_constants = {}

    # First compute MSE median (our reference scale)
    if all_values['mse']:
        mse_median = float(np.median(all_values['mse']))
        norm_constants['mse'] = max(mse_median, 1e-6)
    else:
        norm_constants['mse'] = 1.0

    # For conservation losses, use a floor based on MSE scale
    min_floor = max(norm_constants['mse'] * 0.01, 1e-4)

    for key in ['energy', 'momentum', 'angular_momentum']:
        if all_values[key]:
            median_val = float(np.median(all_values[key]))
            norm_constants[key] = max(median_val, min_floor)
        else:
            norm_constants[key] = 1.0

    return norm_constants


@torch.no_grad()
def update_norm_constants(model, loader, device, config, current_norm,
                          num_batches=5, momentum=0.9, use_fp64_training=False):
    """
    Update normalization constants with exponential moving average.

    EMA formula: new_norm[key] = momentum * old_value + (1-momentum) * new_value
    - momentum=0.9 means 90% old / 10% new (slow adaptation)
    - momentum=0.5 means equal weighting (fast adaptation)

    Args:
        model: Current model
        loader: DataLoader to sample from
        device: Compute device
        config: wandb config
        current_norm: Current normalization constants
        num_batches: Number of batches to sample
        momentum: EMA momentum (fixed at 0.9 for stability)
        use_fp64_training: Whether the model is in FP64 mode (affects batch dtype)

    Returns:
        Updated norm_constants dict
    """
    model.eval()
    all_values = {'mse': [], 'energy': [], 'momentum': [], 'angular_momentum': []}
    batch_dtype = torch.float64 if use_fp64_training else None

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch = batch_to_device(batch, device, dtype=batch_dtype)
        # CRITICAL: Use temporal weighting since that's what's used during training
        losses = compute_loss(batch, model, device, config, norm_constants=None,
                              apply_temporal_weighting=True)

        # Use the weighted versions for normalization constants
        if 'mse_weighted' in losses:
            all_values['mse'].append(losses['mse_weighted'].item() if isinstance(losses['mse_weighted'], torch.Tensor) else losses['mse_weighted'])
        if 'energy' in losses:
            all_values['energy'].append(losses['energy'].item() if isinstance(losses['energy'], torch.Tensor) else losses['energy'])
        if 'momentum' in losses:
            all_values['momentum'].append(losses['momentum'].item() if isinstance(losses['momentum'], torch.Tensor) else losses['momentum'])
        if 'angular_momentum' in losses:
            all_values['angular_momentum'].append(losses['angular_momentum'].item() if isinstance(losses['angular_momentum'], torch.Tensor) else losses['angular_momentum'])

    # Update with EMA
    new_norm = {}
    min_floor = max(current_norm['mse'] * 0.01, 1e-4)

    for key in current_norm:
        if all_values[key]:
            new_val = float(np.median(all_values[key]))
            new_val = max(new_val, min_floor if key != 'mse' else 1e-6)
            # EMA update
            new_norm[key] = momentum * current_norm[key] + (1 - momentum) * new_val
        else:
            new_norm[key] = current_norm[key]

    return new_norm


@torch.no_grad()
def eval_epoch(loader, model, device, config, norm_constants=None, use_amp=False, use_fp64_physics=False,
               apply_temporal_weighting=False, use_fp64_batches=False):
    """
    Evaluate model on full dataset.

    Args:
        loader: DataLoader to evaluate on
        model: The neural network
        device: Compute device
        config: wandb config
        norm_constants: Normalization constants (if None, raw losses are used)
        use_amp: Whether to use automatic mixed precision
        use_fp64_physics: Whether to use FP64 for physics computations
        apply_temporal_weighting: Whether to apply temporal weighting during evaluation.
            Default is False to get true unweighted performance across all timesteps.
            Set to True only if you want to validate with the same weighting as training.
        use_fp64_batches: Whether to convert input batches to FP64 (needed when model is in FP64 mode)
    """
    model.eval()
    totals = {}
    n = 0
    batch_dtype = torch.float64 if use_fp64_batches else None

    for batch in loader:
        batch = batch_to_device(batch, device, dtype=batch_dtype)
        with torch.cuda.amp.autocast(enabled=use_amp):
            losses = compute_loss(batch, model, device, config, norm_constants,
                                  use_fp64_physics=use_fp64_physics,
                                  apply_temporal_weighting=apply_temporal_weighting)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_conservation_metrics(model, test_loader, device, config, steps=500, use_fp64_training=False):
    """
    Evaluate conservation law preservation over long trajectories.

    Uses FP64 for accurate measurement of conservation errors.

    Args:
        model: The neural network
        test_loader: DataLoader for test data
        device: Compute device
        config: wandb config
        steps: Number of steps to rollout
        use_fp64_training: Whether the model is in FP64 mode
    """
    model.eval()
    batch = next(iter(test_loader))

    # Determine dtype based on model precision
    model_dtype = torch.float64 if use_fp64_training else torch.float32
    tracking_dtype = torch.float64  # Always track in FP64 for accurate conservation measurement

    # Get batch and convert to model dtype for forward pass
    pos = batch['pos_seq'][:8, 0].to(device).to(tracking_dtype)
    vel = batch['vel_seq'][:8, 0].to(device).to(tracking_dtype)
    mass = batch['mass'][:8].to(device).to(tracking_dtype)
    dt = config.dt

    E_init = compute_total_energy(pos, vel, mass, config.G, eps=config.softening)
    P_init = compute_linear_momentum(vel, mass)
    L_init = compute_angular_momentum(pos, vel, mass)

    pos_sim, vel_sim = pos.clone(), vel.clone()
    for _ in range(steps):
        # Convert to model dtype for forward pass
        pos_model = pos_sim.to(model_dtype)
        vel_model = vel_sim.to(model_dtype)
        mass_model = mass.to(model_dtype)

        vel_next_model = model(pos_model, vel_model, mass_model)
        vel_half = 0.5 * (vel_model + vel_next_model)
        pos_next_model = pos_model + dt * vel_half

        # Back to tracking dtype for conservation measurement
        pos_sim = pos_next_model.to(tracking_dtype)
        vel_sim = vel_next_model.to(tracking_dtype)

    E_final = compute_total_energy(pos_sim, vel_sim, mass, config.G, eps=config.softening)
    P_final = compute_linear_momentum(vel_sim, mass)
    L_final = compute_angular_momentum(pos_sim, vel_sim, mass)

    # Energy drift: relative change (E_init is large, so this is well-conditioned)
    energy_drift = ((E_final - E_init) / (torch.abs(E_init) + 1e-9)).abs().mean().item()

    # Momentum drift: normalize by sum of individual |m_i * v_i| magnitudes
    # (total momentum P_init ≈ 0 in CoM frame, making relative drift ill-conditioned)
    individual_momenta_init = mass * vel  # (B, N, D)
    P_char = individual_momenta_init.norm(dim=-1).sum(dim=1)  # (B,) - characteristic momentum scale
    dP = P_final - P_init  # (B, D)
    momentum_drift = (torch.norm(dP, dim=-1) / (P_char + 1e-9)).mean().item()

    # Angular momentum drift: normalize by sum of individual |m_i * (r_i x v_i)| magnitudes
    # (total L_init can be small in CoM frame, making relative drift ill-conditioned)
    B, N, D = pos.shape
    total_mass_am = mass.sum(dim=1, keepdim=True)  # (B, 1, 1)
    pos_com = (mass * pos).sum(dim=1, keepdim=True) / total_mass_am  # (B, 1, D)
    vel_com = (mass * vel).sum(dim=1, keepdim=True) / total_mass_am  # (B, 1, D)
    pos_rel = pos - pos_com  # (B, N, D)
    vel_rel = vel - vel_com  # (B, N, D)
    dL = L_final - L_init
    if D == 2:
        cross_per_body = pos_rel[..., 0] * vel_rel[..., 1] - pos_rel[..., 1] * vel_rel[..., 0]  # (B, N)
        L_char = (mass.squeeze(-1) * cross_per_body).abs().sum(dim=1)  # (B,)
        angular_drift = (dL.abs() / (L_char + 1e-9)).mean().item()
    else:
        cross_per_body = torch.cross(pos_rel, vel_rel, dim=-1)  # (B, N, 3)
        L_per_body = mass * cross_per_body  # (B, N, 3)
        L_char = L_per_body.norm(dim=-1).sum(dim=1)  # (B,) - sum of individual |L_i|
        angular_drift = (torch.norm(dL, dim=-1) / (L_char + 1e-9)).mean().item()

    return {
        'energy_drift': energy_drift,
        'momentum_drift': momentum_drift,
        'angular_momentum_drift': angular_drift
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """
    Main training function - called for each wandb sweep run.

    v3.6: Removed Fourier features, added drift modes, early stopping, per-batch losses.
    """
    run = wandb.init()
    config = wandb.config

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: {dict(config)}")

    # -------------------------------------------------------------------------
    # Create datasets
    # -------------------------------------------------------------------------
    print("Generating datasets...")

    train_ds = NBodyTrajDataset(
        num_traj=config.num_train_traj,
        steps=config.traj_steps,
        N=config.n_bodies,
        dt=config.dt,
        dim=config.dim,
        G=config.G,
        softening=config.softening,
        device='cpu',
        seed=123,
        rollout_steps=config.rollout_steps
    )

    val_ds = NBodyTrajDataset(
        num_traj=config.num_val_traj,
        steps=config.traj_steps,
        N=config.n_bodies,
        dt=config.dt,
        dim=config.dim,
        G=config.G,
        softening=config.softening,
        device='cpu',
        seed=456,
        rollout_steps=config.rollout_steps
    )

    test_ds = NBodyTrajDataset(
        num_traj=config.num_test_traj,
        steps=config.traj_steps,
        N=config.n_bodies,
        dt=config.dt,
        dim=config.dim,
        G=config.G,
        softening=config.softening,
        device='cpu',
        seed=789,
        rollout_steps=config.rollout_steps
    )

    # Windows has multiprocessing issues with num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} samples")

    # -------------------------------------------------------------------------
    # Create model, optimizer, and scheduler
    # -------------------------------------------------------------------------
    hidden_sizes = getattr(config, 'hidden_sizes', [256, 256, 256, 128])
    if isinstance(hidden_sizes, str):
        import ast
        hidden_sizes = ast.literal_eval(hidden_sizes)

    model = DirectOrbitMLP(
        n_bodies=config.n_bodies, 
        dim=config.dim, 
        hidden_sizes=hidden_sizes,
    ).to(device)
    
    print(f"Model architecture: hidden_sizes={hidden_sizes}")
    print(f"Input features: {config.n_bodies * (2 * config.dim + 1)} (raw pos + vel + mass)")
    
    # FP64 training mode (disabled by default, ~2-3x slower but more numerically stable)
    # When enabled, model forward/backward runs in FP64. Optimizer stays in FP32 for stability.
    use_fp64_training = getattr(config, 'use_fp64_training', False)
    if use_fp64_training:
        # Use to() with dtype for more robust conversion than .double()
        # This ensures all tensors (including buffers) are converted
        model = model.to(dtype=torch.float64)
        print("FP64 training mode enabled (slower but more numerically stable)")
    
    # Compile model for faster execution (Linux only)
    import platform
    if device.type == 'cuda' and platform.system() != 'Windows':
        print("Compiling model with torch.compile (max-autotune mode)...")
        model = torch.compile(model, mode="max-autotune")
    elif device.type == 'cuda':
        print("Skipping torch.compile (not supported on Windows)")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create LR scheduler
    scheduler, is_plateau_scheduler = create_scheduler(optimizer, config)
    scheduler_type = getattr(config, 'scheduler_type', 'cosine')
    print(f"LR Scheduler: {scheduler_type}")

    # Create AMP GradScaler for mixed precision training (disabled for FP64 mode)
    use_amp = device.type == 'cuda' and not use_fp64_training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision training enabled (AMP)")
    elif use_fp64_training:
        print("AMP disabled (using FP64 training)")

    # v3.6: Log drift mode configurations
    use_integrated_energy_drift = getattr(config, 'use_integrated_energy_drift', True)
    use_integrated_momentum_drift = getattr(config, 'use_integrated_momentum_drift', True)
    use_integrated_angular_drift = getattr(config, 'use_integrated_angular_drift', True)
    print(f"Drift modes: energy={use_integrated_energy_drift}, momentum={use_integrated_momentum_drift}, angular={use_integrated_angular_drift}")
    
    # Adaptive temporal weighting settings
    temporal_decay_alpha = getattr(config, 'temporal_decay_alpha', 1.0)
    apply_temporal_to_conservation = getattr(config, 'apply_temporal_to_conservation', True)
    print(f"Adaptive temporal weighting: alpha={temporal_decay_alpha}, apply_to_conservation={apply_temporal_to_conservation}")

    # Dynamic gradient balancing settings
    norm_update_interval = getattr(config, 'norm_update_interval', 10)
    norm_update_momentum = getattr(config, 'norm_update_momentum', 0.9)
    print(f"Dynamic gradient balancing: update_interval={norm_update_interval}, momentum={norm_update_momentum}")
    
    # Early stopping settings
    early_stopping_patience = getattr(config, 'early_stopping_patience', 20)
    print(f"Early stopping: patience={early_stopping_patience} epochs")

    # -------------------------------------------------------------------------
    # Compute initial loss scales for normalization (if enabled)
    # -------------------------------------------------------------------------
    use_dynamic_norm = getattr(config, 'use_dynamic_normalization', False)

    if use_dynamic_norm:
        print("Computing initial loss scales for normalization (using train + val data)...")
        norm_constants = compute_initial_loss_scales(
            model, [train_loader, val_loader], device, config, use_fp64_training=use_fp64_training
        )

        print(f"  MSE norm: {norm_constants['mse']:.6f}")
        print(f"  Energy norm: {norm_constants['energy']:.6f}")
        print(f"  Momentum norm: {norm_constants['momentum']:.6f}")
        print(f"  Angular momentum norm: {norm_constants['angular_momentum']:.6f}")

        # Log normalization constants to wandb
        wandb.config.update({
            'norm_mse': norm_constants['mse'],
            'norm_energy': norm_constants['energy'],
            'norm_momentum': norm_constants['momentum'],
            'norm_angular_momentum': norm_constants['angular_momentum'],
        }, allow_val_change=True)
    else:
        print("Dynamic normalization disabled - using per-sample normalization only")
        norm_constants = None

    # Log drift modes to wandb
    wandb.config.update({
        'drift_mode_energy': 'integrated' if use_integrated_energy_drift else 'per_step',
        'drift_mode_momentum': 'integrated' if use_integrated_momentum_drift else 'per_step',
        'drift_mode_angular': 'integrated' if use_integrated_angular_drift else 'per_step',
    }, allow_val_change=True)

    # -------------------------------------------------------------------------
    # Training loop with early stopping
    # -------------------------------------------------------------------------
    best_val_mse = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            # Pass dtype directly to batch_to_device for FP64 training
            # This converts tensors to the correct dtype on the target device,
            # avoiding issues with AMP scaler consistency
            batch_dtype = torch.float64 if use_fp64_training else None
            batch = batch_to_device(batch, device, dtype=batch_dtype)

            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass (disabled for FP64 mode)
            with torch.cuda.amp.autocast(enabled=use_amp):
                losses = compute_loss(batch, model, device, config, norm_constants, 
                                     use_fp64_physics=use_fp64_training)
            
            # Scaled backward pass for AMP
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
            n_batches += 1

        train_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        # Validation with FP64 physics and FP64 batches (if model is in FP64 mode)
        val_losses = eval_epoch(val_loader, model, device, config, norm_constants,
                                use_amp=use_amp, use_fp64_physics=True,
                                use_fp64_batches=use_fp64_training)

        # Dynamic gradient balancing: update norm constants periodically (if enabled)
        if use_dynamic_norm and norm_update_interval > 0 and epoch % norm_update_interval == 0:
            old_norm = norm_constants.copy()
            norm_constants = update_norm_constants(
                model, train_loader, device, config, norm_constants,
                momentum=norm_update_momentum, use_fp64_training=use_fp64_training
            )
            # Log norm constant updates
            wandb.log({
                'norm_update/mse_old': old_norm['mse'],
                'norm_update/mse_new': norm_constants['mse'],
                'norm_update/energy_old': old_norm['energy'],
                'norm_update/energy_new': norm_constants['energy'],
                'norm_update/momentum_old': old_norm['momentum'],
                'norm_update/momentum_new': norm_constants['momentum'],
                'norm_update/angular_old': old_norm['angular_momentum'],
                'norm_update/angular_new': norm_constants['angular_momentum'],
            }, step=epoch)

        # Step the scheduler
        if scheduler is not None:
            if is_plateau_scheduler:
                scheduler.step(val_losses['mse'])
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Compute raw validation total (without normalization) for backward compatibility
        val_raw_total = val_losses.get('mse_weighted', val_losses['mse'])
        for key in ['energy', 'momentum', 'angular_momentum']:
            if key in val_losses:
                weight = getattr(config, f'weight_{key}', 0)
                val_raw_total = val_raw_total + weight * val_losses[key]

        # Log to wandb
        log_dict = {
            'epoch': epoch,
            'lr': current_lr,
            'train/total': train_losses['total'],
            'train/mse_raw': train_losses['mse'],
            'train/mse_weighted': train_losses.get('mse_weighted', train_losses['mse']),
            'train/avg_temporal_weight': train_losses.get('avg_temporal_weight', 1.0),
            'val/total': val_losses['total'],
            'val/total_raw': val_raw_total,
            'val/mse_raw': val_losses['mse'],
            'val/mse_weighted': val_losses.get('mse_weighted', val_losses['mse']),
        }

        # Add norm constants to log dict only if dynamic normalization is enabled
        if use_dynamic_norm and norm_constants is not None:
            log_dict.update({
                'norm_mse': norm_constants['mse'],
                'norm_energy': norm_constants['energy'],
                'norm_momentum': norm_constants['momentum'],
                'norm_angular_momentum': norm_constants['angular_momentum']
            })

        for key in ['energy', 'momentum', 'angular_momentum']:
            if key in train_losses:
                log_dict[f'train/{key}_raw'] = train_losses[key]
            if key in val_losses:
                log_dict[f'val/{key}_raw'] = val_losses[key]

        wandb.log(log_dict)

        # Track best model (based on validation MSE)
        val_mse = val_losses['mse']
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best checkpoint
            checkpoint_path = "checkpoint_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_mse,
                'norm_constants': norm_constants,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_mse={val_mse:.6f})")
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"LR: {current_lr:.2e} | "
                  f"Train Total: {train_losses['total']:.4f} | "
                  f"Val Total: {val_losses['total']:.4f} | "
                  f"Val MSE: {val_losses['mse']:.6f} | "
                  f"Avg Weight: {train_losses.get('avg_temporal_weight', 1.0):.4f} | "
                  f"Best: {best_val_mse:.6f} @ {best_epoch}")

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
            wandb.log({
                'early_stopping/triggered': True,
                'early_stopping/epoch': epoch,
                'early_stopping/best_epoch': best_epoch,
            })
            wandb.summary['early_stopping_epoch'] = epoch
            break

    # -------------------------------------------------------------------------
    # Load best model for final evaluation
    # -------------------------------------------------------------------------
    if os.path.exists("checkpoint_best.pt"):
        print(f"\nLoading best checkpoint from epoch {best_epoch}...")
        checkpoint = torch.load("checkpoint_best.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------
    print("\nEvaluating conservation metrics...")
    conservation_metrics = evaluate_conservation_metrics(model, test_loader, device, config,
                                                          use_fp64_training=use_fp64_training)

    wandb.log({
        'final/val_mse': best_val_mse,
        'final/best_epoch': best_epoch,
        'final/energy_drift': conservation_metrics['energy_drift'],
        'final/momentum_drift': conservation_metrics['momentum_drift'],
        'final/angular_momentum_drift': conservation_metrics['angular_momentum_drift'],
    })

    wandb.summary['best_val_mse'] = best_val_mse
    wandb.summary['best_epoch'] = best_epoch
    wandb.summary['energy_drift'] = conservation_metrics['energy_drift']
    wandb.summary['momentum_drift'] = conservation_metrics['momentum_drift']
    wandb.summary['angular_momentum_drift'] = conservation_metrics['angular_momentum_drift']

    print(f"\nFinal Results (from best checkpoint at epoch {best_epoch}):")
    print(f"  Best Val MSE: {best_val_mse:.6f}")
    print(f"  Energy Drift: {conservation_metrics['energy_drift']:.4f}")
    print(f"  Momentum Drift: {conservation_metrics['momentum_drift']:.4f}")
    print(f"  Angular Momentum Drift: {conservation_metrics['angular_momentum_drift']:.4f}")

    # Save final model
    model_path = f"model_{wandb.run.id}.pt"
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
    artifact.add_file(model_path)
    # Also add best checkpoint to artifact
    if os.path.exists("checkpoint_best.pt"):
        artifact.add_file("checkpoint_best.pt")
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    train()
