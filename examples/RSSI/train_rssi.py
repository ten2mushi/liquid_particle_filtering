#!/usr/bin/env python3
"""RSSI Particle Filter Training: PFNCP with AutoNCP Wiring

Complete training pipeline for RF localization using Particle Filter Neural
Circuit Policies (PFNCP) with AutoNCP wiring architecture.

Architecture:
    Input (5) → AutoNCP → Output (2)
    - Sensory: [RSSI_front, RSSI_back, rotation, speed, sensor_heading]
    - Motor: [distance_norm, bearing_norm] - polar representation in sensor frame
    - No encoder layer - direct sensory input per task requirements

Output Representation:
    - distance_norm: distance to target / r_max, normalized [0, 1]
    - bearing_norm: angle to target relative to sensor heading, normalized [0, 1]

Run: python examples/RSSI/train_rssi.py --data_dir examples/RSSI/data/dpf_dataset_polar_double_single_target
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import uuid
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "examples/RSSI/data/dpf_dataset_polar_double_single_target"
    env_size: float = 200.0  # meters (for bearing computation)
    r_max: float = 150.0     # Max detection range for distance normalization
    rssi_min: float = -100.0  # dBm normalization
    rssi_max: float = -20.0   # dBm normalization
    trigonometric_processing: bool = False  # Use sin/cos encoding for angles



@dataclass
class ModelConfig:
    """PFNCP model configuration."""
    # ----- Core Architecture -----
    input_size: int = 5      # Default: [RSSI_front, RSSI_back, rotation, speed, sensor_heading]
    output_size: int = 2     # Default: [distance_norm, bearing_norm]
    units: int = 32          # AutoNCP hidden units
    trigonometric_processing: bool = False  # Use sin/cos encoding

    def __post_init__(self):
        # Adjust sizes if trigonometric processing is enabled
        # Input: 5 -> 6 (heading -> sin, cos)
        # Output: 2 -> 3 (bearing -> sin, cos)
        if self.trigonometric_processing:
            if self.input_size == 5:
                self.input_size = 6
            if self.output_size == 2:
                self.output_size = 3


    # ----- Particle Filter -----
    n_particles: int = 8
    approach: str = "state"  # "state", "param", "dual", "sde"

    # ----- CfC Cell Parameters -----
    mode: str = "default"    # "default", "pure", "no_gate"

    # ----- Particle Filter Noise -----
    # noise_type: "learned", "time_scaled", "constant", "state_dependent"
    noise_type: str = "learned"
    noise_init: float = 0.1

    # ----- Soft Resampling -----
    # alpha_mode: "adaptive", "fixed", "learnable"
    alpha_mode: str = "adaptive"
    alpha_init: float = 0.5
    resample_threshold: float = 0.5

    # ----- Observation Model -----
    # obs_model: "gaussian", "heteroscedastic", "learned", "energy", "attention"
    obs_model: str = "heteroscedastic"
    obs_noise_std: float = 0.05
    obs_min_noise: float = 0.01
    obs_max_noise: float = 1.0
    obs_hidden_sizes: List[int] = field(default_factory=lambda: [16, 8])
    learnable_obs_noise: bool = True

    # ----- Spatial Heatmap -----
    map_size: int = 64       # Heatmap resolution (H x W)
    sigma_min: float = 0.01  # Minimum spatial uncertainty
    sigma_max: float = 0.5   # Maximum spatial uncertainty
    projection_hidden: int = 16  # Hidden size for projection MLP
    egocentric: bool = True  # Sensor-centered heatmaps (True) or world-frame (False)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    device: str = "auto"

    # ----- Loss Configuration -----
    # loss_type: "mse", "huber", "weighted_mse", "nll"
    loss_type: str = "huber"
    loss_beta: float = 0.1  # Beta for Huber loss
    use_circular_bearing: bool = True  # Circular loss for bearing (handles wraparound)

    # ----- Loss Weights (Polar representation) -----
    distance_weight: float = 1.0  # Weight for distance loss
    bearing_weight: float = 1.0   # Weight for bearing loss

    # ----- Heatmap Loss Configuration -----
    use_heatmap_loss: bool = True  # Enable spatial heatmap loss
    heatmap_weight: float = 0.1   # Weight for heatmap NLL loss

    # ----- Performance Options -----
    num_workers: int = 4  # DataLoader workers (0 for single-threaded)
    use_compile: bool = True  # Use torch.compile if available
    ess_log_interval: int = 10  # Log ESS every N batches (for efficiency)
    trigonometric_processing: bool = False  # Use sin/cos encoding for angles



@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    output_dir: str = "examples/RSSI/output"
    dpi: int = 150
    n_animation_sequences: int = 5
    animation_fps: int = 10
    # Debug visualization
    debug_viz: bool = False
    n_debug_episodes: int = 3


# =============================================================================
# 1b. CONFIG LOADING UTILITIES
# =============================================================================

def generate_run_id(name: str = "pfncp") -> str:
    """Generate unique run identifier."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"{name}_{short_uuid}"


def load_model_config(config_path: str) -> dict:
    """Load model configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, output_path: Path):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_to_dataclasses(config: dict, data_dir: str, output_dir: str) -> Tuple[DataConfig, ModelConfig, TrainingConfig, VisualizationConfig]:
    """Convert YAML config dict to dataclass instances.
    
    Args:
        config: Loaded YAML config dictionary
        data_dir: Override for data directory (from CLI)
        output_dir: Override for output directory (from CLI)
        
    Returns:
        Tuple of (DataConfig, ModelConfig, TrainingConfig, VisualizationConfig)
    """
    # Data config (load from dataset metadata if available)
    data_path = Path(data_dir)
    metadata_path = data_path / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        dataset_cfg = metadata.get('config', {})
        data_cfg = DataConfig(
            data_dir=data_dir,
            env_size=dataset_cfg.get('env_size', 200.0),
            r_max=dataset_cfg.get('r_max', 150.0),
            rssi_min=-100.0,
            rssi_max=-20.0,
            trigonometric_processing=config.get('model', {}).get('trigonometric_processing', False),
        )
    else:
        data_cfg = DataConfig(
            data_dir=data_dir,
            trigonometric_processing=config.get('model', {}).get('trigonometric_processing', False),
        )
    
    # Model config
    model_sect = config.get('model', {})
    model_cfg = ModelConfig(
        input_size=model_sect.get('input_size', 5),
        output_size=model_sect.get('output_size', 2),
        trigonometric_processing=model_sect.get('trigonometric_processing', False),

        units=model_sect.get('units', 64),
        n_particles=model_sect.get('n_particles', 32),
        approach=model_sect.get('approach', 'state'),
        mode=model_sect.get('mode', 'default'),
        noise_type=model_sect.get('noise_type', 'learned'),
        noise_init=model_sect.get('noise_init', 0.1),
        alpha_mode=model_sect.get('alpha_mode', 'adaptive'),
        alpha_init=model_sect.get('alpha_init', 0.5),
        resample_threshold=model_sect.get('resample_threshold', 0.5),
        obs_model=model_sect.get('obs_model', 'heteroscedastic'),
        obs_noise_std=model_sect.get('obs_noise_std', 0.1),
        obs_min_noise=model_sect.get('obs_min_noise', 0.01),
        obs_max_noise=model_sect.get('obs_max_noise', 1.0),
        obs_hidden_sizes=model_sect.get('obs_hidden_sizes', [16, 8]),
        learnable_obs_noise=model_sect.get('learnable_obs_noise', True),
        # Spatial heatmap
        map_size=model_sect.get('map_size', 64),
        sigma_min=model_sect.get('sigma_min', 0.01),
        sigma_max=model_sect.get('sigma_max', 0.5),
        projection_hidden=model_sect.get('projection_hidden', 16),
        egocentric=model_sect.get('egocentric', True),
    )

    # Training config
    train_sect = config.get('training', {})
    train_cfg = TrainingConfig(
        epochs=train_sect.get('epochs', 30),
        batch_size=train_sect.get('batch_size', 16),
        learning_rate=train_sect.get('learning_rate', 1e-3),
        grad_clip=train_sect.get('grad_clip', 1.0),
        device=train_sect.get('device', 'auto'),
        loss_type=train_sect.get('loss_type', 'huber'),
        loss_beta=train_sect.get('loss_beta', 0.1),
        use_circular_bearing=train_sect.get('use_circular_bearing', True),
        distance_weight=train_sect.get('distance_weight', 1.0),
        bearing_weight=train_sect.get('bearing_weight', 1.0),
        # Heatmap loss
        use_heatmap_loss=train_sect.get('use_heatmap_loss', True),
        heatmap_weight=train_sect.get('heatmap_weight', 0.1),
        # Performance
        num_workers=train_sect.get('num_workers', 4),
        use_compile=train_sect.get('use_compile', True),
        ess_log_interval=train_sect.get('ess_log_interval', 10),
        trigonometric_processing=config.get('model', {}).get('trigonometric_processing', False),
    )

    
    # Visualization config
    viz_sect = config.get('visualization', {})
    viz_cfg = VisualizationConfig(
        output_dir=output_dir,
        dpi=viz_sect.get('dpi', 150),
        n_animation_sequences=viz_sect.get('n_animation_sequences', 5),
        animation_fps=viz_sect.get('animation_fps', 10),
        debug_viz=viz_sect.get('debug_viz', False),
        n_debug_episodes=viz_sect.get('n_debug_episodes', 3),
    )
    
    return data_cfg, model_cfg, train_cfg, viz_cfg


# =============================================================================
# 2. DATASET
# =============================================================================

class RSSIDataset(torch.utils.data.Dataset):
    """Dataset for pre-generated RSSI trajectories.

    Outputs polar representation: (distance_norm, bearing_norm) in sensor frame.
    - distance_norm: distance to target / r_max, clamped to [0, 1]
    - bearing_norm: bearing to target relative to sensor heading, normalized [0, 1]

    Also returns target_pos for spatial heatmap supervision.
    """

    def __init__(
        self,
        trajectories: List[Dict],
        rssi_min: float = -100.0,
        rssi_max: float = -20.0,

        r_max: float = 150.0,  # Max range for distance normalization
        trigonometric_processing: bool = False,
    ):
        self.trajectories = trajectories
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max
        self.r_max = r_max
        self.trigonometric_processing = trigonometric_processing

    def __len__(self) -> int:

        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        traj = self.trajectories[idx]

        # === INPUTS (unchanged) ===
        # Observations: (T, 2) RSSI -> normalize to [-1, 1]
        rssi = torch.from_numpy(traj['observations']).float()
        rssi_norm = 2.0 * (rssi - self.rssi_min) / (self.rssi_max - self.rssi_min) - 1.0
        rssi_norm = rssi_norm.clamp(-1.0, 1.0)

        # Actions: (T, 2) -> proper normalization
        actions = torch.from_numpy(traj['actions']).float()
        rotation_norm = actions[:, 0:1] / 30.0   # [-30,0,30] -> [-1,0,1]
        speed_norm = actions[:, 1:2] / 4.0       # [0,4] -> [0,1]

        # Sensor positions: (T, 3) [x, y, heading] where heading is normalized [0, 1]
        sensor_pos = torch.from_numpy(traj['sensor_positions']).float()
        
        if self.trigonometric_processing:
            # Convert normalized heading [0,1] -> radians [-pi, pi] -> sin/cos
            heading_rad = sensor_pos[:, 2:3] * 2 * np.pi - np.pi
            sin_h = torch.sin(heading_rad)
            cos_h = torch.cos(heading_rad)
            # Input: (T, 6) [rssi_front, rssi_back, rot, speed, sin_h, cos_h]
            inputs = torch.cat([rssi_norm, rotation_norm, speed_norm, sin_h, cos_h], dim=-1)
        else:
            sensor_heading_norm = sensor_pos[:, 2:3]  # (T, 1) already in [0, 1]
            # Input: (T, 5) [rssi_front, rssi_back, rot, speed, heading_norm]
            inputs = torch.cat([rssi_norm, rotation_norm, speed_norm, sensor_heading_norm], dim=-1)


        # === TARGETS: POLAR REPRESENTATION ===
        # Output (distance_norm, bearing_norm) in sensor-centric frame

        # 1. Distance: use true_distances directly, normalize by r_max
        true_dist = torch.from_numpy(traj['true_distances']).float()  # (T,) in meters
        dist_norm = (true_dist / self.r_max).clamp(0.0, 1.0)  # [0, 1]

        # 2. Bearing: compute angle to target relative to sensor heading
        # Target absolute position in [0, 1] normalized coordinates
        target_pos = torch.from_numpy(traj['positions'][:, :2]).float()  # (T, 2) normalized [0,1]
        sensor_xy = sensor_pos[:, :2]  # (T, 2) normalized [0,1]

        # Vector from sensor to target (in normalized coords)
        dx = target_pos[:, 0] - sensor_xy[:, 0]
        dy = target_pos[:, 1] - sensor_xy[:, 1]

        # World-frame bearing (angle from world X-axis to target)
        world_bearing = torch.atan2(dy, dx)  # [-π, π]

        # Sensor heading in radians: [0,1] -> [-π, π]
        sensor_heading_rad = sensor_pos[:, 2] * 2 * np.pi - np.pi

        # Relative bearing (angle from sensor's heading to target)
        rel_bearing = world_bearing - sensor_heading_rad

        # Wrap to [-π, π] using atan2
        rel_bearing = torch.atan2(torch.sin(rel_bearing), torch.cos(rel_bearing))

        if self.trigonometric_processing:
            # Targets: (T, 3) [distance_norm, sin_bearing, cos_bearing]
            # rel_bearing is in [-pi, pi]
            sin_b = torch.sin(rel_bearing)
            cos_b = torch.cos(rel_bearing)
            targets = torch.stack([dist_norm, sin_b, cos_b], dim=-1)
        else:
            # Normalize to [0, 1]: [-π, π] -> [0, 1]
            bearing_norm = (rel_bearing + np.pi) / (2 * np.pi)
            
            # Targets: (T, 2) [distance_norm, bearing_norm]
            targets = torch.stack([dist_norm, bearing_norm], dim=-1)


        length = traj['episode_length']

        # Return target_pos for heatmap supervision
        return inputs, targets, sensor_pos, target_pos, length



def collate_fn(batch):
    """Collate variable-length trajectories with padding."""
    inputs_list, targets_list, sensor_list, target_pos_list, lengths = zip(*batch)
    lengths = torch.tensor(lengths)

    inputs_padded = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)
    targets_padded = nn.utils.rnn.pad_sequence(targets_list, batch_first=True)
    sensor_padded = nn.utils.rnn.pad_sequence(sensor_list, batch_first=True)
    target_pos_padded = nn.utils.rnn.pad_sequence(target_pos_list, batch_first=True)

    return inputs_padded, targets_padded, sensor_padded, target_pos_padded, lengths


def load_data(data_dir: str, data_cfg: DataConfig):
    """Load pre-generated train and validation datasets."""
    data_path = Path(data_dir)
    
    train_data = torch.load(data_path / "train_dataset.pt", weights_only=False)
    val_data = torch.load(data_path / "val_dataset.pt", weights_only=False)
    
    train_dataset = RSSIDataset(
        train_data, data_cfg.rssi_min, data_cfg.rssi_max, data_cfg.r_max,
        trigonometric_processing=data_cfg.trigonometric_processing
    )
    val_dataset = RSSIDataset(
        val_data, data_cfg.rssi_min, data_cfg.rssi_max, data_cfg.r_max,
        trigonometric_processing=data_cfg.trigonometric_processing
    )
    
    print(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val trajectories")
    
    return train_dataset, val_dataset



# =============================================================================
# 3. OBSERVATION MODEL FACTORY
# =============================================================================

def create_observation_model(cfg: ModelConfig, hidden_size: int):
    """Create observation model based on configuration.
    
    Args:
        cfg: Model configuration with obs_model type
        hidden_size: Hidden size from the wiring (motor_size)
        
    Returns:
        ObservationModel instance
    """
    from pfncps.nn import (
        GaussianObservationModel,
        HeteroscedasticGaussianObservationModel,
        LearnedMLPObservationModel,
        EnergyBasedObservationModel,
        AttentionObservationModel,
    )
    
    obs_type = cfg.obs_model.lower()
    
    if obs_type == "gaussian":
        return GaussianObservationModel(
            hidden_size=hidden_size,
            obs_size=cfg.output_size,
            obs_noise_std=cfg.obs_noise_std,
            learnable_noise=cfg.learnable_obs_noise,
        )
    
    elif obs_type == "heteroscedastic":
        return HeteroscedasticGaussianObservationModel(
            hidden_size=hidden_size,
            obs_size=cfg.output_size,
            min_noise_std=cfg.obs_min_noise,
            max_noise_std=cfg.obs_max_noise,
            init_noise_std=cfg.obs_noise_std,
        )
    
    elif obs_type == "learned":
        return LearnedMLPObservationModel(
            hidden_size=hidden_size,
            obs_size=cfg.output_size,
            mlp_hidden_sizes=cfg.obs_hidden_sizes,
            activation="tanh",
        )
    
    elif obs_type == "energy":
        return EnergyBasedObservationModel(
            hidden_size=hidden_size,
            obs_size=cfg.output_size,
            energy_hidden_sizes=cfg.obs_hidden_sizes,
            temperature=1.0,
            learnable_temperature=True,
        )
    
    elif obs_type == "attention":
        return AttentionObservationModel(
            hidden_size=hidden_size,
            obs_size=cfg.output_size,
            n_heads=4,
        )
    
    else:
        raise ValueError(
            f"Unknown observation model: {obs_type}. "
            "Choose from: gaussian, heteroscedastic, learned, energy, attention"
        )


# =============================================================================
# 4. LOSS COMPUTATION
# =============================================================================

def circular_mse(pred_h: torch.Tensor, gt_h: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss with circular wraparound for bearing.
    
    Handles the case where 0.01 and 0.99 should have error ~0.02, not ~0.98.
    Assumes bearing is normalized to [0, 1].
    
    Args:
        pred_h: Predicted bearing [batch, seq, 1] or [seq]
        gt_h: Ground truth bearing [batch, seq, 1] or [seq]
        
    Returns:
        MSE loss with circular correction
    """
    diff = pred_h - gt_h
    # Wrap difference to [-0.5, 0.5]
    diff = torch.remainder(diff + 0.5, 1.0) - 0.5
    return (diff ** 2).mean()


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    train_cfg: TrainingConfig,
    r_max: float = 150.0,
    trigonometric_processing: bool = False,

) -> torch.Tensor:
    """Compute loss for polar (distance, bearing) representation.
    
    Args:
        predictions: Model predictions [batch, seq, 2] (distance_norm, bearing_norm)
        targets: Ground truth [batch, seq, 2] (distance_norm, bearing_norm)
        lengths: Sequence lengths [batch]
        train_cfg: Training configuration
        r_max: Max range for distance (unused in normalized loss, kept for API)
        
    Returns:
        Scalar loss tensor
    """
    device = predictions.device
    batch_size, max_len, _ = predictions.shape
    
    # Create sequence mask for variable lengths [B, T]
    if isinstance(lengths, torch.Tensor):
        lengths_t = lengths.to(device)
    else:
        lengths_t = torch.tensor(lengths, device=device)
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
    mask = (range_tensor < lengths_t.unsqueeze(1)).float()  # [B, T]
    
    # Extract distance and bearing
    pred_dist = predictions[:, :, 0]  # [B, T]
    gt_dist = targets[:, :, 0]
    
    # ----- Distance Loss (Huber) -----
    dist_diff = torch.abs(pred_dist - gt_dist)
    huber_dist = torch.where(
        dist_diff < train_cfg.loss_beta,
        0.5 * dist_diff ** 2 / train_cfg.loss_beta,
        dist_diff - 0.5 * train_cfg.loss_beta
    )
    dist_loss = (huber_dist * mask).sum() / (mask.sum() + 1e-8)
    
    # ----- Bearing Loss -----
    if trigonometric_processing:
        # Targets/Preds are [dist, sin, cos]
        pred_sin = predictions[:, :, 1]
        pred_cos = predictions[:, :, 2]
        gt_sin = targets[:, :, 1]
        gt_cos = targets[:, :, 2]
        
        # Vector MSE for angle (sin^2 + cos^2)
        sin_diff = (pred_sin - gt_sin) ** 2
        cos_diff = (pred_cos - gt_cos) ** 2
        bear_diff = sin_diff + cos_diff
        
        bear_loss = (bear_diff * mask).sum() / (mask.sum() + 1e-8)
    else:
        # Targets/Preds are [dist, angle_norm]
        pred_bear = predictions[:, :, 1]
        pred_bear = pred_bear.squeeze(-1) if pred_bear.dim() == 3 else pred_bear
        gt_bear = targets[:, :, 1]
        gt_bear = gt_bear.squeeze(-1) if gt_bear.dim() == 3 else gt_bear
        
        if train_cfg.use_circular_bearing:
            diff_bear = pred_bear - gt_bear
            diff_bear = torch.remainder(diff_bear + 0.5, 1.0) - 0.5  # Wrap to [-0.5, 0.5]
            bear_loss = ((diff_bear ** 2) * mask).sum() / (mask.sum() + 1e-8)
        else:
            bear_diff = (pred_bear - gt_bear) ** 2
            bear_loss = (bear_diff * mask).sum() / (mask.sum() + 1e-8)
    
    # ----- Combined Loss -----
    total_loss = train_cfg.distance_weight * dist_loss + train_cfg.bearing_weight * bear_loss


    return total_loss


def compute_heatmap_nll(
    heatmaps: torch.Tensor,
    true_positions: torch.Tensor,
    sensor_positions: torch.Tensor,
    lengths: torch.Tensor,
    r_max: float = 150.0,
    env_size: float = 200.0,
    egocentric: bool = True,
) -> torch.Tensor:
    """Compute negative log-likelihood of heatmaps at true target positions.

    Encourages the spatial heatmap to assign high probability to ground truth.

    For egocentric mode, transforms world-frame true_positions to sensor-centered
    frame before sampling the heatmap.

    Args:
        heatmaps: Spatial probability [batch, seq, 1, H, W]
        true_positions: Target positions [batch, seq, 2] (x_norm, y_norm) in [0, 1] world frame
        sensor_positions: Sensor positions [batch, seq, 3] (x_norm, y_norm, heading_norm)
        lengths: Sequence lengths [batch]
        r_max: Maximum detection range in meters
        env_size: Environment size in meters
        egocentric: If True, heatmaps are in sensor-centered frame

    Returns:
        Scalar NLL loss (lower = better calibration)
    """
    device = heatmaps.device
    B, T, _, H, W = heatmaps.shape

    # Create sequence mask
    if isinstance(lengths, torch.Tensor):
        lengths_t = lengths.to(device)
    else:
        lengths_t = torch.tensor(lengths, device=device)
    range_tensor = torch.arange(T, device=device).unsqueeze(0)
    mask = (range_tensor < lengths_t.unsqueeze(1)).float()  # [B, T]

    if egocentric:
        # Transform true_positions from world frame to sensor-centered frame
        # 1. Convert positions to meters centered at world origin
        target_m = (true_positions - 0.5) * env_size  # [B, T, 2] meters
        sensor_m = (sensor_positions[..., :2] - 0.5) * env_size  # [B, T, 2] meters
        sensor_heading = sensor_positions[..., 2] * 2 * np.pi - np.pi  # [B, T] radians

        # 2. Translate: target position relative to sensor
        dx = target_m[..., 0] - sensor_m[..., 0]  # [B, T]
        dy = target_m[..., 1] - sensor_m[..., 1]  # [B, T]

        # 3. Rotate by -sensor_heading to get sensor-frame coordinates
        # Rotation matrix: [[cos, -sin], [sin, cos]] applied with -heading
        cos_h = torch.cos(-sensor_heading)
        sin_h = torch.sin(-sensor_heading)
        x_sensor = cos_h * dx - sin_h * dy  # [B, T]
        y_sensor = sin_h * dx + cos_h * dy  # [B, T]

        # 4. Scale to grid coordinates [-1, 1] where r_max maps to edge
        # In renderer: sensor at center (0,0), r_max maps to 1.0
        scale = 1.0 / r_max
        x_grid = x_sensor * scale  # [-1, 1] if within r_max
        y_grid = y_sensor * scale

        # 5. Convert [-1, 1] to grid indices [0, W-1] and [0, H-1]
        x_idx = ((x_grid + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
        y_idx = ((y_grid + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
    else:
        # World-frame: positions directly map to grid
        # Note: positions are (x, y) where x is horizontal, y is vertical
        x_idx = (true_positions[..., 0] * (W - 1)).long().clamp(0, W - 1)  # [B, T]
        y_idx = (true_positions[..., 1] * (H - 1)).long().clamp(0, H - 1)  # [B, T]

    # Sample probability at true positions using advanced indexing
    # heatmaps: [B, T, 1, H, W] -> squeeze channel -> [B, T, H, W]
    heatmaps_2d = heatmaps.squeeze(2)  # [B, T, H, W]

    # Create batch and time indices for gathering
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, T)
    time_idx = torch.arange(T, device=device).view(1, T).expand(B, T)

    # Sample at (y_idx, x_idx) - note: first dim is row (y), second is col (x)
    probs = heatmaps_2d[batch_idx, time_idx, y_idx, x_idx]  # [B, T]

    # Negative log-likelihood with numerical stability
    nll = -torch.log(probs + 1e-8)  # [B, T]

    # Apply sequence mask and compute mean
    nll_masked = (nll * mask).sum() / (mask.sum() + 1e-8)

    return nll_masked


def compute_loss_with_heatmap(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    heatmaps: Optional[torch.Tensor],
    true_positions: torch.Tensor,
    sensor_positions: torch.Tensor,
    lengths: torch.Tensor,
    train_cfg: TrainingConfig,
    r_max: float = 150.0,
    env_size: float = 200.0,
    egocentric: bool = True,
) -> torch.Tensor:
    """Compute combined loss for polar predictions and spatial heatmaps.

    Args:
        predictions: Model predictions [batch, seq, 2] (distance_norm, bearing_norm)
        targets: Ground truth [batch, seq, 2] (distance_norm, bearing_norm)
        heatmaps: Spatial probability [batch, seq, 1, H, W] or None
        true_positions: Target positions [batch, seq, 2] (x_norm, y_norm) in [0, 1]
        sensor_positions: Sensor positions [batch, seq, 3] (x_norm, y_norm, heading_norm)
        lengths: Sequence lengths [batch]
        train_cfg: Training configuration
        r_max: Max range for distance
        env_size: Environment size in meters
        egocentric: If True, heatmaps are in sensor-centered frame

    Returns:
        Scalar loss tensor
    """
    # Point estimate loss (distance + bearing)
    point_loss = compute_loss(
        predictions, targets, lengths, train_cfg, 
        r_max=r_max,
        trigonometric_processing=train_cfg.trigonometric_processing
    )



    # Heatmap calibration loss (NLL at true position)
    if heatmaps is not None and train_cfg.use_heatmap_loss:
        heatmap_loss = compute_heatmap_nll(
            heatmaps, true_positions, sensor_positions, lengths,
            r_max=r_max, env_size=env_size, egocentric=egocentric
        )
        total_loss = point_loss + train_cfg.heatmap_weight * heatmap_loss
    else:
        total_loss = point_loss

    return total_loss


# =============================================================================
# 5. MODEL CREATION
# =============================================================================

def create_model(cfg: ModelConfig, data_cfg: DataConfig) -> nn.Module:
    """Create SpatialPFNCP model with AutoNCP wiring.

    Architecture:
        Sensory (5) -> Interneurons -> Motor (2) + Spatial Heatmap

    No encoder - observations feed directly to sensory layer.
    Motor neurons output distance and bearing in polar representation.
    Spatial head generates probability heatmaps from particles.
    """
    from pfncps.nn import SpatialPFNCP
    from pfncps.wirings import AutoNCP

    # AutoNCP wiring: automatic sensory -> inter -> motor connectivity
    wiring = AutoNCP(units=cfg.units, output_size=cfg.output_size)
    wiring.build(cfg.input_size)

    # Observation model for particle weight updates (use factory)
    # hidden_size = wiring.units (particle hidden dimension)
    # obs_size = wiring.output_dim (motor output dimension) - set in cfg.output_size
    observation_model = create_observation_model(cfg, hidden_size=wiring.units)

    # Build SpatialPFNCP model (PFNCP with spatial heatmap output)
    model = SpatialPFNCP(
        wiring=wiring,
        input_size=cfg.input_size,
        n_particles=cfg.n_particles,
        # Spatial parameters
        map_size=cfg.map_size,
        r_max=data_cfg.r_max,
        env_size=data_cfg.env_size,
        egocentric=cfg.egocentric,  # Sensor-centered heatmaps
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        projection_hidden=cfg.projection_hidden,
        # PFNCP parameters
        approach=cfg.approach,
        cell_type="cfc",
        mode=cfg.mode,
        noise_type=cfg.noise_type,
        noise_init=cfg.noise_init,
        alpha_mode=cfg.alpha_mode,
        alpha_init=cfg.alpha_init,
        resample_threshold=cfg.resample_threshold,
        observation_model=observation_model,
        return_sequences=True,
    )

    return model



# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_cfg: TrainingConfig,
    r_max: float = 150.0,
    env_size: float = 200.0,
    egocentric: bool = True,
    checkpoint_dir: Optional[Path] = None,
    resume_checkpoint: Optional[str] = None,

    run_id: Optional[str] = None,
    trigonometric_processing: bool = False,

) -> Dict:
    """Train the model with polar (distance, bearing) representation.

    Uses Huber loss for distance and circular loss for bearing.

    Optimizations applied:
    - Vectorized loss computation
    - Vectorized validation metrics
    - Lazy ESS logging (every N batches)
    - Optional torch.compile

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        train_cfg: Training configuration
        r_max: Maximum range for distance normalization
        env_size: Environment size in meters
        egocentric: If True, heatmaps are in sensor-centered frame
        checkpoint_dir: Directory to save checkpoints
        resume_checkpoint: Path to checkpoint to resume from
        run_id: Unique identifier for this training run
        
    Returns:
        Tuple of (history dict, device)
    """
    # Device selection
    if train_cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(train_cfg.device)
    
    print(f"Training on: {device}")
    model = model.to(device)
    
    # Optional torch.compile for speedup (PyTorch 2.0+)
    # NOTE: torch.compile on MPS is an early prototype with dynamic shape issues
    # Only enable on CUDA where it's stable
    can_compile = (
        train_cfg.use_compile 
        and hasattr(torch, 'compile') 
        and device.type == "cuda"
    )
    if can_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  Applied torch.compile (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile failed, using eager mode: {e}")
    elif train_cfg.use_compile and device.type == "mps":
        print("  Skipping torch.compile (MPS prototype is unstable)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Checkpoint tracking
    best_val_loss = float('inf')
    start_epoch = 0
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Checkpoints: {checkpoint_dir}")
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dist_error_m": [],
        "val_bearing_error_rad": [],
        "ess": [],
    }
    
    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f"  Resuming from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])
        history = checkpoint.get('history', history)
        print(f"  ✓ Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    
    for epoch in tqdm(range(start_epoch, train_cfg.epochs), desc="Training", unit="epoch"):
        # ===== Training =====
        model.train()
        train_loss = 0.0
        train_ess = 0.0
        n_batches = 0
        ess_samples = 0

        for inputs, targets, sensor_positions, target_pos, lengths in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            sensor_positions = sensor_positions.to(device)
            target_pos = target_pos.to(device)

            optimizer.zero_grad()

            # Forward pass with SpatialPFNCP
            # Returns (predictions, heatmaps, state)
            # observations=targets enables Bayesian weight updates in the particle filter
            predictions, heatmaps, state = model(
                inputs,
                observations=targets,
                sensor_positions=sensor_positions
            )

            # Compute combined loss (point + heatmap)
            loss = compute_loss_with_heatmap(
                predictions, targets, heatmaps, target_pos, sensor_positions,
                lengths, train_cfg, r_max=r_max, env_size=env_size, egocentric=egocentric
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            # Compute ESS only every N batches (lazy evaluation for efficiency)
            if n_batches % train_cfg.ess_log_interval == 0 and state is not None and len(state) >= 2:
                log_weights = state[1]
                log_w = log_weights - log_weights.max(dim=-1, keepdim=True)[0]
                w = torch.exp(log_w)
                w = w / w.sum(dim=-1, keepdim=True)
                ess = (1.0 / (w ** 2).sum(dim=-1)).mean().item()
                train_ess += ess
                ess_samples += 1

        avg_train_loss = train_loss / n_batches
        avg_ess = train_ess / ess_samples if ess_samples > 0 else 0
        
        # ===== Validation (Polar metrics) =====
        model.eval()
        val_loss = 0.0
        val_dist_error = 0.0
        val_bear_error = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for inputs, targets, sensor_positions, target_pos, lengths in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                sensor_positions = sensor_positions.to(device)
                target_pos = target_pos.to(device)
                batch_size = inputs.size(0)
                max_len = inputs.size(1)

                # Forward pass with SpatialPFNCP
                # observations=targets enables Bayesian weight updates in the particle filter
                predictions, heatmaps, _ = model(
                    inputs,
                    observations=targets,
                    sensor_positions=sensor_positions
                )

                # Compute combined loss (point + heatmap)
                batch_loss = compute_loss_with_heatmap(
                    predictions, targets, heatmaps, target_pos, sensor_positions,
                    lengths, train_cfg, r_max=r_max, env_size=env_size, egocentric=egocentric
                )
                val_loss += batch_loss.item() * batch_size

                # Vectorized validation metrics
                if isinstance(lengths, torch.Tensor):
                    lengths_t = lengths.to(device)
                else:
                    lengths_t = torch.tensor(lengths, device=device)
                range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
                mask = (range_tensor < lengths_t.unsqueeze(1)).float()  # [B, T]
                total_valid = mask.sum().item()

                # Distance error in meters (direct!)
                dist_error_m = torch.abs(predictions[:, :, 0] - targets[:, :, 0]) * r_max  # [B, T]
                val_dist_error += (dist_error_m * mask).sum().item()

                # Bearing error in radians (circular)
                if train_cfg.trigonometric_processing:
                    # Convert sin/cos back to angle [0, 1]
                    pred_sin = predictions[..., 1]
                    pred_cos = predictions[..., 2]
                    gt_sin = targets[..., 1]
                    gt_cos = targets[..., 2]
                    
                    # atan2 returns [-pi, pi]
                    pred_angle = torch.atan2(pred_sin, pred_cos)
                    gt_angle = torch.atan2(gt_sin, gt_cos)
                    
                    # Circular difference in radians
                    angle_diff = torch.atan2(torch.sin(pred_angle - gt_angle), torch.cos(pred_angle - gt_angle))
                    bear_error_rad = torch.abs(angle_diff) # Radians
                else:
                    diff_bear = predictions[:, :, 1] - targets[:, :, 1]
                    diff_bear = torch.remainder(diff_bear + 0.5, 1.0) - 0.5  # Wrap to [-0.5, 0.5]
                    bear_error_rad = torch.abs(diff_bear * 2 * np.pi)  # [B, T]
                val_bear_error += (bear_error_rad * mask).sum().item()

                n_val_samples += total_valid
        
        avg_val_loss = val_loss / (n_val_samples + 1e-8) if n_val_samples > 0 else 0
        avg_dist_error = val_dist_error / (n_val_samples + 1e-8) if n_val_samples > 0 else 0
        avg_bear_error = val_bear_error / (n_val_samples + 1e-8) if n_val_samples > 0 else 0
        
        scheduler.step(avg_val_loss)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dist_error_m"].append(avg_dist_error)
        history["val_bearing_error_rad"].append(avg_bear_error)
        history["ess"].append(avg_ess)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, "
                  f"dist_err={avg_dist_error:.1f}m, bear_err={avg_bear_error:.2f}rad, ESS={avg_ess:.1f}")
        
        # Save checkpoints
        if checkpoint_dir is not None:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'val_dist_error_m': avg_dist_error,
                'val_bearing_error_rad': avg_bear_error,
                'history': history,
                'run_id': run_id,
                'trigonometric_processing': train_cfg.trigonometric_processing,
            }
            
            # Save latest checkpoint
            torch.save(checkpoint, checkpoint_dir / 'latest.pt')
            
            # Save best checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint['best_val_loss'] = best_val_loss  # Update before saving best
                torch.save(checkpoint, checkpoint_dir / 'best.pt')
                print(f"  ✓ New best model (val_loss={avg_val_loss:.4f})")
    
    return history, device


# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def visualize_results(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    history: Dict,
    viz_cfg: VisualizationConfig,
    r_max: float = 150.0,
    device: torch.device = torch.device("cpu"),
):
    """Generate comprehensive visualizations for polar (distance, bearing) output."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    output_dir = Path(viz_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to import PFVisualizer
    try:
        from pfncps.utils.visualization import PFVisualizer
        from pfncps.utils.visualization.collectors import StateCollector
        from pfncps.utils.visualization.plots import (
            plot_ess_timeline,
            plot_weight_distribution,
            plot_particle_trajectories,
            create_dashboard,
        )
        has_visualizer = True
    except ImportError as e:
        has_visualizer = False
        print(f"PFVisualizer not available ({e}), using basic plots")
    
    # Collect validation data - keep per-batch to handle variable lengths
    model.eval()
    batch_data = []  # Store (inputs, targets, sensors, lengths) tuples

    with torch.no_grad():
        for batch in val_loader:
            # Handle both old (4 returns) and new (5 returns with target_pos) formats
            if len(batch) == 5:
                inputs, targets, sensors, target_pos, lengths = batch
            else:
                inputs, targets, sensors, lengths = batch

            inputs_dev = inputs.to(device)
            # SpatialPFNCP returns (outputs, heatmaps, state)
            result = model(inputs_dev)
            if len(result) == 3:
                preds, heatmaps, _ = result
            else:
                preds, _ = result
                heatmaps = None

            batch_data.append({
                'inputs': inputs.cpu(),
                'targets': targets,
                'sensors': sensors,
                'predictions': preds.cpu(),
                'lengths': lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths,
            })
    
    # =========================================================================
    # Plot 1: Training Curves
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history["train_loss"], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(history["val_loss"], 'r--', label='Val', linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history["val_dist_error_m"], 'g-', linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Distance Error (m)")
    axes[0, 1].set_title("Validation Distance Error")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history["val_bearing_error_rad"], 'orange', linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Bearing Error (rad)")
    axes[1, 0].set_title("Validation Bearing Error")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history["ess"], 'purple', linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("ESS")
    axes[1, 1].set_title("Effective Sample Size")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=viz_cfg.dpi)
    plt.close()
    print(f"Saved: {output_dir / 'training_curves.png'}")
    
    # =========================================================================
    # Plot 2: PFVisualizer Diagnostics
    # =========================================================================
    if has_visualizer:
        print("Generating particle filter diagnostics...")
        collector = StateCollector(max_history=500)
        
        # Use first sequence from first batch
        first_batch = batch_data[0]
        sample_inputs = first_batch['inputs'][:1].to(device)
        sample_length = first_batch['lengths'][0]
        
        with torch.no_grad():
            hx = None
            for t in range(sample_length):
                x_t = sample_inputs[:, t:t+1, :]
                
                # Pass observation for weight update!
                obs_t = first_batch['targets'][0, t:t+1, :].to(device)
                _, hx = model.cell(sample_inputs[:, t, :], hx, observation=obs_t)
                
                if hx is not None:
                    particles, log_weights = hx
                    collector.log_step(particles=particles, log_weights=log_weights)
        
        try:
            fig, ax = plot_ess_timeline(collector, threshold=0.5)
            plt.savefig(output_dir / "pf_ess_timeline.png", dpi=viz_cfg.dpi)
            plt.close()
            print(f"Saved: {output_dir / 'pf_ess_timeline.png'}")
            
            fig, ax = plot_weight_distribution(collector, mode="heatmap")
            plt.savefig(output_dir / "pf_weights_heatmap.png", dpi=viz_cfg.dpi)
            plt.close()
            print(f"Saved: {output_dir / 'pf_weights_heatmap.png'}")
            
            fig, ax = plot_particle_trajectories(collector, dims=[0, 1], style="fan", n_particles=8)
            plt.savefig(output_dir / "pf_trajectories.png", dpi=viz_cfg.dpi)
            plt.close()
            print(f"Saved: {output_dir / 'pf_trajectories.png'}")
            
            fig = create_dashboard(collector, which="debug")
            plt.savefig(output_dir / "pf_debug_dashboard.png", dpi=viz_cfg.dpi)
            plt.close()
            print(f"Saved: {output_dir / 'pf_debug_dashboard.png'}")
            
        except Exception as e:
            print(f"PFVisualizer warning: {e}")
    
    # =========================================================================
    # Plot 3: Animated Trajectory Predictions (5 random sequences)
    # Converts polar (distance, bearing) predictions to Cartesian for visualization
    # =========================================================================
    print(f"Generating {viz_cfg.n_animation_sequences} trajectory animations...")
    
    # Build flat index mapping: (batch_idx, seq_idx_in_batch, length)
    flat_indices = []
    for batch_idx, batch in enumerate(batch_data):
        for seq_idx, length in enumerate(batch['lengths']):
            flat_indices.append((batch_idx, seq_idx, length))
    
    n_total = len(flat_indices)
    sample_indices = np.random.choice(n_total, min(viz_cfg.n_animation_sequences, n_total), replace=False)
    
    for anim_idx, flat_idx in enumerate(sample_indices):
        try:
            batch_idx, seq_idx, length = flat_indices[flat_idx]
            batch = batch_data[batch_idx]
            
            # Targets and predictions are polar: [distance_norm, bearing_norm]
            gt_polar = batch['targets'][seq_idx, :length].numpy()      # (T, 2): [dist_norm, bear_norm]
            pred_polar = batch['predictions'][seq_idx, :length].numpy()  # (T, 2): [dist_norm, bear_norm]
            sensor_norm = batch['sensors'][seq_idx, :length].numpy()   # (T, 3): [x_norm, y_norm, heading_norm]
            
            # === CONVERT POLAR TO CARTESIAN FOR VISUALIZATION ===
            
            # Sensor position in meters
            sensor_pos = (sensor_norm[:, :2] - 0.5) * 200.0  # Environment coordinates in meters
            sensor_heading_rad = sensor_norm[:, 2] * 2 * np.pi - np.pi  # [0,1] -> [-π, π]
            
            # Ground truth: polar to Cartesian
            gt_dist_m = gt_polar[:, 0] * r_max  # distance in meters
            
            # Handle trigonometric processing for bearing
            if gt_polar.shape[1] == 3: # [dist, sin, cos]
                gt_bear_rad = np.arctan2(gt_polar[:, 1], gt_polar[:, 2]) # bearing in radians
            else: # [dist, angle_norm]
                gt_bear_rad = gt_polar[:, 1] * 2 * np.pi - np.pi  # bearing in radians
            
            # World bearing = sensor heading + relative bearing
            gt_world_bearing = sensor_heading_rad + gt_bear_rad
            # Target position relative to sensor in world frame
            gt_rel_x = gt_dist_m * np.cos(gt_world_bearing)
            gt_rel_y = gt_dist_m * np.sin(gt_world_bearing)
            target_abs = sensor_pos + np.stack([gt_rel_x, gt_rel_y], axis=-1)
            
            # Prediction: polar to Cartesian
            pred_dist_m = pred_polar[:, 0] * r_max
            
            # Handle trigonometric processing for bearing
            if pred_polar.shape[1] == 3: # [dist, sin, cos]
                pred_bear_rad = np.arctan2(pred_polar[:, 1], pred_polar[:, 2]) # bearing in radians
            else: # [dist, angle_norm]
                pred_bear_rad = pred_polar[:, 1] * 2 * np.pi - np.pi
            
            pred_world_bearing = sensor_heading_rad + pred_bear_rad
            pred_rel_x = pred_dist_m * np.cos(pred_world_bearing)
            pred_rel_y = pred_dist_m * np.sin(pred_world_bearing)
            pred_abs = sensor_pos + np.stack([pred_rel_x, pred_rel_y], axis=-1)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Set up plot limits based on all positions
            all_x = np.concatenate([sensor_pos[:, 0], target_abs[:, 0], pred_abs[:, 0]])
            all_y = np.concatenate([sensor_pos[:, 1], target_abs[:, 1], pred_abs[:, 1]])
            margin = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) * 0.15
            margin = max(margin, 20)
            
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Start markers
            ax.scatter(sensor_pos[0, 0], sensor_pos[0, 1], c='blue', s=200, marker='^', label='Sensor Start', zorder=6)
            ax.scatter(target_abs[0, 0], target_abs[0, 1], c='green', s=150, marker='*', label='Target Start', zorder=6)
            
            # Initialize animated elements
            sensor_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='Sensor Path')
            target_line, = ax.plot([], [], 'r-', linewidth=2, label='Target (GT)')
            pred_line, = ax.plot([], [], 'orange', linewidth=2, linestyle='--', label='Target (Pred)')
            sensor_point = ax.scatter([], [], c='blue', s=150, marker='^', zorder=8)
            target_point = ax.scatter([], [], c='red', s=120, marker='*', zorder=8)
            pred_point = ax.scatter([], [], c='orange', s=80, zorder=7)
            connect_line, = ax.plot([], [], 'gray', linewidth=1, alpha=0.5, linestyle=':')
            
            title = ax.set_title("Polar (dist, bearing) → Cartesian Visualization")
            ax.legend(loc='upper right', fontsize=9)
            
            def init():
                sensor_line.set_data([], [])
                target_line.set_data([], [])
                pred_line.set_data([], [])
                sensor_point.set_offsets(np.empty((0, 2)))
                target_point.set_offsets(np.empty((0, 2)))
                pred_point.set_offsets(np.empty((0, 2)))
                connect_line.set_data([], [])
                return sensor_line, target_line, pred_line, sensor_point, target_point, pred_point, connect_line, title
            
            def update(frame):
                # Paths up to current frame
                sensor_line.set_data(sensor_pos[:frame+1, 0], sensor_pos[:frame+1, 1])
                target_line.set_data(target_abs[:frame+1, 0], target_abs[:frame+1, 1])
                pred_line.set_data(pred_abs[:frame+1, 0], pred_abs[:frame+1, 1])
                
                # Current positions
                sensor_point.set_offsets([[sensor_pos[frame, 0], sensor_pos[frame, 1]]])
                target_point.set_offsets([[target_abs[frame, 0], target_abs[frame, 1]]])
                pred_point.set_offsets([[pred_abs[frame, 0], pred_abs[frame, 1]]])
                
                # Line from sensor to GT target
                connect_line.set_data(
                    [sensor_pos[frame, 0], target_abs[frame, 0]],
                    [sensor_pos[frame, 1], target_abs[frame, 1]]
                )
                
                # Compute metrics (polar errors)
                dist_err_m = abs(gt_polar[frame, 0] - pred_polar[frame, 0]) * r_max
                
                # Bearing error calculation depends on output format
                if gt_polar.shape[1] == 3: # [dist, sin, cos]
                    gt_bear_rad_frame = np.arctan2(gt_polar[frame, 1], gt_polar[frame, 2])
                    pred_bear_rad_frame = np.arctan2(pred_polar[frame, 1], pred_polar[frame, 2])
                    bear_diff = gt_bear_rad_frame - pred_bear_rad_frame
                    bear_err_rad = np.abs(np.arctan2(np.sin(bear_diff), np.cos(bear_diff)))
                else: # [dist, angle_norm]
                    bear_diff = gt_polar[frame, 1] - pred_polar[frame, 1]
                    bear_diff = (bear_diff + 0.5) % 1.0 - 0.5  # Wrap to [-0.5, 0.5]
                    bear_err_rad = abs(bear_diff) * 2 * np.pi
                
                title.set_text(f"t={frame+1}/{length} | DistErr: {dist_err_m:.1f}m | BearErr: {bear_err_rad:.2f}rad")
                
                return sensor_line, target_line, pred_line, sensor_point, target_point, pred_point, connect_line, title
            
            anim = animation.FuncAnimation(
                fig, update, init_func=init,
                frames=length, interval=1000 // viz_cfg.animation_fps, blit=True
            )
            
            anim.save(output_dir / f"trajectory_{anim_idx+1}.gif", writer='pillow', fps=viz_cfg.animation_fps)
            plt.close()
            print(f"Saved: {output_dir / f'trajectory_{anim_idx+1}.gif'}")
            
        except Exception as e:
            print(f"Animation {anim_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir}")


def run_debug_visualization(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    viz_cfg: VisualizationConfig,
    train_cfg: TrainingConfig,
    r_max: float = 150.0,
    env_size: float = 200.0,
    egocentric: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """Run comprehensive debug visualization with multi-panel dashboard.

    Uses SpatialPFNCP to generate spatial heatmaps from particle distributions,
    showing uncertainty quantification and latent space dynamics.

    Generates per-episode GIF animations showing:
    - Environment map with particle uncertainty heatmap overlay
    - Spatial probability distribution from particles (egocentric or world-frame)
    - Particle latent space (PCA projection)
    - Weight distribution and ESS health
    - Polar coordinate comparison (distance, bearing)
    - Error metrics over time
    """
    from rssi_debug_visualizer import RSSIDebugVisualizer, RSSIDebugConfig

    output_dir = Path(viz_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {viz_cfg.n_debug_episodes} debug episode visualizations...")
    print(f"  Using SpatialPFNCP with heatmap generation")

    # Collect per-batch data to handle variable lengths
    model.eval()
    batch_data = []

    with torch.no_grad():
        for batch in val_loader:
            # Handle both old (4 returns) and new (5 returns with target_pos) formats
            if len(batch) == 5:
                inputs, targets, sensors, target_pos, lengths = batch
                batch_data.append({
                    'inputs': inputs,
                    'targets': targets,
                    'sensors': sensors,
                    'target_pos': target_pos,
                    'lengths': lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths,
                })
            else:
                inputs, targets, sensors, lengths = batch
                batch_data.append({
                    'inputs': inputs,
                    'targets': targets,
                    'sensors': sensors,
                    'target_pos': None,
                    'lengths': lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths,
                })

    # Build flat index mapping: (batch_idx, seq_idx_in_batch, length)
    flat_indices = []
    for batch_idx, batch in enumerate(batch_data):
        for seq_idx, length in enumerate(batch['lengths']):
            flat_indices.append((batch_idx, seq_idx, length))

    n_total = len(flat_indices)
    n_episodes = min(viz_cfg.n_debug_episodes, n_total)
    sample_indices = np.random.choice(n_total, n_episodes, replace=False)

    # Get model info
    n_particles = getattr(model, 'n_particles', 32)
    hidden_size = getattr(model, 'hidden_size', 64)
    map_size = getattr(model, 'map_size', 64)

    # Check if model has spatial components
    has_spatial = hasattr(model, 'projection_head') and hasattr(model, 'renderer')

    debug_config = RSSIDebugConfig(
        figsize=(24, 16),
        dpi=viz_cfg.dpi,
        frame_duration_ms=100,
    )

    for ep_idx, flat_idx in enumerate(sample_indices):
        try:
            batch_idx, seq_idx, length = flat_indices[flat_idx]
            batch = batch_data[batch_idx]

            # Extract this sequence
            inputs = batch['inputs'][seq_idx:seq_idx+1, :length, :].to(device)
            targets = batch['targets'][seq_idx, :length, :]
            sensors = batch['sensors'][seq_idx, :length, :]
            sensor_pos_batch = sensors.unsqueeze(0).to(device)

            # Create visualizer for this episode
            visualizer = RSSIDebugVisualizer(
                save_dir=str(output_dir),
                env_size=env_size,
                r_max=r_max,
                n_particles=n_particles,
                hidden_size=hidden_size,
                config=debug_config,
                episode_prefix="debug_episode",
                egocentric=egocentric,
            )
            visualizer.start_episode(ep_idx + 1)

            # Step through the sequence with spatial heatmap generation
            state = None
            with torch.no_grad():
                # Store all outputs, heatmaps, and states for this sequence
                all_outputs = []
                all_heatmaps = []
                all_states = []
                all_projected_positions = []
                all_projected_sigmas = []

                for t in range(length):
                    x_t = inputs[:, t:t+1, :]  # Keep seq dim for cell
                    sensor_t = sensor_pos_batch[:, t:t+1, :]
                    
                    # Pass observation for weight update!
                    obs_t = targets[t:t+1, :].to(device)

                    # Forward through cell
                    output, state = model.pfncp.cell(inputs[:, t, :], state, observation=obs_t)

                    # Extract particles and weights
                    particles, log_weights = state

                    # Generate spatial heatmap if model has spatial components
                    heatmap = None
                    projected_positions = None
                    projected_sigmas = None

                    if has_spatial:
                        # Get projected positions and sigmas from projection head
                        positions, sigmas, weights = model.projection_head(
                            particles, log_weights
                        )
                        projected_positions = positions
                        projected_sigmas = sigmas

                        # Render heatmap
                        heatmap = model.renderer(
                            positions, sigmas, weights, sensor_pos_batch[:, t, :]
                        )
                    
                    all_outputs.append(output.cpu())
                    all_heatmaps.append(heatmap.cpu() if heatmap is not None else None)
                    all_projected_positions.append(projected_positions.cpu() if projected_positions is not None else None)
                    all_projected_sigmas.append(projected_sigmas.cpu() if projected_sigmas is not None else None)
                    all_states.append(state) # Keep on device for now

                # Concatenate results for the whole sequence
                outputs_seq = torch.cat(all_outputs, dim=0) # (T, output_size)
                heatmaps_seq = torch.cat(all_heatmaps, dim=0) if all_heatmaps[0] is not None else None # (T, 1, H, W)
                
                # Convert states to (T, N_particles, hidden_size) and (T, N_particles)
                # Assuming state is (particles, log_weights)
                final_particles = torch.cat([s[0] for s in all_states], dim=0).cpu() # (T, N_particles, hidden_size)
                final_log_weights = torch.cat([s[1] for s in all_states], dim=0).cpu() # (T, N_particles)
                final_state = (final_particles, final_log_weights)

            # Check if using trigonometric output
            if train_cfg.trigonometric_processing:
                # Convert predictions [dist, sin, cos] -> [dist, bear_norm]
                # for visualization compatibility
                dist = outputs_seq[..., 0]
                sin_b = outputs_seq[..., 1]
                cos_b = outputs_seq[..., 2]
                angle_rad = torch.atan2(sin_b, cos_b)
                bear_norm = (angle_rad + np.pi) / (2 * np.pi)
                outputs_viz = torch.stack([dist, bear_norm], dim=-1)
                
                # Also convert targets
                t_dist = targets[..., 0]
                t_sin = targets[..., 1]
                t_cos = targets[..., 2]
                t_angle = torch.atan2(t_sin, t_cos)
                t_bear_norm = (t_angle + np.pi) / (2 * np.pi)
                targets_viz = torch.stack([t_dist, t_bear_norm], dim=-1)
                
            else:
                outputs_viz = outputs_seq
                targets_viz = targets

            # Use output_viz for visualizer
            for t in range(length):
                # Collect step data
                step_data = visualizer.collect_step(
                    step=t,
                    inputs=inputs[:, t, :].cpu(), # inputs are already on device, move to cpu for viz
                    targets=targets_viz[t:t+1, :],
                    predictions=outputs_viz[t:t+1, :],
                    state=(
                        final_state[0][t:t+1, :, :] if final_state else None,
                        final_state[1][t:t+1, :] if final_state else None
                    ),
                    sensor_positions=sensors[t:t+1, :],
                    heatmap=heatmaps_seq[t:t+1, :, :, :] if heatmaps_seq is not None else None,
                    projected_positions=all_projected_positions[t],
                    projected_sigmas=all_projected_sigmas[t],
                    batch_idx=0,
                )
                visualizer.add_step(step_data)

            # Save outputs
            gif_path = visualizer.save_episode()
            summary_path = visualizer.save_summary()

            final_err = visualizer.pos_error_history[-1] if visualizer.pos_error_history else 0
            mean_ess = np.mean(visualizer.ess_history) if visualizer.ess_history else 0

            print(f"  Episode {ep_idx + 1}: {length} steps, "
                  f"final_err={final_err:.1f}m, mean_ess={mean_ess:.1f}")

        except Exception as e:
            print(f"  Episode {ep_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"Debug visualizations saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete RSSI Particle Filter training pipeline.
    
    Supports:
    - Config-first approach with YAML model config file
    - Unique run IDs for checkpoint management
    - Resume training from checkpoints
    """
    parser = argparse.ArgumentParser(
        description="RSSI Particle Filter Training with Config-First Architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # ----- Core Arguments -----
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config YAML file (e.g., configs/default.yaml)")
    parser.add_argument("--data_dir", type=str, 
                        default="examples/RSSI/data/dpf_dataset_polar_double_single_target",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="examples/RSSI/output",
                        help="Output directory for models and visualizations")
    
    # ----- Resume Training -----
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., output/models/abc123/latest.pt)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Override run ID (default: auto-generated)")
    
    # ----- CLI Overrides (take precedence over config file) -----
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--n_particles", type=int, default=None,
                        help="Override number of particles from config")
    parser.add_argument("--units", type=int, default=None,
                        help="Override hidden units from config")
    
    # ----- Debug Visualization Arguments -----
    parser.add_argument("--debug_viz", action="store_true",
                        help="Enable comprehensive debug visualization")
    parser.add_argument("--n_debug_episodes", type=int, default=3,
                        help="Number of episodes for debug visualization")
    
    # ----- Performance Arguments -----
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (avoids warmup latency)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RSSI Particle Filter Training (PFNCP + AutoNCP)")
    print("=" * 60)
    
    # Load model config from YAML
    print(f"\nLoading config: {args.config}")
    config = load_model_config(args.config)
    
    # Determine run_id
    if args.run_id:
        run_id = args.run_id
    elif args.resume:
        # Extract run_id from resume path (e.g., output/models/{run_id}/latest.pt)
        run_id = Path(args.resume).parent.name
        print(f"  Resuming run: {run_id}")
    else:
        exp_name = config.get('experiment', {}).get('name', 'pfncp')
        run_id = generate_run_id(exp_name)
        print(f"  New run ID: {run_id}")
    
    # Set up checkpoint directory with run_id
    checkpoint_dir = Path(args.output_dir) / "models" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to checkpoint directory (only for new runs)
    if not args.resume:
        save_config(config, checkpoint_dir / "config.yaml")
        print(f"  Config saved: {checkpoint_dir / 'config.yaml'}")
    
    # Convert config to dataclasses
    data_cfg, model_cfg, train_cfg, viz_cfg = config_to_dataclasses(
        config, args.data_dir, args.output_dir
    )
    
    # Apply CLI overrides
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.n_particles is not None:
        model_cfg.n_particles = args.n_particles
    if args.units is not None:
        model_cfg.units = args.units
    if args.no_compile:
        train_cfg.use_compile = False
    if args.debug_viz:
        viz_cfg.debug_viz = True
    if args.n_debug_episodes:
        viz_cfg.n_debug_episodes = args.n_debug_episodes
    
    print(f"\nConfiguration:")
    print(f"  Run ID: {run_id}")
    print(f"  Data: {data_cfg.data_dir}")
    print(f"  Output: Polar (distance, bearing) - r_max={data_cfg.r_max}m")
    print(f"  Approach: {model_cfg.approach}")
    print(f"  Particles: {model_cfg.n_particles}")
    print(f"  Units: {model_cfg.units}")
    print(f"  Obs Model: {model_cfg.obs_model}")
    print(f"  Loss Type: {train_cfg.loss_type}")
    print(f"  Circular Bearing: {train_cfg.use_circular_bearing}")
    print(f"  Trigonometric Processing: {train_cfg.trigonometric_processing}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Checkpoints: {checkpoint_dir}")
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    train_dataset, val_dataset = load_data(args.data_dir, data_cfg)
    
    # Determine DataLoader optimization settings
    # pin_memory only benefits CUDA, not MPS or CPU
    use_pin_memory = torch.cuda.is_available()
    num_workers = train_cfg.num_workers
    persistent_workers = num_workers > 0
    
    print(f"  DataLoader: num_workers={num_workers}, pin_memory={use_pin_memory}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
    )
    
    # 2. Create model
    print(f"\n[2/4] Creating SpatialPFNCP model (approach={model_cfg.approach}, obs={model_cfg.obs_model})...")
    model = create_model(model_cfg, data_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Wiring: AutoNCP(units={model_cfg.units}, output_size={model_cfg.output_size})")
    print(f"  Heatmap: {model_cfg.map_size}x{model_cfg.map_size}, loss_weight={train_cfg.heatmap_weight}")
    
    # 3. Train (with optional resume)
    print("\n[3/4] Training...")
    history, device = train(
        model, train_loader, val_loader, train_cfg,
        r_max=data_cfg.r_max,
        env_size=data_cfg.env_size,
        egocentric=model_cfg.egocentric,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=args.resume,
        run_id=run_id,
    )
    
    # 4. Visualize
    print("\n[4/4] Generating visualizations...")
    visualize_results(model, val_loader, history, viz_cfg, data_cfg.r_max, device)
    
    # 4b. Debug visualization (if enabled)
    if viz_cfg.debug_viz:
        print("\n[4b] Generating debug visualizations...")
        run_debug_visualization(
            model, val_loader, viz_cfg, train_cfg,
            r_max=data_cfg.r_max,
            env_size=data_cfg.env_size, 
            egocentric=model_cfg.egocentric,
            device=device
        )
    
    # Final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Run ID: {run_id}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final Distance Error: {history['val_dist_error_m'][-1]:.1f} meters")
    print(f"  Final Bearing Error: {history['val_bearing_error_rad'][-1]:.2f} radians")
    print(f"  Final ESS: {history['ess'][-1]:.1f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

