#!/usr/bin/env python3
"""Run inference and debug visualization for trained RSSI models.

Usage:
    python examples/RSSI/inference_rssi.py \
        --model_dir examples/RSSI/output/models/run_v2 \
        --n_episodes 5
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
# Assuming script is run from project root (where poetry run usually happens)
sys.path.append(os.getcwd())

# Add examples/RSSI to path to import train_rssi
sys.path.append(os.path.join(os.getcwd(), "examples", "RSSI"))

from train_rssi import (
    load_model_config,
    config_to_dataclasses,
    create_model,
    load_data,
    run_debug_visualization,
    collate_fn
)


def main():
    parser = argparse.ArgumentParser(description="Run inference for RSSI Particle Filter")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model directory (containing config.yaml)")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint filename (default: best.pt)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory (optional)")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Number of debug episodes to generate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on (cpu, cuda, mps, auto)")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.yaml"
    checkpoint_path = model_dir / args.checkpoint
    output_dir = model_dir / "inference_output"
    
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
        
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading model from: {model_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # 1. Load configuration
    raw_config = load_model_config(str(config_path))
    
    # Override data dir if provided
    data_dir = args.data_dir if args.data_dir else raw_config.get('data', {}).get('data_dir', raw_config['training']['data_dir'] if 'data_dir' in raw_config['training'] else "examples/RSSI/data/dpf_dataset_polar_double_single_target")
    
    # Reconstruct dataclasses
    data_cfg, model_cfg, train_cfg, viz_cfg = config_to_dataclasses(
        raw_config, 
        data_dir=data_dir,
        output_dir=str(output_dir)
    )
    
    # Override viz config
    viz_cfg.n_debug_episodes = args.n_episodes
    viz_cfg.debug_viz = True
    
    # 2. Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Running on: {device}")
    
    # 3. Load data
    print(f"Loading data from: {data_cfg.data_dir}")
    _, val_dataset = load_data(data_cfg.data_dir, data_cfg)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Process one by one for consistent visualization
        shuffle=True,  # Random episodes
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # 4. Create model and load weights
    print(f"Creating model...")
    model = create_model(model_cfg, data_cfg)
    model.to(device)
    
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 5. Run visualization
    print(f"Generating visualizations for {args.n_episodes} episodes...")
    run_debug_visualization(
        model, 
        val_loader, 
        viz_cfg,
        train_cfg,
        r_max=data_cfg.r_max,
        env_size=data_cfg.env_size,
        egocentric=model_cfg.egocentric,
        device=device,
    )
    
    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
