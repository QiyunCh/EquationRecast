#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train_Server.py - Multi-GPU DDP FNO training script for plasma temperature prediction.

Key features:
  1) Uses rdzv_backend=c10d for robust multi-node rendezvous (torchrun)
  2) Proper DDP initialization with device_id to avoid warnings
  3) weight_decay = 1e-5, lr = 1e-3
  4) LR decay: ×0.5 every 1000 epochs
  5) Masked MSE loss for handling invalid regions
  6) Optimized multi-threading configuration

Usage (launched via sbatch, see Train.sbatch):
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
             --rdzv_endpoint=<master>:29500 Train_Server.py
"""

import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data import (
    load_stats,
    FNODataset,
    compute_train_val_split,
)
from Model import SingleFNO


# =============================================================================
# Config (edit here)
# =============================================================================

CONFIG = {
    # Data - MUST match Data.py file names
    "data_h5": "Data_ML_Merged_No_t000_t001.h5",
    "stats_json": "stats_train_Median.json",
    "batch_size": 16,           # per-GPU batch size
    "val_frac": 0.20,           # MUST be 1 - TRAIN_FRAC from Data.py
    "num_workers": 4,           # DataLoader workers per process

    # Model
    "modes1": 32,
    "modes2": 32,
    "width": 128,
    "padding": 16,              # match Model.py default

    # Optimization
    "epochs": 5000,
    "lr": 1e-3,
    "weight_decay": 1e-5,       # Changed from 1e-4 to 1e-5
    "lr_step_size": 1000,       # Decay LR every 1000 epochs
    "lr_gamma": 0.5,            # Multiply by 0.5

    # Reproducibility
    "model_seed": 12345,

    # I/O
    "output_dir": "runs/fno_ddp_run",
    "resume_checkpoint": None,  # e.g. "runs/.../checkpoints/ckpt_ep01000.pt"
}

# MUST match Data.py convention (SHUFFLE_SEED = 816)
DATA_SPLIT_SEED = 816

# Numerical stability
EPS = 1e-12


# =============================================================================
# DDP Setup - Using torchrun environment variables
# =============================================================================

def setup_ddp():
    """
    Initialize DDP from torchrun environment variables.
    
    torchrun sets: RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    Using rdzv_backend=c10d, the rendezvous is handled automatically.
    
    Returns:
        rank, world_size, local_rank, device
    """
    # Verify we're running under torchrun
    if "RANK" not in os.environ:
        raise RuntimeError(
            "This script must be launched with torchrun. "
            "Example: torchrun --nnodes=2 --nproc_per_node=4 "
            "--rdzv_backend=c10d --rdzv_endpoint=<master>:29500 Train_Server.py"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set CUDA device before initializing process group
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize process group with explicit device_id to avoid warnings
    # Using "env://" init_method works with torchrun's environment variables
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    # Verify initialization
    if rank == 0:
        master_addr = os.environ.get("MASTER_ADDR", "unknown")
        master_port = os.environ.get("MASTER_PORT", "unknown")
        print(f"[DDP] Initialized: world_size={world_size}, master={master_addr}:{master_port}")

    # Print per-rank info for debugging
    hostname = os.uname().nodename
    gpu_name = torch.cuda.get_device_name(local_rank)
    print(f"[rank {rank}] host={hostname} local_rank={local_rank} cuda={gpu_name}")

    # Synchronize all processes before proceeding
    dist.barrier()

    return rank, world_size, local_rank, device


def cleanup_ddp():
    """
    Clean up DDP process group.
    
    Includes a barrier and small sleep to reduce "connection closed" warnings
    that can occur when some ranks exit before others.
    """
    if dist.is_initialized():
        dist.barrier()
        time.sleep(1)  # Allow all ranks to sync before cleanup
        dist.destroy_process_group()


# =============================================================================
# DDP Data Loader Factory
# =============================================================================

def make_ddp_dataloaders(
    data_h5_path: Path,
    stats_json_path: Path,
    batch_size: int,
    val_frac: float,
    seed: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    """
    Create DDP-compatible DataLoaders with DistributedSampler.
    
    Returns:
        train_loader, val_loader, train_sampler, val_sampler, stats
    """
    import h5py
    
    stats = load_stats(stats_json_path)
    train_frac = 1.0 - val_frac
    
    # Get dataset size
    with h5py.File(data_h5_path, "r") as f:
        n_total = f["IC"].shape[0]
        H, W = f["IC"].shape[1], f["IC"].shape[2]
    
    # Split indices (must match Data.py convention)
    train_idx, val_idx = compute_train_val_split(n_total, train_frac, seed)
    
    if rank == 0:
        print(f"[Data] {data_h5_path.name}: {n_total} samples, {H}x{W}")
        print(f"[Data] Split: {len(train_idx)} train, {len(val_idx)} val (seed={seed})")
    
    # Create Datasets
    train_ds = FNODataset(data_h5_path, stats, train_idx)
    val_ds = FNODataset(data_h5_path, stats, val_idx)
    
    # Create DistributedSamplers
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    
    # Common DataLoader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    
    val_loader = DataLoader(
        val_ds,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    
    return train_loader, val_loader, train_sampler, val_sampler, stats


# =============================================================================
# Checkpoint utilities
# =============================================================================

def load_checkpoint(model, ckpt_path, device, rank):
    """Load model state dict if resume path is provided."""
    if ckpt_path is None:
        return False, 1, float("inf")

    if rank == 0:
        print(f"[Resume] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    return True, start_epoch, best_val


def load_optimizer_scheduler_state(ckpt_path, optimizer, scheduler, device, rank):
    """Load optimizer, scheduler and histories from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    train_hist = ckpt.get("train_hist", [])
    val_hist = ckpt.get("val_hist", [])

    if rank == 0:
        print(f"[Resume] Loaded optimizer/scheduler state. History length: {len(train_hist)}")

    return train_hist, val_hist


# =============================================================================
# Loss helpers
# =============================================================================

def masked_mse_num_den(pred, target, mask):
    """
    Return (numerator, denominator) for masked MSE.
    
    Shapes:
      pred:   (B, C, H, W)
      target: (B, C, H, W) or broadcastable
      mask:   (B, 1, H, W) or broadcastable
    """
    mask = mask.to(pred.dtype)
    diff_sq = (pred - target) ** 2
    num = torch.sum(mask * diff_sq)
    den = torch.sum(mask) + EPS
    return num, den


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = CONFIG

    # Initialize DDP
    rank, world_size, local_rank, device = setup_ddp()

    if rank == 0:
        print("=" * 60)
        print("Multi-GPU DDP Training")
        print(f"World size: {world_size}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print("=" * 60)

        # Setup output directories (rank 0 only)
        os.makedirs(cfg["output_dir"], exist_ok=True)
        os.makedirs(f"{cfg['output_dir']}/checkpoints", exist_ok=True)

    dist.barrier()

    # Seeds for reproducibility
    model_seed = cfg["model_seed"]
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    np.random.seed(model_seed)

    # Reproducibility flags (may reduce performance slightly)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


    if rank == 0:
        print(f"Data split seed: {DATA_SPLIT_SEED}")
        print(f"Model init seed: {model_seed}")
        print(f"Device: {device}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(device)}")

    # Data (DDP loaders)
    train_loader, val_loader_ddp, train_sampler, val_sampler, stats = make_ddp_dataloaders(
        data_h5_path=Path(cfg["data_h5"]),
        stats_json_path=Path(cfg["stats_json"]),
        batch_size=cfg["batch_size"],
        val_frac=cfg["val_frac"],
        seed=DATA_SPLIT_SEED,
        num_workers=cfg["num_workers"],
        rank=rank,
        world_size=world_size,
    )

    # Rank-0 full validation loader (no DistributedSampler, no padding/repeats)
    val_loader_rank0 = None
    if rank == 0:
        loader_kwargs = {
            "batch_size": cfg["batch_size"],
            "num_workers": cfg["num_workers"],
            "pin_memory": True,
        }
        if cfg["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        val_loader_rank0 = DataLoader(
            val_loader_ddp.dataset,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        print(f"Train: {len(train_loader)} batches/GPU")
        print(f"Val (rank0 full): {len(val_loader_rank0)} batches")
        print(f"Effective batch size: {cfg['batch_size'] * world_size}")
        print(f"Normalization scales: T0={stats['scales']['T0_fixed']:.2e}, "
              f"q0={stats['scales']['q0']:.2e}, y0={stats['scales']['y0']:.2e}")

    dist.barrier()

    # Model
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)

    model = SingleFNO(
        modes1=cfg["modes1"],
        modes2=cfg["modes2"],
        width=cfg["width"],
        padding=cfg["padding"],
    ).to(device)

    dist.barrier()

    # Resume checkpoint before wrapping with DDP
    checkpoint_loaded, start_epoch, best_val = load_checkpoint(
        model,
        cfg["resume_checkpoint"],
        device,
        rank,
    )

    dist.barrier()

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {n_params:,}")
        print("\nTraining config:")
        print(f"  - Epochs: {start_epoch} to {cfg['epochs']}")
        print(f"  - Initial LR: {cfg['lr']}")
        print(f"  - Weight decay: {cfg['weight_decay']}")
        print(f"  - LR decay: ×{cfg['lr_gamma']} every {cfg['lr_step_size']} epochs")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    
    # StepLR: decay every lr_step_size epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["lr_step_size"],
        gamma=cfg["lr_gamma"],
    )

    # Load optimizer/scheduler state and history if resuming
    train_hist, val_hist = [], []
    if checkpoint_loaded:
        train_hist, val_hist = load_optimizer_scheduler_state(
            cfg["resume_checkpoint"],
            optimizer,
            scheduler,
            device,
            rank,
        )

    dist.barrier()

    # Training loop
    t0 = time.time()
    if rank == 0:
        print(f"\nTraining from epoch {start_epoch} to {cfg['epochs']}...")
        print(f"LR schedule: {cfg['lr']} × {cfg['lr_gamma']} every {cfg['lr_step_size']} epochs\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # Set epoch for proper shuffling with DistributedSampler
        train_sampler.set_epoch(epoch)

        # -----------------------------
        # Train
        # -----------------------------
        model.train()

        train_num = torch.tensor(0.0, device=device)
        train_den = torch.tensor(0.0, device=device)

        for x, y, mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)

            num, den = masked_mse_num_den(pred, y, mask)
            loss = num / den

            loss.backward()
            optimizer.step()

            train_num += num.detach()
            train_den += den.detach()

        # Global train loss across all ranks
        dist.all_reduce(train_num, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_den, op=dist.ReduceOp.SUM)
        train_loss = (train_num / train_den).item()

        # -----------------------------
        # Validate (rank 0 only, full)
        # -----------------------------
        val_loss_tensor = torch.tensor(0.0, device=device)

        if rank == 0:
            model.eval()
            val_num = torch.tensor(0.0, device=device)
            val_den = torch.tensor(0.0, device=device)

            with torch.no_grad():
                for x, y, mask in val_loader_rank0:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    pred = model(x)
                    num, den = masked_mse_num_den(pred, y, mask)
                    val_num += num
                    val_den += den

            val_loss = (val_num / val_den).item()
            val_loss_tensor.fill_(val_loss)

        # Broadcast val loss to all ranks
        dist.broadcast(val_loss_tensor, src=0)
        val_loss = float(val_loss_tensor.item())

        # Keep histories aligned
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        # Save checkpoints (rank 0 only)
        if rank == 0:
            # Save best model
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.module.state_dict(), f"{cfg['output_dir']}/best_model.pt")

            # Save checkpoint every 1000 epochs and at final epoch
            if epoch % cfg["lr_step_size"] == 0 or epoch == cfg["epochs"]:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val": best_val,
                        "train_hist": train_hist,
                        "val_hist": val_hist,
                        "config": cfg,
                        "data_split_seed": DATA_SPLIT_SEED,
                    },
                    f"{cfg['output_dir']}/checkpoints/ckpt_ep{epoch:05d}.pt",
                )

        # Step scheduler
        scheduler.step()

        # Log (rank 0 only)
        if rank == 0 and (epoch % 50 == 0 or epoch == 1 or epoch == start_epoch):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"Epoch {epoch:5d} | Train: {train_loss:.6e} | Val: {val_loss:.6e} | "
                  f"LR: {lr:.1e} | Best: {best_val:.6e} | Time: {elapsed/60:.1f}min")

        dist.barrier()

    # -----------------------------
    # Finish (rank 0 only)
    # -----------------------------
    if rank == 0:
        total_time = time.time() - t0
        print(f"\nDone in {total_time/60:.1f} min")
        print(f"Best val loss: {best_val:.6e}")

        history_file = f"{cfg['output_dir']}/history.json"
        with open(history_file, "w") as f:
            json.dump(
                {
                    "train": train_hist,
                    "val": val_hist,
                    "config": cfg,
                    "data_split_seed": DATA_SPLIT_SEED,
                },
                f,
                indent=2,
            )
        print(f"Saved: {history_file}")

        # Plot loss curve
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(train_hist) + 1)
        plt.plot(epochs_range, train_hist, label="Train", alpha=0.8)
        plt.plot(epochs_range, val_hist, label="Val", alpha=0.8)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Masked MSE Loss")
        plt.title(f"Training Loss (LR={cfg['lr']}, decay ×{cfg['lr_gamma']} every {cfg['lr_step_size']})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Mark LR decay points
        for step in range(cfg["lr_step_size"], cfg["epochs"] + 1, cfg["lr_step_size"]):
            if step <= len(train_hist):
                plt.axvline(x=step, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_file = f"{cfg['output_dir']}/loss_curve.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved: {plot_file}")

    # Clean shutdown
    print(f"[rank {rank}] Training loop finished cleanly.")
    cleanup_ddp()


if __name__ == "__main__":
    main()
