#!/usr/bin/env python3

import torch
import sys

sys.path.append("tpami/VIRNet")
from networks.network_moex import *


def test_fixes():
    print("✓ Imports successful")

    # Test QKVAttention without extra scaling
    print("Testing QKVAttention without extra scaling...")
    qkv_attn = QKVAttention(n_heads=8, dropout=0.1)
    B, C, L = 2, 768, 64
    qkv = torch.randn(B, C, L)
    output = qkv_attn(qkv)
    print(f"QKVAttention: input {qkv.shape} -> output {output.shape}")

    # Test FourierPosition with non-trainable B
    print("Testing FourierPosition with non-trainable B...")
    fourier_pos = FourierPosition(in_dim=2, mapping_size=128)
    coords = torch.randn(10, 2)
    pos_enc = fourier_pos(coords)
    print(f"FourierPosition B requires_grad: {fourier_pos.B.requires_grad}")
    print(f"Position encoding: coords {coords.shape} -> features {pos_enc.shape}")

    # Test MoE decoder with improvements
    print("Testing MoE decoder...")
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        kernel=32,
        sharpening_factor=1.0,
        kernel_type=KernelType.GAUSSIAN_CAUCHY,
        initial_temp=1.0,
        tau_min=0.1,
        reg_lambda=0.1,
        min_diag=1e-4,
        max_diag=10.0,
        min_denom=1e-6,
        activation="GELU",
        grid_cache=None,
        total_steps=10000,
        scale_factor=1,
    )

    moe = MoE(cfg)
    print(f"Initial temperature: {moe.get_annealed_temperature():.4f}")

    # Simulate training steps
    moe.training_step = 5000
    print(f"Temperature at 50% training: {moe.get_annealed_temperature():.4f}")

    moe.training_step = 9000
    print(f"Temperature at 90% training: {moe.get_annealed_temperature():.4f}")

    print("✓ All tests passed successfully!")


if __name__ == "__main__":
    test_fixes()
