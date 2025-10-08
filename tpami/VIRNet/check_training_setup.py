#!/usr/bin/env python3
"""
Pre-training validation script for N-SMoE
Checks configuration, data paths, and model initialization
"""

import os
import sys
import torch
import commentjson as json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from networks.network_moex import (
    EncoderConfig,
    MoEConfig,
    AutoencoderConfig,
    Autoencoder,
    KernelType,
)


def check_cuda():
    """Check CUDA availability"""
    print("=" * 60)
    print("CUDA Check")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        return num_gpus
    else:
        print("WARNING: No CUDA devices found!")
        return 0
    print()


def check_data_paths(config_path):
    """Check if training/validation data exists"""
    print("=" * 60)
    print("Data Paths Check")
    print("=" * 60)

    with open(config_path, "r") as f:
        config = json.load(f)

    train_path = config.get("train_hr_patchs", "")
    val_path = config.get("val_hr_path", "")

    print(f"Training data: {train_path}")
    if os.path.exists(train_path):
        num_files = len([f for f in os.listdir(train_path) if f.endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        print(f"  ✓ Exists ({num_files} images)")
    else:
        print(f"  ✗ NOT FOUND!")

    print(f"Validation data: {val_path}")
    if os.path.exists(val_path):
        num_files = len([f for f in os.listdir(val_path) if f.endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        print(f"  ✓ Exists ({num_files} images)")
    else:
        print(f"  ✗ NOT FOUND!")
    print()


def check_model_init(config_path):
    """Check model initialization"""
    print("=" * 60)
    print("Model Initialization Check")
    print("=" * 60)

    with open(config_path, "r") as f:
        args = json.load(f)

    # Build encoder config
    encoder_cfg = EncoderConfig(
        sigma_chn=args.get("sigma_chn", 1),
        kernel_chn=args.get("kernel_chn", 3),
        noise_cond=args.get("noise_cond", "True") == "True",
        kernel_cond=args.get("kernel_cond", "True") == "True",
        noise_avg=args.get("noise_avg", "True") == "True",
        model_channels=args.get("model_channels", 32),
        num_res_blocks=args.get("num_res_blocks", 2),
        attention_resolutions=args.get("attention_resolutions", [16, 8, 4]),
        dropout=args.get("dropout", 0.1),
        channel_mult=tuple(args.get("channel_mult", [2, 4, 8])),
        conv_resample=args.get("conv_resample", "False") == "True",
        dims=args.get("dims", 2),
        use_checkpoint=args.get("use_checkpoint", "True") == "True",
        use_fp16=args.get("use_fp16", "False") == "True",
        num_heads=args.get("num_heads", 4),
        num_head_channels=args.get("num_head_channels", -1),
        resblock_updown=args.get("resblock_updown", "True") == "True",
        num_groups=args.get("num_groups", 32),
        resample_2d=args.get("resample_2d", "False") == "True",
        scale_factor=args.get("sf", 2),
        resizer_num_layers=args.get("resizer_num_layers", 3),
        resizer_avg_pool=args.get("resizer_avg_pool", "False") == "True",
        activation=args.get("activation", "LeakyReLU"),
        rope_theta=args.get("rope_theta", 10000.0),
        attention_type=args.get("attention_type", "attention"),
    )

    # Build decoder config
    use_sh = args.get("use_sh_color", False)
    decoder_cfg = MoEConfig(
        kernel=args.get("kernel", 32),
        sharpening_factor=args.get("sharpening_factor", 1.0),
        kernel_type=KernelType(args.get("kernel_type", "gaussian")),
        activation=args.get("activation", "LeakyReLU"),
        use_sh_color=use_sh,
        sh_degree=args.get("sh_degree", 3) if use_sh else 3,
    )

    # Build autoencoder config
    autoencoder_cfg = AutoencoderConfig(
        EncoderConfig=encoder_cfg,
        DecoderConfig=decoder_cfg,
        d_in=args.get("im_chn", 3),
        phw=args.get("phw", 16),
        overlap=args.get("overlap", 8),
        dep_S=args.get("dep_S", 5),
        dep_K=args.get("dep_K", 3),
    )

    print("Creating model...")
    try:
        net = Autoencoder(cfg=autoencoder_cfg)
        print("✓ Model created successfully")

        # Count parameters
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nParameter counts:")
        print(f"  Encoder: {count_params(net.encoder) / 1e6:.2f}M")
        print(f"  Decoder: {count_params(net.decoder) / 1e6:.2f}M")
        print(f"  SNet: {count_params(net.snet) / 1e6:.2f}M")
        print(f"  KNet: {count_params(net.knet) / 1e6:.2f}M")
        print(f"  Total: {count_params(net) / 1e6:.2f}M")

        # Check SH mode
        if use_sh:
            print(f"\n✓ Spherical Harmonics ENABLED")
            print(f"  SH Degree: {decoder_cfg.sh_degree}")
            print(f"  Coefficients per channel: {(decoder_cfg.sh_degree + 1)**2}")
            print(f"  Total SH params (RGB): {3 * (decoder_cfg.sh_degree + 1)**2}")
        else:
            print(f"\n  Spherical Harmonics: Disabled (standard mode)")

        # Test forward pass
        print(f"\nTesting forward pass...")
        batch_size = 1
        im_chn = args.get("im_chn", 3)
        hr_size = args.get("hr_size", 96)
        sf = args.get("sf", 2)
        lr_size = hr_size // sf

        dummy_input = torch.randn(batch_size, im_chn, lr_size, lr_size)

        with torch.no_grad():
            mu, kinfo, sigma = net(dummy_input)

        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {mu.shape}")
        print(f"  Expected: [{batch_size}, {im_chn}, {hr_size}, {hr_size}]")

        if mu.shape == (batch_size, im_chn, hr_size, hr_size):
            print(f"✓ Forward pass successful!")
        else:
            print(f"✗ Unexpected output shape!")

    except Exception as e:
        print(f"✗ Model initialization failed!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    print()
    return True


def check_config(config_path):
    """Check configuration file"""
    print("=" * 60)
    print("Configuration Check")
    print("=" * 60)

    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False

    print(f"Config file: {config_path}")
    print(f"  ✓ Exists")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Check essential parameters
    essential = ["im_chn", "batch_size", "epochs", "sf", "train_hr_patchs", "val_hr_path"]
    print(f"\nEssential parameters:")
    for param in essential:
        value = config.get(param, "NOT SET")
        print(f"  {param}: {value}")

    # Check SH parameters
    if config.get("use_sh_color", False):
        print(f"\n3DGS-Style Spherical Harmonics:")
        print(f"  use_sh_color: {config.get('use_sh_color')}")
        print(f"  sh_degree: {config.get('sh_degree', 'NOT SET (will use default: 3)')}")
        print(f"  sharpening_factor: {config.get('sharpening_factor', 'NOT SET (will use default: 1.0)')}")

    print()
    return True


def main():
    print("\n" + "=" * 60)
    print("N-SMoE Pre-Training Validation")
    print("=" * 60)
    print()

    # Default config path
    config_path = Path(__file__).parent / "configs" / "local_sisr_x2.json"

    # Run checks
    checks_passed = True

    # 1. CUDA check
    num_gpus = check_cuda()

    # 2. Config check
    if not check_config(config_path):
        checks_passed = False

    # 3. Data paths check
    check_data_paths(config_path)

    # 4. Model initialization check
    if not check_model_init(config_path):
        checks_passed = False

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if checks_passed:
        print("✓ All checks passed!")
        print("\nYou can start training with:")
        print("  python3 train_sr.py")
        if num_gpus > 1:
            print(f"\nMulti-GPU training will use {num_gpus} GPUs automatically")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
