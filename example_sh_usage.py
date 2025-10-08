"""
Example: Using Spherical Harmonics for Color Prediction in N-SMoE

This script demonstrates how to configure and use the spherical harmonics
based color prediction in the N-SMoE autoencoder.
"""

import torch
from tpami.VIRNet.networks.network_moex import (
    Autoencoder,
    AutoencoderConfig,
    EncoderConfig,
    MoEConfig,
    KernelType,
)


def create_sh_autoencoder(
    d_in: int = 3,
    sh_lmax: int = 2,
    kernel_count: int = 8,
    kernel_type: KernelType = KernelType.GAUSSIAN_CAUCHY,
):
    """
    Create an autoencoder with spherical harmonics color prediction.

    Args:
        d_in: Number of input channels (1 for grayscale, 3 for RGB)
        sh_lmax: Maximum degree of spherical harmonics (0-4 recommended)
        kernel_count: Number of Gaussian mixture components per location
        kernel_type: GAUSSIAN or GAUSSIAN_CAUCHY

    Returns:
        Autoencoder model configured with SH color prediction
    """

    # Configure the encoder
    encoder_config = EncoderConfig(
        noise_cond=True,
        kernel_cond=True,
        noise_avg=False,
        sigma_chn=3,
        kernel_chn=3,
        model_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.0,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        scale_factor=4,  # 4x super-resolution
        activation="GELU",
    )

    # Configure the decoder with spherical harmonics
    decoder_config = MoEConfig(
        kernel=kernel_count,
        kernel_type=kernel_type,
        use_sh_color=True,  # ← Enable SH color prediction
        sh_lmax=sh_lmax,  # ← SH degree
        sh_mmax=None,  # Use full SH (defaults to lmax)
        sharpening_factor=1.0,
        activation="GELU",
        min_diag=1e-4,
        max_diag=1e2,
        reg_lambda=1e-3,
        scale_factor=4,
    )

    # Configure the autoencoder
    autoencoder_config = AutoencoderConfig(
        EncoderConfig=encoder_config,
        DecoderConfig=decoder_config,
        d_in=d_in,
        dep_S=5,  # Depth of noise estimation network
        dep_K=3,  # Depth of kernel estimation network
        phw=64,  # Patch height/width
        overlap=48,  # Patch overlap
        num_chunks=1,  # Number of chunks for gradient checkpointing
    )

    # Create the model
    model = Autoencoder(autoencoder_config)

    # Print configuration info
    if model.decoder.use_sh_color:
        print("✓ Spherical Harmonics Color Prediction ENABLED")
        print(f"  - SH Degree (lmax): {model.decoder.sh_lmax}")
        print(f"  - SH Coefficients per kernel: {model.decoder.sh_coeffs}")
        print(f"  - Kernel Type: {kernel_type.name}")
        print(f"  - Number of Gaussians: {kernel_count}")

        # Calculate parameter counts
        params_per_kernel = autoencoder_config.DecoderConfig.kernel
        if d_in == 1:
            color_params = model.decoder.sh_coeffs
        else:
            color_params = 3 * model.decoder.sh_coeffs

        spatial_params = 9 if kernel_type == KernelType.GAUSSIAN_CAUCHY else 7
        total_params = spatial_params + color_params

        print(f"  - Parameters per kernel: {total_params}")
        print(f"    └─ Spatial: {spatial_params}, Color (SH): {color_params}")
    else:
        print("✗ Using original color covariance (SH disabled or unavailable)")

    return model


def example_usage():
    """Example of how to use the SH-enabled autoencoder."""

    print("=" * 80)
    print("N-SMoE with Spherical Harmonics Color Prediction")
    print("=" * 80)
    print()

    # Create model with SH color prediction
    model = create_sh_autoencoder(
        d_in=3,  # RGB input
        sh_lmax=2,  # Quadratic SH basis
        kernel_count=8,  # 8 Gaussians per location
        kernel_type=KernelType.GAUSSIAN_CAUCHY,
    )

    print()
    print("-" * 80)
    print("Testing Forward Pass")
    print("-" * 80)

    # Create a dummy input (low-resolution RGB image)
    batch_size = 2
    height, width = 64, 64
    x_lr = torch.randn(batch_size, 3, height, width)

    print(f"Input shape: {x_lr.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        x_sr, sigma, kernel = model(x_lr)

    print(f"Output shape: {x_sr.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Kernel shape: {kernel.shape}")

    # Expected output shape for 4x SR
    expected_h = height * model.encoder.scale_factor
    expected_w = width * model.encoder.scale_factor
    assert x_sr.shape == (batch_size, 3, expected_h, expected_w), f"Unexpected output shape: {x_sr.shape}"

    print()
    print("✓ Forward pass successful!")
    print()

    # Visualize SH basis (optional)
    print("-" * 80)
    print("Spherical Harmonics Basis Visualization")
    print("-" * 80)

    # Generate coordinate grid
    coords = model.decoder.grid(32, 32, device="cpu")
    coords_batch = coords.unsqueeze(0)  # [1, 32, 32, 2]

    # Compute SH basis
    sh_basis = model.decoder.compute_sh_basis(coords_batch)

    print(f"SH Basis shape: {sh_basis.shape}")
    print(f"  - Grid size: 32x32")
    print(f"  - Number of basis functions: {sh_basis.shape[-1]}")

    # Show basis function statistics
    print()
    print("Basis Function Statistics:")
    for i in range(min(9, sh_basis.shape[-1])):
        basis_i = sh_basis[0, :, :, i]
        print(f"  Y_{i}: min={basis_i.min():.3f}, max={basis_i.max():.3f}, " f"mean={basis_i.mean():.3f}")

    print()
    print("=" * 80)
    print("Example Complete!")
    print("=" * 80)


def compare_modes():
    """Compare original vs SH color prediction."""

    print()
    print("=" * 80)
    print("Comparing Original vs Spherical Harmonics Color Prediction")
    print("=" * 80)
    print()

    # Original mode
    encoder_cfg = EncoderConfig(
        noise_cond=True,
        kernel_cond=True,
        noise_avg=False,
        sigma_chn=3,
        kernel_chn=3,
        model_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        scale_factor=4,
    )

    decoder_cfg_original = MoEConfig(
        kernel=8,
        kernel_type=KernelType.GAUSSIAN_CAUCHY,
        use_sh_color=False,  # Original mode
    )

    decoder_cfg_sh = MoEConfig(
        kernel=8,
        kernel_type=KernelType.GAUSSIAN_CAUCHY,
        use_sh_color=True,  # SH mode
        sh_lmax=2,
    )

    # Create both models
    model_original = Autoencoder(
        AutoencoderConfig(
            EncoderConfig=encoder_cfg,
            DecoderConfig=decoder_cfg_original,
            d_in=3,
            dep_S=5,
            dep_K=3,
            phw=64,
            overlap=48,
        )
    )

    model_sh = Autoencoder(
        AutoencoderConfig(
            EncoderConfig=encoder_cfg,
            DecoderConfig=decoder_cfg_sh,
            d_in=3,
            dep_S=5,
            dep_K=3,
            phw=64,
            overlap=48,
        )
    )

    # Count parameters
    params_original = sum(p.numel() for p in model_original.parameters())
    params_sh = sum(p.numel() for p in model_sh.parameters())

    print(f"Original Model Parameters: {params_original:,}")
    print(f"SH Model Parameters:       {params_sh:,}")
    print(
        f"Difference:                {params_sh - params_original:,} " f"({(params_sh/params_original - 1)*100:+.1f}%)"
    )

    # Test inference time (rough estimate)
    x = torch.randn(1, 3, 64, 64)

    import time

    model_original.eval()
    model_sh.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model_original(x)
            _ = model_sh(x)

        # Time original
        start = time.time()
        for _ in range(10):
            _ = model_original(x)
        time_original = (time.time() - start) / 10

        # Time SH
        start = time.time()
        for _ in range(10):
            _ = model_sh(x)
        time_sh = (time.time() - start) / 10

    print()
    print(f"Original Mode:  {time_original*1000:.1f} ms/image")
    print(f"SH Mode:        {time_sh*1000:.1f} ms/image")
    print(f"Overhead:       {(time_sh - time_original)*1000:+.1f} ms " f"({(time_sh/time_original - 1)*100:+.1f}%)")

    print()


if __name__ == "__main__":
    # Run example
    example_usage()

    # Compare modes
    try:
        compare_modes()
    except Exception as e:
        print(f"\nComparison skipped: {e}")

    print()
    print("For more information, see SPHERICAL_HARMONICS_INTEGRATION.md")
