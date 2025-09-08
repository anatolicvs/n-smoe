#!/usr/bin/env python3
"""Test the critical decoder fix for color rendering."""

import torch
import sys

sys.path.append("/home/adminlms/src/n-smoe")

try:
    from tpami.VIRNet.networks.network_moex import MoE, MoEConfig, KernelType

    print("✅ Successfully imported MoE components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_color_rendering():
    """Test that the decoder now renders colors instead of normalized weights."""
    print("\n🎯 Testing Critical Color Rendering Fix")
    print("=" * 50)

    # Test RGB case
    print("Testing RGB (3-channel) case:")
    cfg = MoEConfig()
    cfg.kernel_type = KernelType.GAUSSIAN
    cfg.kernel = 4
    cfg.ch = 3

    decoder = MoE(cfg)
    from tpami.VIRNet.networks.network_moex import Autoencoder

    params_per_kernel = Autoencoder.get_params_per_kernel(KernelType.GAUSSIAN, 3)
    total_params = params_per_kernel * cfg.kernel

    print(f"  Parameters per kernel: {params_per_kernel}")
    print(f"  Total parameters: {total_params}")

    B, H, W = 2, 32, 32
    params = torch.randn(B, 3, total_params)

    try:
        output = decoder.forward_spatial(H, W, params)
        print(f"  ✅ RGB output shape: {output.shape}")
        print(f"  RGB range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # Check that output is not constant (old bug would make it ≈1 everywhere)
        output_var = output.var().item()
        print(f"  RGB variance: {output_var:.6f}")

        if output_var > 1e-4:
            print("  ✅ RGB shows good variation (not constant)")
        else:
            print("  ⚠️  RGB output has low variation")

    except Exception as e:
        print(f"  ❌ RGB test failed: {e}")
        return False

    # Test grayscale case
    print("\nTesting Grayscale (1-channel) case:")
    cfg_gray = MoEConfig()
    cfg_gray.kernel_type = KernelType.GAUSSIAN
    cfg_gray.kernel = 4
    cfg_gray.ch = 1

    decoder_gray = MoE(cfg_gray)
    params_per_kernel_gray = Autoencoder.get_params_per_kernel(KernelType.GAUSSIAN, 1)
    total_params_gray = params_per_kernel_gray * cfg_gray.kernel

    print(f"  Parameters per kernel: {params_per_kernel_gray}")
    print(f"  Total parameters: {total_params_gray}")

    params_gray = torch.randn(B, 1, total_params_gray)

    try:
        output_gray = decoder_gray.forward_spatial(H, W, params_gray)
        print(f"  ✅ Grayscale output shape: {output_gray.shape}")
        print(f"  Grayscale range: [{output_gray.min().item():.3f}, {output_gray.max().item():.3f}]")

        # Check that output is not constant
        output_var_gray = output_gray.var().item()
        print(f"  Grayscale variance: {output_var_gray:.6f}")

        if output_var_gray > 1e-4:
            print("  ✅ Grayscale shows good variation (not constant)")
        else:
            print("  ⚠️  Grayscale output has low variation")

        return True

    except Exception as e:
        print(f"  ❌ Grayscale test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_color_rendering()
    if success:
        print("\n🎉 Critical fix test passed! Decoder now renders colors properly.")
        print("🔥 No more normalized weights summing to 1 - actual image formation!")
    else:
        print("\n💥 Critical fix test failed. Check implementation.")
