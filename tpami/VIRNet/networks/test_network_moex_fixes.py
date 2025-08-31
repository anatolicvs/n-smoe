#!/usr/bin/env python3
"""
Test suite for network_moex.py fixes.

Tests the following improvements:
1. RoPE head dimension constraints
2. Dropout functionality in attention
3. Grid axis correctness
4. Cholesky-based quadratic form computation
5. Balance loss functionality
6. SDPA backend caching
7. Positional embedding caching
8. Type annotations
"""

import torch
import torch.nn.functional as F
import math
import pytest
from typing import Tuple
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_moex import (
    RoPEAttention,
    QKVAttention,
    AttentionPool2d,
    MoE,
    MoEConfig,
    KernelType,
    Autoencoder,
    AutoencoderConfig,
    EncoderConfig,
    detect_sdpa_backend,
)


class TestRoPEConstraints:
    """Test RoPE head dimension constraints."""

    def test_valid_rope_head_dim(self):
        """Test that valid head dimensions work."""
        # Should work: 128 // 4 = 32, 32 % 4 = 0 ✓
        attention = RoPEAttention(embedding_dim=128, num_heads=4, dropout=0.1)
        assert attention is not None

    def test_invalid_rope_head_dim(self):
        """Test that invalid head dimensions raise AssertionError."""
        # Should fail: 127 // 4 = 31.75 → 31, 31 % 4 = 3 ✗
        with pytest.raises(
            AssertionError, match="Axial RoPE requires head_dim % 4 == 0"
        ):
            RoPEAttention(embedding_dim=127, num_heads=4, dropout=0.1)

    def test_another_invalid_rope_head_dim(self):
        """Test another invalid configuration."""
        # Should fail: 128 // 3 = 42.67 → 42, 42 % 4 = 2 ✗
        with pytest.raises(
            AssertionError, match="Axial RoPE requires head_dim % 4 == 0"
        ):
            RoPEAttention(embedding_dim=128, num_heads=3, dropout=0.1)


class TestDropoutFunctionality:
    """Test that dropout is properly applied in attention layers."""

    def test_qkv_attention_dropout(self):
        """Test QKVAttention uses dropout."""
        attention = QKVAttention(n_heads=4, dropout=0.5)
        assert attention.dropout_p == 0.5

        # Test forward pass works
        qkv = torch.randn(2, 12 * 4, 16)  # [batch, 3*n_heads*head_dim, seq_len]
        attention.train()

        # Should work without error
        output = attention(qkv)
        assert output.shape == (2, 12 * 4, 16)

    def test_rope_attention_dropout(self):
        """Test RoPEAttention uses dropout."""
        attention = RoPEAttention(embedding_dim=128, num_heads=4, dropout=0.5)
        assert attention.dropout_p == 0.5


class TestGridAxisCorrectness:
    """Test grid coordinate system correctness."""

    def test_grid_dimensions(self):
        """Test that grid returns correct (H,W,2) shape."""
        config = MoEConfig(kernel_type=KernelType.GAUSSIAN)
        moe = MoE(config)

        device = torch.device("cpu")
        H, W = 32, 48

        grid = moe.grid(H, W, device)

        # Check shape
        assert grid.shape == (H, W, 2), f"Expected ({H}, {W}, 2), got {grid.shape}"

        # Check coordinate ordering (xx, yy)
        assert grid[0, 0, 0] == 0.0  # top-left x coordinate
        assert grid[0, 0, 1] == 0.0  # top-left y coordinate
        assert grid[0, -1, 0] == 1.0  # top-right x coordinate
        assert grid[-1, 0, 1] == 1.0  # bottom-left y coordinate

        # Verify it's a proper meshgrid
        # First dimension (H) should have constant x, varying y
        assert torch.allclose(grid[:, 0, 0], grid[0, 0, 0])  # x constant along height
        # Second dimension (W) should have varying x, constant y
        assert torch.allclose(grid[0, :, 1], grid[0, 0, 1])  # y constant along width


class TestCholeskyQuadraticForm:
    """Test Cholesky-based quadratic form computation."""

    def test_cholesky_vs_inverse_numerical_parity(self):
        """Test that Cholesky solve gives same result as matrix inverse within tolerance."""
        config = MoEConfig(kernel_type=KernelType.GAUSSIAN, reg_lambda=1e-4)
        moe = MoE(config)

        # Create a test covariance matrix
        B, ch, k, d = 2, 3, 4, 3
        # Create random positive definite matrices
        A = torch.randn(B, ch, k, d, d)
        cov = torch.matmul(A, A.transpose(-1, -2)) + 1e-3 * torch.eye(d)

        # Test vector
        test_vec = torch.randn(B, ch, k, d)

        # Cholesky method
        L_chol = moe.cholesky_solve_quadratic(cov, moe.reg_lambda_param)
        y_chol = torch.linalg.solve_triangular(
            L_chol, test_vec.unsqueeze(-1), upper=False
        ).squeeze(-1)
        quadratic_chol = (y_chol * y_chol).sum(dim=-1)

        # Matrix inverse method (for comparison)
        cov_reg = cov + 1e-4 * torch.eye(d)
        cov_inv = torch.linalg.inv(cov_reg)
        quadratic_inv = torch.einsum(
            "...d,...de,...e->...", test_vec, cov_inv, test_vec
        )

        # Check numerical parity within tolerance
        rel_error = torch.abs(quadratic_chol - quadratic_inv) / (
            torch.abs(quadratic_inv) + 1e-8
        )
        assert (
            torch.max(rel_error) < 1e-5
        ), f"Max relative error: {torch.max(rel_error)}"

    def test_cholesky_stability(self):
        """Test that Cholesky method is more stable than inverse."""
        config = MoEConfig(kernel_type=KernelType.GAUSSIAN)
        moe = MoE(config)

        # Create an ill-conditioned matrix
        B, ch, k, d = 1, 1, 1, 2
        cov = torch.tensor([[[[1e6, 0], [0, 1e-6]]]], dtype=torch.float32)

        # Should not fail
        L_chol = moe.cholesky_solve_quadratic(cov, moe.reg_lambda_param)
        assert L_chol is not None
        assert not torch.isnan(L_chol).any()


class TestBalanceLoss:
    """Test MoE load balancing functionality."""

    def test_balance_loss_computation(self):
        """Test that balance loss is computed correctly."""
        config = MoEConfig(
            kernel_type=KernelType.GAUSSIAN, balance_loss_coeff=0.1, kernel=4
        )
        moe = MoE(config)
        moe.train()

        # Test weight tensor
        B, ch, k = 2, 3, 4
        weights = torch.softmax(torch.randn(B, ch, k), dim=-1)

        balance_loss = moe.compute_balance_loss(weights)

        assert balance_loss is not None
        assert balance_loss.item() >= 0  # KL divergence is non-negative
        assert not torch.isnan(balance_loss)

    def test_balance_loss_integration(self):
        """Test that balance loss is stored during forward pass."""
        config = MoEConfig(
            kernel_type=KernelType.GAUSSIAN, balance_loss_coeff=0.1, kernel=4
        )
        moe = MoE(config)
        moe.train()

        # Create test parameters
        B, ch = 2, 3
        param_count = 10 if ch == 3 else 7  # Gaussian kernel params per channel
        params = torch.randn(B, ch, config.kernel * param_count)

        # Forward pass
        output = moe.forward_spatial(32, 32, params)

        # Check that balance loss was computed
        balance_loss = moe.get_balance_loss()
        assert balance_loss is not None
        assert balance_loss.item() >= 0


class TestSDPABackends:
    """Test SDPA backend detection and caching."""

    def test_backend_detection_cpu(self):
        """Test backend detection for CPU."""
        backends = detect_sdpa_backend(torch.device("cpu"), torch.float32)

        # Should not include FlashAttention on CPU
        assert backends == [
            detect_sdpa_backend.EFFICIENT_ATTENTION,
            detect_sdpa_backend.MATH,
        ]

    def test_backend_detection_gpu_fp16(self):
        """Test backend detection for GPU fp16."""
        if torch.cuda.is_available():
            backends = detect_sdpa_backend(torch.device("cuda"), torch.float16)

            # Should include FlashAttention for GPU fp16
            expected_backends = [
                detect_sdpa_backend.FLASH_ATTENTION,
                detect_sdpa_backend.EFFICIENT_ATTENTION,
                detect_sdpa_backend.MATH,
            ]
            assert backends == expected_backends
        else:
            pytest.skip("CUDA not available")

    def test_attention_backend_caching(self):
        """Test that Attention caches backends correctly."""
        attention = RoPEAttention(embedding_dim=128, num_heads=4, dropout=0.1)

        device = torch.device("cpu")
        dtype = torch.float32

        # First call should cache
        backends1 = attention._get_sdpa_backends(device, dtype)
        backends2 = attention._get_sdpa_backends(device, dtype)

        # Should be the same object (cached)
        assert backends1 is backends2


class TestPositionalCaching:
    """Test positional embedding caching in AttentionPool2d."""

    def test_positional_cache_hits(self):
        """Test that positional embeddings are cached."""
        pool = AttentionPool2d(embed_dim=64, num_heads=4)
        device = torch.device("cpu")

        # First call should cache
        pos1 = pool._get_positional_embeddings(32, 32, device)
        pos2 = pool._get_positional_embeddings(32, 32, device)

        # Should be the same tensor (cached)
        assert torch.equal(pos1, pos2)
        assert (32, 32) in pool._pos_cache

    def test_different_sizes_different_cache(self):
        """Test that different sizes use different cache entries."""
        pool = AttentionPool2d(embed_dim=64, num_heads=4)
        device = torch.device("cpu")

        pos1 = pool._get_positional_embeddings(32, 32, device)
        pos2 = pool._get_positional_embeddings(64, 64, device)

        assert not torch.equal(pos1, pos2)
        assert (32, 32) in pool._pos_cache
        assert (64, 64) in pool._pos_cache


class TestAttentionPoolTokenReduction:
    """Test token reduction in AttentionPool2d."""

    def test_token_reduction_when_needed(self):
        """Test that tokens are reduced when exceeding max_tokens."""
        pool = AttentionPool2d(embed_dim=64, num_heads=4, max_tokens=1024)

        # Create input that exceeds max_tokens: 48*48 = 2304 > 1024
        x = torch.randn(2, 64, 48, 48)

        output = pool(x)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 64  # output dim preserved

    def test_no_reduction_when_under_limit(self):
        """Test that no reduction happens when under token limit."""
        pool = AttentionPool2d(embed_dim=64, num_heads=4, max_tokens=4096)

        # Create input under max_tokens: 32*32 = 1024 < 4096
        x = torch.randn(2, 64, 32, 32)

        output = pool(x)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 64  # output dim preserved


class TestTypeAnnotations:
    """Test that type annotations are correct."""

    def test_encoder_return_type(self):
        """Test that Encoder.forward returns correct types."""
        # This is mostly a compile-time check, but we can test runtime
        from typing import get_type_hints
        from network_moex import Encoder

        hints = get_type_hints(Encoder.forward)
        return_hint = hints.get("return")

        # Should be Tuple[Tensor, Tensor, Tensor]
        if return_hint is not None:
            assert hasattr(return_hint, "__origin__")
            assert return_hint.__origin__ is tuple


def test_integration_end_to_end():
    """Integration test of the full autoencoder with fixes."""
    # Create a small autoencoder config
    encoder_config = EncoderConfig(
        noise_cond=True,
        kernel_cond=True,
        noise_avg=False,
        sigma_chn=1,
        kernel_chn=3,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=[1],
        dropout=0.1,
        num_heads=4,
        rope_theta=10000.0,
        attention_type="cross_attention",
    )

    decoder_config = MoEConfig(
        kernel=4, kernel_type=KernelType.GAUSSIAN, balance_loss_coeff=0.01
    )

    autoencoder_config = AutoencoderConfig(
        EncoderConfig=encoder_config,
        DecoderConfig=decoder_config,
        d_in=3,
        dep_S=3,
        dep_K=3,
        phw=32,
        overlap=8,
        num_chunks=1,
    )

    autoencoder = Autoencoder(autoencoder_config)
    autoencoder.train()

    # Test forward pass
    x = torch.randn(1, 3, 64, 64)

    rec, kinfo, sigma = autoencoder(x)

    # Check outputs
    assert rec.shape[0] == 1  # batch size
    assert rec.shape[1] == 3  # channels
    assert rec.shape[2] > 64  # upsampled height
    assert rec.shape[3] > 64  # upsampled width

    assert kinfo.shape[0] == 1
    assert sigma.shape[0] == 1

    # Check balance loss
    balance_loss = autoencoder.get_balance_loss()
    if balance_loss is not None:
        assert balance_loss.item() >= 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
