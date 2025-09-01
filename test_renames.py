#!/usr/bin/env python3
"""Test script to verify all renamed symbols work correctly."""

import sys
import os

sys.path.append("tpami/VIRNet")


def test_class_renames():
    """Test that renamed classes can be imported and instantiated."""
    print("Testing class renames...")

    try:
        from networks.network_moex import (
            LegacyAttentionPool2d,
            MixtureOfExpertsDecoder,
            MoEConfig,
            # Test backward compatibility aliases
            AttentionPool2d_,
            MoE,
        )

        # Test new names work
        config = MoEConfig()
        decoder = MixtureOfExpertsDecoder(config)
        print("✓ New class names work")

        # Test aliases work
        legacy_pool = AttentionPool2d_(64, 128, 8, 256)
        alias_decoder = MoE(config)
        print("✓ Backward compatibility aliases work")

        # Test that new and old refer to same classes
        assert AttentionPool2d_ is LegacyAttentionPool2d
        assert MoE is MixtureOfExpertsDecoder
        print("✓ Aliases point to correct classes")

    except Exception as e:
        print(f"✗ Class rename test failed: {e}")
        return False

    return True


def test_config_fields():
    """Test that config field renames work."""
    print("\nTesting config field renames...")

    try:
        from networks.network_moex import MoEConfig

        config = MoEConfig()
        # Test new field name
        assert hasattr(config, "num_kernels")
        assert config.num_kernels == 4  # default value
        print("✓ num_kernels field exists with correct default")

        # Test that old field name is gone
        assert not hasattr(config, "kernel")
        print("✓ Old 'kernel' field is removed")

    except Exception as e:
        print(f"✗ Config field test failed: {e}")
        return False

    return True


def test_method_renames():
    """Test that method renames work."""
    print("\nTesting method renames...")

    try:
        from networks.network_moex import MixtureOfExpertsDecoder, MoEConfig

        config = MoEConfig()
        decoder = MixtureOfExpertsDecoder(config)

        # Test new method names exist
        assert hasattr(decoder, "forward_spatial_fixed")
        assert hasattr(decoder, "forward_spatial_dynamic")
        print("✓ New method names exist")

        # Test old method names are gone
        assert not hasattr(decoder, "forward_spatial")
        assert not hasattr(decoder, "forward_spatial_")
        print("✓ Old method names are removed")

    except Exception as e:
        print(f"✗ Method rename test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Rename Refactor")
    print("=" * 50)

    tests = [test_class_renames, test_config_fields, test_method_renames]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All rename tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
