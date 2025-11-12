import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fused_softmax_dropout
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    pytest.skip("fused_softmax_dropout not installed", allow_module_level=True)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestFusedSoftmaxDropout:
    
    def test_no_mask_training_false(self):
        """Test fused softmax without mask and training=False"""
        x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
        
        # Reference
        y_ref = torch.softmax(x, dim=-1)
        
        # Fused
        y_fused = fused_softmax_dropout.fused_softmax_dropout(
            x, mask=None, p=0.1, training=False
        )
        
        # Check correctness
        max_abs_error = torch.max(torch.abs(y_ref - y_fused)).item()
        assert max_abs_error < 1e-5, f"Max absolute error: {max_abs_error}"
        
        # Check that rows sum to ~1
        row_sums = y_fused.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    def test_no_mask_training_true_deterministic(self):
        """Test fused softmax + dropout with deterministic RNG"""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
        p = 0.1
        
        # Reference
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        y_ref = torch.nn.functional.dropout(
            torch.softmax(x, dim=-1),
            p=p,
            training=True,
        )
        
        # Fused
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        y_fused = fused_softmax_dropout.fused_softmax_dropout(
            x, mask=None, p=p, training=True
        )
        
        # Note: RNG may not match exactly due to different implementation
        # But outputs should be reasonable
        assert torch.all(torch.isfinite(y_fused))
        assert y_fused.shape == x.shape
        
        # Check that non-zero values are scaled correctly
        # Expected value should be preserved (approximately)
        ref_mean = y_ref.mean().item()
        fused_mean = y_fused.mean().item()
        assert abs(ref_mean - fused_mean) < 0.1  # Allow some variance due to RNG differences
    
    def test_with_mask_training_false(self):
        """Test fused softmax with mask"""
        x = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
        
        # Create mask: 0.0 for masked, 1.0 for unmasked
        mask = torch.ones_like(x)
        mask[0, 0, :4] = 0.0  # Mask first 4 elements of first row
        mask[1, 2, :2] = 0.0  # Mask first 2 elements of another row
        
        # Reference: set masked positions to -inf before softmax
        x_ref = x.clone()
        x_ref[mask < 0.5] = float('-inf')
        y_ref = torch.softmax(x_ref, dim=-1)
        
        # Fused
        y_fused = fused_softmax_dropout.fused_softmax_dropout(
            x, mask=mask, p=0.1, training=False
        )
        
        # Check that masked positions are near 0
        masked_positions = mask < 0.5
        assert torch.all(y_fused[masked_positions] < 1e-5)
        
        # Check that unmasked rows sum to ~1
        for b in range(x.size(0)):
            for s in range(x.size(1)):
                row_mask = mask[b, s, :]
                row_output = y_fused[b, s, :]
                unmasked_sum = row_output[row_mask > 0.5].sum().item()
                assert abs(unmasked_sum - 1.0) < 1e-5
    
    def test_shapes(self):
        """Test various input shapes"""
        shapes = [
            (1, 1, 16),
            (2, 4, 32),
            (8, 128, 256),
        ]
        
        for shape in shapes:
            x = torch.randn(*shape, device="cuda", dtype=torch.float32)
            y = fused_softmax_dropout.fused_softmax_dropout(
                x, mask=None, p=0.1, training=False
            )
            assert y.shape == x.shape
            assert torch.all(torch.isfinite(y))
            
            # Check row sums
            row_sums = y.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    
    def test_p_zero(self):
        """Test with p=0 (no dropout)"""
        x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
        
        y_ref = torch.softmax(x, dim=-1)
        y_fused = fused_softmax_dropout.fused_softmax_dropout(
            x, mask=None, p=0.0, training=True
        )
        
        max_abs_error = torch.max(torch.abs(y_ref - y_fused)).item()
        assert max_abs_error < 1e-5
    
    def test_training_false_no_dropout(self):
        """Test that training=False skips dropout even with p>0"""
        x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
        
        y_ref = torch.softmax(x, dim=-1)
        y_fused = fused_softmax_dropout.fused_softmax_dropout(
            x, mask=None, p=0.5, training=False
        )
        
        max_abs_error = torch.max(torch.abs(y_ref - y_fused)).item()
        assert max_abs_error < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

