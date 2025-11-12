# Fused Softmax + Dropout CUDA Extension

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Required-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**High-performance fused softmax + dropout operation for PyTorch**

Optimized CUDA kernel combining numerically stable softmax with inverted dropout for transformer architectures and attention mechanisms.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [API Reference](#api-reference) • [Performance](#performance) • [Contributing](#contributing)

</div>

---

## 🚀 Features

- ⚡ **Fused Operation**: Single CUDA kernel combining softmax and dropout for reduced memory bandwidth and improved performance
- 🔢 **Numerically Stable**: Row-wise softmax with max subtraction to prevent overflow/underflow
- 🎭 **Masking Support**: Optional attention mask with efficient `-inf` handling for masked positions
- 🎲 **PyTorch-Compatible RNG**: Uses Philox random number generator compatible with PyTorch's RNG state
- 🔄 **Inverted Dropout**: Standard inverted dropout implementation matching PyTorch's behavior
- 📊 **Comprehensive Testing**: Full test suite with correctness checks and benchmarks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Implementation Details](#implementation-details)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- **CUDA-capable GPU** (Compute Capability 3.5+)
- **PyTorch** with CUDA support (1.9.0+)
- **CUDA Toolkit** (compatible with your PyTorch CUDA version)
- **Python** 3.7 or higher
- **C++ compiler** with C++14 support
- **NVCC** (NVIDIA CUDA Compiler)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/richafltr/fused-softmax-dropout-CUDA.git
cd fused-softmax-dropout-cuda

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Verify Installation

```python
import torch
import fused_softmax_dropout

# Check if CUDA is available
assert torch.cuda.is_available(), "CUDA not available"

# Test import
x = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
y = fused_softmax_dropout.fused_softmax_dropout(x)
print("✓ Installation successful!")
```

## 🎯 Quick Start

```python
import torch
import fused_softmax_dropout

# Create input tensor: (batch, seq, d_model)
x = torch.randn(4, 128, 512, device="cuda", dtype=torch.float32)

# Basic usage: fused softmax + dropout
y = fused_softmax_dropout.fused_softmax_dropout(
    x, 
    mask=None, 
    p=0.1, 
    training=True
)
```

## 💡 Usage Examples

### Basic Usage

```python
import torch
import fused_softmax_dropout

# Input tensor: (batch_size, sequence_length, d_model)
x = torch.randn(32, 128, 512, device="cuda", dtype=torch.float32)

# Apply fused softmax + dropout
output = fused_softmax_dropout.fused_softmax_dropout(
    x, 
    p=0.1,           # Dropout probability
    training=True     # Training mode
)
```

### With Attention Mask

```python
# Create attention mask (0.0 = masked, 1.0 = unmasked)
mask = torch.ones(32, 128, 512, device="cuda", dtype=torch.float32)
mask[:, :, 256:] = 0.0  # Mask second half of each sequence

# Apply with mask
output = fused_softmax_dropout.fused_softmax_dropout(
    x, 
    mask=mask, 
    p=0.1, 
    training=True
)
```

### Inference Mode (No Dropout)

```python
# During inference, set training=False
output = fused_softmax_dropout.fused_softmax_dropout(
    x, 
    p=0.1, 
    training=False  # No dropout applied
)
```

### Integration with Transformer Models

```python
class AttentionLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
    
    def forward(self, q, k, v, mask=None):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply fused softmax + dropout
        attn_weights = fused_softmax_dropout.fused_softmax_dropout(
            scores,
            mask=mask,
            p=self.dropout,
            training=self.training
        )
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        return output
```

## 📚 API Reference

### `fused_softmax_dropout(x, mask=None, p=0.1, training=True)`

Fused softmax + dropout operation for CUDA tensors.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor of shape `(batch, seq, d_model)` on CUDA, dtype `float32` |
| `mask` | `torch.Tensor`, optional | Mask tensor of same shape as `x`. Values:<br>• `0.0`: Position is masked (treated as `-inf` in softmax)<br>• `1.0`: Position is unmasked |
| `p` | `float` | Dropout probability (default: `0.1`) |
| `training` | `bool` | Whether in training mode (default: `True`)<br>• `True`: Apply dropout<br>• `False`: Skip dropout, return softmax only |

#### Returns

- **`y`** (`torch.Tensor`): Output tensor of same shape as `x`

#### Notes

- ⚠️ Currently only supports `float32` tensors
- ⚠️ Input must be on CUDA device
- ✅ Masked positions in output are exactly `0.0`
- ✅ Unmasked rows sum to `~1.0` (exactly `1.0` when `training=False`)

#### Raises

- `ValueError`: If input is not on CUDA or not `float32`
- `RuntimeError`: If CUDA operations fail

## ⚡ Performance

### Benchmark Results

Run benchmarks to compare performance:

```bash
python benchmarks/benchmark.py
```

Expected speedups (varies by hardware and input size):
- **Small tensors** (64×128×512): ~1.2-1.5x speedup
- **Medium tensors** (32×1024×512): ~1.5-2.0x speedup
- **Large tensors** (16×2048×1024): ~2.0-3.0x speedup

### Why Fused Operations Matter

Fused operations reduce:
- **Memory bandwidth**: Single kernel reduces memory reads/writes
- **Kernel launch overhead**: One kernel instead of two
- **Intermediate storage**: No need to store softmax output before dropout

## 🔬 Implementation Details

### Kernel Design

- **Parallelization Strategy**: One CUDA block per row (one row = one sequence position)
- **Thread Mapping**: Each thread processes multiple columns using strided access pattern
- **Block Size**: 128 threads per block (configurable via `BLOCK_SIZE` macro)
- **Reductions**: Efficient warp-level and block-level reductions for max and sum computation
- **RNG**: Philox-based random number generation compatible with PyTorch's RNG state

### Numerical Stability

The softmax is computed using the standard numerically stable approach:

1. **Compute row-wise maximum** using parallel reduction
2. **Subtract max from all elements** to prevent overflow
3. **Compute exp(x - max)** element-wise
4. **Normalize by sum** to get probability distribution

Masked positions are set to `-inf` before the max computation, ensuring they contribute `0.0` to the softmax output.

### Memory Access Pattern

- **Coalesced access**: Threads access contiguous memory locations
- **Shared memory**: Used for broadcasting reduction results within blocks
- **Register usage**: Optimized to minimize register pressure

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fused_softmax_dropout --cov-report=html

# Run specific test
pytest tests/test_correctness.py::TestFusedSoftmaxDropout::test_no_mask_training_false -v
```

### Test Coverage

- ✅ Correctness checks vs PyTorch baseline
- ✅ Mask handling (various mask patterns)
- ✅ Multiple input shapes and sizes
- ✅ Edge cases (p=0, training=False, empty masks)
- ✅ Numerical stability verification
- ✅ RNG reproducibility

## 🛣️ Roadmap

### Current Status: ✅ v0.1.0 - Initial Release

- [x] Basic fused softmax + dropout kernel
- [x] Masking support
- [x] PyTorch RNG integration
- [x] Comprehensive test suite
- [x] Benchmarking infrastructure

### Planned Features

- [ ] **Half Precision (FP16) Support**
  - [ ] FP16 kernel implementation
  - [ ] Mixed precision training support
  - [ ] Performance optimizations for Tensor Cores

- [ ] **BFloat16 Support**
  - [ ] BFloat16 kernel implementation
  - [ ] Compatibility with modern transformer models

- [ ] **Backward Pass**
  - [ ] Gradient computation for autograd
  - [ ] Integration with PyTorch's autograd system

- [ ] **Additional Optimizations**
  - [ ] Tensor Core utilization for large dimensions
  - [ ] Multi-row processing per block
  - [ ] Improved memory access patterns

- [ ] **Extended Mask Support**
  - [ ] Causal mask optimization
  - [ ] Sparse mask support
  - [ ] Variable-length sequence batching

- [ ] **Documentation**
  - [ ] API documentation with Sphinx
  - [ ] Performance tuning guide
  - [ ] Architecture diagrams

- [ ] **CI/CD**
  - [ ] GitHub Actions for automated testing
  - [ ] Multi-CUDA version testing
  - [ ] Pre-built wheels for common configurations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/richafltr/fused-softmax-dropout-CUDA.git
cd fused-softmax-dropout-cuda
pip install -e ".[dev]"

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with tests
4. **Run tests** (`pytest tests/ -v`)
5. **Run benchmarks** (`python benchmarks/benchmark.py`)
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Code Style

- Follow PEP 8 for Python code
- Use `black` for code formatting
- Add docstrings to all public functions
- Include tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent C++ extension API
- NVIDIA for CUDA toolkit and documentation
- The open-source community for inspiration and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/richafltr/fused-softmax-dropout-CUDA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/richafltr/fused-softmax-dropout-CUDA/discussions)

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ for the PyTorch community**

[Report Bug](https://github.com/richafltr/fused-softmax-dropout-CUDA/issues) • [Request Feature](https://github.com/richafltr/fused-softmax-dropout-CUDA/issues) • [Documentation](#api-reference)

</div>
