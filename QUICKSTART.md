# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd fused-softmax-dropout-cuda

# Install in development mode
pip install -e .
```

## Basic Usage

```python
import torch
import fused_softmax_dropout

# Create input tensor
x = torch.randn(4, 128, 512, device="cuda", dtype=torch.float32)

# Run fused softmax + dropout
y = fused_softmax_dropout.fused_softmax_dropout(x, p=0.1, training=True)
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

```bash
python benchmarks/benchmark.py
```

## Troubleshooting

- **CUDA not found**: Ensure PyTorch was installed with CUDA support
- **Compilation errors**: Check that your CUDA toolkit version matches PyTorch's CUDA version
- **Import errors**: Make sure you've run `pip install -e .` after cloning
