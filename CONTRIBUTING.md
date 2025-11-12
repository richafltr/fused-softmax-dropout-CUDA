# Contributing to Fused Softmax + Dropout CUDA Extension

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fused-softmax-dropout-CUDA.git
   cd fused-softmax-dropout-cuda
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .
   ```

4. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Run benchmarks** (if performance-related):
   ```bash
   python benchmarks/benchmark.py
   ```

5. **Check code style**:
   ```bash
   black --check .
   flake8 .
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request** on GitHub

## Coding Standards

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use `black` for code formatting (line length: 88)
- Add type hints where appropriate
- Write docstrings for all public functions and classes
- Use descriptive variable names

### C++/CUDA Code

- Follow Google C++ Style Guide
- Use meaningful variable names
- Add comments for complex logic
- Document kernel launch configurations
- Use `const` and `__restrict__` where appropriate

### Code Formatting

We use `black` for Python code formatting:

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Test edge cases and error conditions
- Use descriptive test names
- Group related tests in classes

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fused_softmax_dropout --cov-report=html

# Run specific test file
pytest tests/test_correctness.py -v

# Run specific test
pytest tests/test_correctness.py::TestFusedSoftmaxDropout::test_no_mask_training_false -v
```

### Test Coverage

- Aim for >90% code coverage
- Test both success and failure paths
- Include integration tests for complex workflows

## Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features
3. **Run all tests** and ensure they pass
4. **Update CHANGELOG.md** (if applicable) with your changes
5. **Rebase** your branch on the latest `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

### Pull Request Template

When opening a PR, please include:

- **Description**: Clear description of changes
- **Type**: Bug fix, feature, documentation, etc.
- **Testing**: How you tested your changes
- **Checklist**: Confirm you've completed all steps

### Review Process

- All PRs require at least one approval
- Address review comments promptly
- Keep PRs focused and reasonably sized
- Update your PR if requested

## Areas for Contribution

### High Priority

- **FP16/BFloat16 Support**: Half-precision kernels for memory efficiency
- **Backward Pass**: Gradient computation for autograd integration
- **Performance Optimizations**: Tensor Core utilization, improved memory access patterns
- **Extended Mask Support**: Causal masks, sparse masks, variable-length batching

### Documentation

- API documentation improvements
- Performance tuning guides
- Architecture diagrams
- Code examples and tutorials

### Testing

- Additional test cases
- Performance regression tests
- Multi-GPU testing
- Different CUDA version compatibility

### Infrastructure

- CI/CD setup (GitHub Actions)
- Pre-built wheels
- Docker images for testing
- Benchmark automation

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Minimal code example
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: CUDA version, PyTorch version, GPU model, OS
- **Error Messages**: Full error traceback if applicable

### Feature Requests

For feature requests, please include:

- **Use Case**: Why this feature would be useful
- **Proposed Solution**: How you envision it working
- **Alternatives**: Other approaches you've considered

## Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points for multiple changes
- Reference issues/PRs if applicable
```

Examples:
- `Fix mask handling bug in kernel.cu`
- `Add FP16 kernel implementation`
- `Improve error messages in bindings.cpp`
- `Update README with installation instructions`

## Questions?

- Open an issue for questions
- Check existing issues and discussions
- Review the codebase and documentation

Thank you for contributing!

