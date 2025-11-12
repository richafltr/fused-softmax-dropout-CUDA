import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fused_softmax_dropout
except ImportError:
    print("Error: fused_softmax_dropout not installed. Run: pip install -e .")
    sys.exit(1)

if not torch.cuda.is_available():
    print("Error: CUDA not available")
    sys.exit(1)


def baseline(x, p, training=True):
    """PyTorch baseline: softmax + dropout"""
    y = torch.softmax(x, dim=-1)
    return torch.nn.functional.dropout(y, p=p, training=training)


def benchmark_function(func, x, p, training, num_warmup=10, num_iter=100):
    """Benchmark a function"""
    # Warmup
    for _ in range(num_warmup):
        _ = func(x, p, training)
    torch.cuda.synchronize()
    
    # Timing
    start = time.time()
    for _ in range(num_iter):
        _ = func(x, p, training)
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = (end - start) / num_iter * 1000  # ms per iteration
    return elapsed


def main():
    shapes = [
        (64, 128, 512),
        (32, 1024, 512),
        (16, 2048, 1024),
    ]
    
    p = 0.1
    training = True
    
    print("=" * 80)
    print("Benchmark: Fused Softmax + Dropout vs PyTorch Baseline")
    print("=" * 80)
    print(f"Dropout probability: {p}")
    print(f"Training mode: {training}")
    print()
    
    results = []
    
    for shape in shapes:
        print(f"Shape: {shape}")
        print("-" * 80)
        
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        
        # Baseline
        baseline_time = benchmark_function(
            lambda x, p, t: baseline(x, p, t),
            x, p, training
        )
        
        # Fused
        fused_time = benchmark_function(
            lambda x, p, t: fused_softmax_dropout.fused_softmax_dropout(x, None, p, t),
            x, p, training
        )
        
        speedup = baseline_time / fused_time
        
        print(f"  Baseline:  {baseline_time:.4f} ms")
        print(f"  Fused:     {fused_time:.4f} ms")
        print(f"  Speedup:   {speedup:.2f}x")
        print()
        
        results.append({
            "shape": shape,
            "baseline_ms": baseline_time,
            "fused_ms": fused_time,
            "speedup": speedup,
        })
    
    print("=" * 80)
    print("Summary:")
    print("-" * 80)
    for r in results:
        print(f"Shape {r['shape']}: {r['speedup']:.2f}x speedup")
    print("=" * 80)


if __name__ == "__main__":
    main()

