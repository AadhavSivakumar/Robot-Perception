import cupy as cp
import time

print("=" * 60)
print("Testing CUDA/GPU...")
print("=" * 60)

try:
    # Test basic operation
    x = cp.array([1, 2, 3])
    y = x + 1
    print("✓ Basic CuPy operation works!")
    print(f"  Result: {cp.asnumpy(y)}")

    # Test GPU info
    print(f"\n✓ CUDA available: {cp.cuda.is_available()}")
    print(f"✓ GPU device: {cp.cuda.Device()}")

    # Get device properties
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"✓ GPU name: {props['name'].decode()}")
    print(f"✓ Total memory: {props['totalGlobalMem'] / 1e9:.2f} GB")

    # Performance test
    print("\n" + "=" * 60)
    print("Running performance test...")
    print("=" * 60)

    size = 5000
    a = cp.random.random((size, size))
    b = cp.random.random((size, size))

    # Warmup
    _ = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()

    # Actual test
    start = time.time()
    c = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start

    print(f"✓ Matrix multiplication ({size}x{size}): {gpu_time:.3f} seconds")
    print("\n" + "=" * 60)
    print("🎉 GPU is working perfectly!")
    print("=" * 60)
    print("\nYou can now run GPU-accelerated ICP!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nGPU test failed!")