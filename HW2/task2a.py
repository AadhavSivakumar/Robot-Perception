import open3d as o3d
import numpy as np
import copy
from typing import Tuple
from tqdm import tqdm

# Try to import CuPy for GPU acceleration
CUDA_AVAILABLE = False
try:
    import cupy as cp

    # Test if CUDA actually works
    try:
        test_array = cp.array([1, 2, 3])
        _ = test_array + 1
        CUDA_AVAILABLE = True
        print("✓ CUDA acceleration available and working")
    except Exception as e:
        print(f"✗ CUDA libraries found but not functional: {type(e).__name__}")
        print("  Falling back to optimized CPU implementation")
        CUDA_AVAILABLE = False
except ImportError:
    print("✗ CuPy not installed, using optimized CPU implementation")
    print("  To enable GPU: pip install cupy-cuda12x (requires NVIDIA GPU + CUDA toolkit)")


def find_closest_points_gpu(source_points: np.ndarray,
                            target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated closest point finding using CuPy.

    Parameters:
    -----------
    source_points : np.ndarray
        Source point cloud points (N x 3)
    target_points : np.ndarray
        Target point cloud points (M x 3)

    Returns:
    --------
    distances : np.ndarray
        Distance to closest point for each source point
    indices : np.ndarray
        Index of closest target point for each source point
    """
    try:
        # Transfer data to GPU
        print(f"\n      → Transferring {len(source_points):,} points to GPU...", end='', flush=True)
        source_gpu = cp.asarray(source_points)
        target_gpu = cp.asarray(target_points)
        print(" ✓", flush=True)

        n_source = source_gpu.shape[0]
        n_target = target_gpu.shape[0]

        # Process in batches to avoid GPU memory issues
        batch_size = 1000
        num_batches = (n_source + batch_size - 1) // batch_size
        distances = []
        indices = []

        print(f"      → Processing {num_batches} batches on GPU...", flush=True)

        import time
        for batch_idx, i in enumerate(range(0, n_source, batch_size)):
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"         Batch {batch_idx + 1}/{num_batches} ({100 * (batch_idx + 1) / num_batches:.1f}%)",
                      flush=True)

            batch_end = min(i + batch_size, n_source)
            batch_source = source_gpu[i:batch_end]

            # Compute all pairwise distances for this batch
            diff = batch_source[:, cp.newaxis, :] - target_gpu[cp.newaxis, :, :]
            batch_distances = cp.linalg.norm(diff, axis=2)

            # Find minimum distance and index for each point in batch
            batch_min_distances = cp.min(batch_distances, axis=1)
            batch_min_indices = cp.argmin(batch_distances, axis=1)

            distances.append(batch_min_distances)
            indices.append(batch_min_indices)

        print(f"      → Transferring results back to CPU...", end='', flush=True)
        # Concatenate results and transfer back to CPU
        distances = cp.concatenate(distances)
        indices = cp.concatenate(indices)

        result = cp.asnumpy(distances), cp.asnumpy(indices)
        print(" ✓", flush=True)

        return result

    except Exception as e:
        # If GPU operations fail, fall back to CPU
        print(f"\n⚠ GPU operation failed ({type(e).__name__}), falling back to CPU")
        return find_closest_points_cpu(source_points, target_points)


def find_closest_points_cpu(source_points: np.ndarray,
                            target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized CPU-based closest point finding using vectorized operations.
    Uses chunking to balance speed and memory usage.

    Parameters:
    -----------
    source_points : np.ndarray
        Source point cloud points (N x 3)
    target_points : np.ndarray
        Target point cloud points (M x 3)

    Returns:
    --------
    distances : np.ndarray
        Distance to closest point for each source point
    indices : np.ndarray
        Index of closest target point for each source point
    """
    n_source = source_points.shape[0]
    n_target = target_points.shape[0]

    # Adjust batch size based on memory constraints
    # For large point clouds, use smaller batches
    if n_source > 50000 or n_target > 50000:
        batch_size = 200
    elif n_source > 10000 or n_target > 10000:
        batch_size = 500
    else:
        batch_size = 1000

    distances = np.zeros(n_source, dtype=np.float32)
    indices = np.zeros(n_source, dtype=np.int32)

    num_batches = (n_source + batch_size - 1) // batch_size
    print(f"\n      → Processing {num_batches} batches on CPU...", flush=True)

    # Process in batches for memory efficiency
    for batch_idx, i in enumerate(range(0, n_source, batch_size)):
        if batch_idx % 5 == 0:  # Print every 5 batches
            print(f"         Batch {batch_idx + 1}/{num_batches} ({100 * (batch_idx + 1) / num_batches:.1f}%)",
                  flush=True)

        batch_end = min(i + batch_size, n_source)
        batch_source = source_points[i:batch_end]

        # Vectorized distance computation using broadcasting
        # Shape: (batch_size, 1, 3) - (1, n_target, 3) = (batch_size, n_target, 3)
        diff = batch_source[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        batch_distances = np.sqrt(np.sum(diff ** 2, axis=2))  # Faster than linalg.norm

        batch_min_distances = np.min(batch_distances, axis=1)
        batch_min_indices = np.argmin(batch_distances, axis=1)

        distances[i:batch_end] = batch_min_distances
        indices[i:batch_end] = batch_min_indices

    return distances, indices


def find_closest_points(source_points: np.ndarray,
                        target_points: np.ndarray,
                        use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the closest point in target for each point in source.
    Automatically uses GPU if available and requested.

    Parameters:
    -----------
    source_points : np.ndarray
        Source point cloud points (N x 3)
    target_points : np.ndarray
        Target point cloud points (M x 3)
    use_gpu : bool
        Whether to use GPU acceleration if available

    Returns:
    --------
    distances : np.ndarray
        Distance to closest point for each source point
    indices : np.ndarray
        Index of closest target point for each source point
    """
    if use_gpu and CUDA_AVAILABLE:
        return find_closest_points_gpu(source_points, target_points)
    else:
        return find_closest_points_cpu(source_points, target_points)


def compute_transformation(source_points: np.ndarray,
                           target_points: np.ndarray) -> np.ndarray:
    """
    Compute the optimal transformation (rotation + translation) that aligns
    source points to target points using SVD.

    This implements the solution to the Procrustes problem.

    Parameters:
    -----------
    source_points : np.ndarray
        Source point cloud points (N x 3)
    target_points : np.ndarray
        Corresponding target point cloud points (N x 3)

    Returns:
    --------
    transformation : np.ndarray
        4x4 homogeneous transformation matrix
    """
    # Step 1: Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Step 2: Center the point clouds
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Step 3: Compute the covariance matrix H
    H = source_centered.T @ target_centered

    # Step 4: Compute SVD of H
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Compute rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case (det(R) = -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 6: Compute translation vector
    t = target_centroid - R @ source_centroid

    # Step 7: Build 4x4 homogeneous transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation


def icp_registration(source: o3d.geometry.PointCloud,
                     target: o3d.geometry.PointCloud,
                     max_iterations: int = 50,
                     tolerance: float = 1e-6,
                     max_correspondence_distance: float = 0.05,
                     use_gpu: bool = True) -> np.ndarray:
    """
    Custom implementation of the Iterative Closest Point (ICP) algorithm.
    GPU-accelerated when CuPy is available.

    Algorithm:
    1. Find closest points between source and target (GPU-accelerated)
    2. Compute optimal transformation using SVD
    3. Apply transformation to source
    4. Repeat until convergence

    Parameters:
    -----------
    source : o3d.geometry.PointCloud
        Source point cloud to be aligned
    target : o3d.geometry.PointCloud
        Target point cloud (reference)
    max_iterations : int
        Maximum number of ICP iterations
    tolerance : float
        Convergence threshold (change in mean squared error)
    max_correspondence_distance : float
        Maximum distance for a point pair to be considered a correspondence
    use_gpu : bool
        Whether to use GPU acceleration if available

    Returns:
    --------
    transformation : np.ndarray
        Final 4x4 homogeneous transformation matrix
    """
    # Convert to numpy arrays
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # Initialize transformation as identity
    current_transformation = np.eye(4)

    # Make a copy of source points for iteration
    current_source = source_points.copy()

    prev_error = float('inf')

    device_type = "GPU" if (use_gpu and CUDA_AVAILABLE) else "CPU"
    print(f"\n{'=' * 70}")
    print(f"Starting ICP iterations on {device_type}")
    print(f"{'=' * 70}")
    print(f"Source points: {len(source_points):,}")
    print(f"Target points: {len(target_points):,}")

    # Estimate time
    if use_gpu and CUDA_AVAILABLE:
        print("\n⚡ GPU mode enabled!")
        print("   Note: First iteration will be slower (GPU warmup/JIT compilation)")
        print("   Subsequent iterations will be much faster!")

    print(f"\n{'=' * 70}")
    import sys
    sys.stdout.flush()

    import time
    start_time = time.time()

    for iteration in range(max_iterations):
        iter_start = time.time()

        print(f"\n{'─' * 70}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'─' * 70}")

        # Step 1: Find closest points (GPU-accelerated if available)
        print(f"[1/4] Finding closest points...", end='', flush=True)
        step_start = time.time()
        distances, indices = find_closest_points(current_source, target_points, use_gpu=use_gpu)
        print(f" ✓ Done ({time.time() - step_start:.2f}s)", flush=True)

        print(f"[2/4] Filtering correspondences...", end='', flush=True)
        step_start = time.time()

        print(f"[2/4] Filtering correspondences...", end='', flush=True)
        step_start = time.time()

        # Filter correspondences by maximum distance
        valid_mask = distances < max_correspondence_distance
        valid_source = current_source[valid_mask]
        valid_target = target_points[indices[valid_mask]]

        num_inliers = len(valid_source)
        print(f" ✓ Done ({time.time() - step_start:.3f}s)", flush=True)
        print(
            f"      Inliers: {num_inliers:,}/{len(current_source):,} ({100 * num_inliers / len(current_source):.1f}%)")

        if num_inliers < 3:
            print(f"\n⚠ Only {num_inliers} valid correspondences found. Stopping.")
            break

        # Step 2: Compute mean squared error
        print(f"[3/4] Computing error metrics...", end='', flush=True)
        step_start = time.time()
        mean_error = np.mean(distances[valid_mask] ** 2)
        error_change = abs(prev_error - mean_error)
        print(f" ✓ Done ({time.time() - step_start:.3f}s)", flush=True)
        print(f"      Mean Error: {mean_error:.6e}")
        print(f"      Error Change: {error_change:.6e}")

        # Check convergence
        if error_change < tolerance:
            elapsed = time.time() - start_time
            print(f"\n{'=' * 70}")
            print(f"✓ CONVERGED at iteration {iteration + 1}")
            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Average time per iteration: {elapsed / (iteration + 1):.2f}s")
            print(f"{'=' * 70}")
            break

        # Step 4: Compute transformation for this iteration
        print(f"[4/4] Computing transformation...", end='', flush=True)
        step_start = time.time()
        iteration_transformation = compute_transformation(valid_source, valid_target)
        print(f" ✓ Done ({time.time() - step_start:.3f}s)", flush=True)

        # Step 5: Update cumulative transformation
        current_transformation = iteration_transformation @ current_transformation

        # Step 6: Apply transformation to source points
        current_source = (iteration_transformation[:3, :3] @ source_points.T).T + \
                         iteration_transformation[:3, 3]

        prev_error = mean_error

        iter_time = time.time() - iter_start
        print(f"\n⏱  Iteration completed in {iter_time:.2f}s")

        # Estimate remaining time
        if iteration > 0:
            avg_time = (time.time() - start_time) / (iteration + 1)
            est_remaining = avg_time * (max_iterations - iteration - 1)
            print(f"📊 Estimated time remaining: {est_remaining:.1f}s")

    total_time = time.time() - start_time

    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"ICP COMPLETED")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final mean error: {mean_error:.6e}")
    print(f"Final inliers: {num_inliers:,}/{len(source_points):,} ({100 * num_inliers / len(source_points):.1f}%)")
    print(f"\nFinal transformation matrix:")
    print(current_transformation)
    print(f"{'=' * 70}\n")

    return current_transformation


def draw_registration_result(source, target, transformation):
    """
    Visualize the registration result.

    Parameters:
    -----------
    source : o3d.geometry.PointCloud
        Source point cloud
    target : o3d.geometry.PointCloud
        Target point cloud
    transformation : np.ndarray
        4x4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # Color the point clouds
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange for source
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue for target

    # Apply transformation to source
    source_temp.transform(transformation)

    # Visualize
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


# Demo usage
if __name__ == "__main__":
    # Load demo ICP point clouds
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    print("=" * 55)
    print("ICP Point Cloud Registration")
    print("=" * 55)
    print("\nSource point cloud:")
    print(source)
    print("\nTarget point cloud:")
    print(target)

    # Apply custom ICP with GPU acceleration
    transformation = icp_registration(source, target,
                                      max_iterations=50,
                                      tolerance=1e-6,
                                      max_correspondence_distance=0.05,
                                      use_gpu=True)  # Set to False to force CPU

    # Visualize the result
    print("\nLaunching visualization...")
    draw_registration_result(source, target, transformation)