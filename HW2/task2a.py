import open3d as o3d
import numpy as np
import copy
import sys

# Try to import CUDA support
try:
    from numba import cuda
    import math

    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        print("CUDA is available!")
        print(f"CUDA Device: {cuda.get_current_device().name.decode()}")
    else:
        print("CUDA not available, will use CPU fallback")
except Exception as e:
    print(f"Error initializing CUDA: {e}")
    print("Will use CPU fallback")
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    # CUDA kernel for finding nearest neighbors
    @cuda.jit
    def find_nearest_neighbors_kernel(source_points, target_points, indices, distances):
        """
        CUDA kernel to find nearest neighbor for each source point in target point cloud

        Args:
            source_points: Nx3 array of source points
            target_points: Mx3 array of target points
            indices: Output array of size N containing nearest neighbor indices
            distances: Output array of size N containing nearest neighbor distances
        """
        idx = cuda.grid(1)

        if idx < source_points.shape[0]:
            min_dist = 1e10
            min_idx = -1

            # Find nearest neighbor in target
            for j in range(target_points.shape[0]):
                dx = source_points[idx, 0] - target_points[j, 0]
                dy = source_points[idx, 1] - target_points[j, 1]
                dz = source_points[idx, 2] - target_points[j, 2]
                dist = dx * dx + dy * dy + dz * dz

                if dist < min_dist:
                    min_dist = dist
                    min_idx = j

            indices[idx] = min_idx
            distances[idx] = math.sqrt(min_dist)


    @cuda.jit
    def transform_points_kernel(points, transformation, output):
        """
        CUDA kernel to apply 4x4 transformation matrix to points

        Args:
            points: Nx3 array of points
            transformation: 4x4 transformation matrix
            output: Nx3 output array of transformed points
        """
        idx = cuda.grid(1)

        if idx < points.shape[0]:
            x = points[idx, 0]
            y = points[idx, 1]
            z = points[idx, 2]

            # Apply transformation: [R|t] * [x,y,z,1]^T
            output[idx, 0] = transformation[0, 0] * x + transformation[0, 1] * y + transformation[0, 2] * z + \
                             transformation[0, 3]
            output[idx, 1] = transformation[1, 0] * x + transformation[1, 1] * y + transformation[1, 2] * z + \
                             transformation[1, 3]
            output[idx, 2] = transformation[2, 0] * x + transformation[2, 1] * y + transformation[2, 2] * z + \
                             transformation[2, 3]


def find_nearest_neighbors_cpu(source_points, target_points):
    """
    CPU fallback for finding nearest neighbors
    """
    n_source = source_points.shape[0]
    indices = np.zeros(n_source, dtype=np.int32)
    distances = np.zeros(n_source, dtype=np.float32)

    for i in range(n_source):
        diffs = target_points - source_points[i]
        dists = np.sum(diffs ** 2, axis=1)
        min_idx = np.argmin(dists)
        indices[i] = min_idx
        distances[i] = np.sqrt(dists[min_idx])

    return indices, distances


def transform_points_cpu(points, transformation):
    """
    CPU fallback for transforming points
    """
    n_points = points.shape[0]
    output = np.zeros_like(points)

    for i in range(n_points):
        x, y, z = points[i]
        output[i, 0] = transformation[0, 0] * x + transformation[0, 1] * y + transformation[0, 2] * z + transformation[
            0, 3]
        output[i, 1] = transformation[1, 0] * x + transformation[1, 1] * y + transformation[1, 2] * z + transformation[
            1, 3]
        output[i, 2] = transformation[2, 0] * x + transformation[2, 1] * y + transformation[2, 2] * z + transformation[
            2, 3]

    return output


def compute_transformation_svd(source_matched, target_matched):
    """
    Compute optimal transformation using SVD

    Args:
        source_matched: Nx3 array of matched source points
        target_matched: Nx3 array of matched target points

    Returns:
        4x4 transformation matrix
    """
    # Compute centroids
    centroid_source = np.mean(source_matched, axis=0)
    centroid_target = np.mean(target_matched, axis=0)

    # Center the points
    source_centered = source_matched - centroid_source
    target_centered = target_matched - centroid_target

    # Compute cross-covariance matrix
    H = source_centered.T @ target_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_target - R @ centroid_source

    # Build 4x4 transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation


def icp_cuda(source, target, max_iterations=50, tolerance=1e-6, use_cuda=None):
    """
    Iterative Closest Point algorithm with CUDA acceleration (if available)

    Args:
        source: Open3D point cloud (source)
        target: Open3D point cloud (target)
        max_iterations: Maximum number of ICP iterations
        tolerance: Convergence tolerance
        use_cuda: Force CUDA on/off (None for auto-detect)

    Returns:
        Final 4x4 transformation matrix
    """
    # Determine whether to use CUDA
    if use_cuda is None:
        use_cuda = CUDA_AVAILABLE
    elif use_cuda and not CUDA_AVAILABLE:
        print("Warning: CUDA requested but not available. Using CPU.")
        use_cuda = False

    # Convert to numpy arrays
    source_points = np.asarray(source.points).astype(np.float32)
    target_points = np.asarray(target.points).astype(np.float32)

    n_source = source_points.shape[0]
    n_target = target_points.shape[0]

    # Initialize transformation
    transformation = np.eye(4, dtype=np.float32)

    # Current source points (will be updated each iteration)
    current_source = source_points.copy()

    print(f"\nStarting ICP refinement {'with CUDA acceleration' if use_cuda else 'with CPU'}...")
    print(f"Source points: {n_source}, Target points: {n_target}")

    # CUDA-specific setup
    if use_cuda:
        try:
            # CUDA configuration
            threads_per_block = 256
            blocks = (n_source + threads_per_block - 1) // threads_per_block

            # Allocate device memory for target (stays on GPU)
            d_target = cuda.to_device(target_points)
            print(f"CUDA initialized: {blocks} blocks × {threads_per_block} threads")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            print("Falling back to CPU")
            use_cuda = False

    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors
        if use_cuda:
            try:
                # CUDA version
                indices = np.zeros(n_source, dtype=np.int32)
                distances = np.zeros(n_source, dtype=np.float32)

                d_current_source = cuda.to_device(current_source)
                d_indices = cuda.to_device(indices)
                d_distances = cuda.to_device(distances)

                find_nearest_neighbors_kernel[blocks, threads_per_block](
                    d_current_source, d_target, d_indices, d_distances
                )

                indices = d_indices.copy_to_host()
                distances = d_distances.copy_to_host()
            except Exception as e:
                print(f"CUDA error during nearest neighbor search: {e}")
                print("Falling back to CPU for this operation")
                indices, distances = find_nearest_neighbors_cpu(current_source, target_points)
        else:
            # CPU version
            indices, distances = find_nearest_neighbors_cpu(current_source, target_points)

        # Compute mean error
        mean_error = np.mean(distances)

        print(f"Iteration {iteration + 1:2d}: Mean error = {mean_error:.6f}")

        # Check convergence
        if abs(prev_error - mean_error) < tolerance:
            print(f"Converged at iteration {iteration + 1}")
            break

        prev_error = mean_error

        # Get matched point pairs
        source_matched = current_source
        target_matched = target_points[indices]

        # Compute transformation using SVD (CPU)
        delta_transformation = compute_transformation_svd(source_matched, target_matched)

        # Update cumulative transformation
        transformation = delta_transformation @ transformation

        # Transform source points
        if use_cuda:
            try:
                # CUDA version
                d_transformation = cuda.to_device(delta_transformation.astype(np.float32))
                d_output = cuda.device_array((n_source, 3), dtype=np.float32)

                transform_points_kernel[blocks, threads_per_block](
                    d_current_source, d_transformation, d_output
                )

                current_source = d_output.copy_to_host()
            except Exception as e:
                print(f"CUDA error during transformation: {e}")
                print("Falling back to CPU for this operation")
                current_source = transform_points_cpu(current_source, delta_transformation)
        else:
            # CPU version
            current_source = transform_points_cpu(current_source, delta_transformation)

    print(f"\nICP refinement completed!")
    print(f"Final mean error: {mean_error:.6f}")
    print(f"\nFinal transformation matrix:")
    print(transformation)

    return transformation


def draw_registration_result(source, target, transformation):
    """
    Visualize the registration result

    Args:
        source: Source point cloud
        target: Target point cloud
        transformation: 4x4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[-0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


# Main execution
if __name__ == "__main__":
    # Load demo point clouds
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    print("=" * 60)
    print("ICP Point Cloud Registration")
    print("=" * 60)
    print(f"Source: {len(source.points)} points")
    print(f"Target: {len(target.points)} points")

    # Perform ICP (will auto-detect CUDA or use CPU)
    transformation = icp_cuda(source, target, max_iterations=50, tolerance=1e-6)

    print("\n" + "=" * 60)
    print("Visualizing registration result...")

    # Visualize result
    draw_registration_result(source, target, transformation)