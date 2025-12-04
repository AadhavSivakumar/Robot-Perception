"""
ICP (Iterative Closest Point) Implementation for Point Cloud Registration
Task 2: Parts a) and b)

Author: Student
Course: Computer Vision / 3D Point Cloud Processing

This script implements a custom ICP algorithm and applies it to:
- Part a) Open3D Demo Point Clouds
- Part b) KITTI Dataset Point Clouds

Requirements: pip install open3d numpy matplotlib
"""

import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# CUSTOM ICP IMPLEMENTATION
# ============================================================================

def icp_algorithm(source, target, max_iterations=25, tolerance=1e-8,
                  initial_transform=None, max_correspondence_dist=None, verbose=True):
    """
    Custom implementation of the Iterative Closest Point (ICP) algorithm.

    The ICP algorithm iteratively refines the transformation between two point clouds
    by repeatedly:
    1. Finding correspondences (closest points)
    2. Computing the optimal rigid transformation
    3. Applying the transformation
    4. Checking for convergence

    Parameters:
    -----------
    source : open3d.geometry.PointCloud
        Source point cloud to be transformed
    target : open3d.geometry.PointCloud
        Target point cloud (reference)
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold for error change
    initial_transform : np.ndarray (4x4)
        Optional initial transformation matrix
    max_correspondence_dist : float
        Maximum distance for valid point correspondences
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    transformation : np.ndarray (4x4)
        Final 4x4 homogeneous transformation matrix
    errors_history : list
        List of error values at each iteration
    """

    # Initialize transformation
    if initial_transform is not None:
        transformation = initial_transform.copy()
    else:
        transformation = np.eye(4)

    # Get source points as numpy array
    source_points = np.asarray(source.points).copy()

    # Apply initial transform if provided
    if initial_transform is not None:
        source_points = (transformation[:3, :3] @ source_points.T).T + transformation[:3, 3]

    # Build KD-tree for target for efficient nearest neighbor search
    target_tree = o3d.geometry.KDTreeFlann(target)
    target_points = np.asarray(target.points)

    # Auto-determine max correspondence distance if not provided
    if max_correspondence_dist is None:
        source_extent = np.max(source_points, axis=0) - np.min(source_points, axis=0)
        max_correspondence_dist = np.max(source_extent) * 0.1

    prev_error = float('inf')
    errors_history = []

    if verbose:
        print("=" * 70)
        print("ICP Registration Progress")
        print("=" * 70)
        print(f"Max correspondence distance: {max_correspondence_dist:.4f}")
        print(f"{'Iter':<6} {'Error':<14} {'Δ Error':<14} {'Inliers':<12} {'Status'}")
        print("-" * 70)

    for iteration in range(max_iterations):
        # ===== STEP 1: Find Correspondences =====
        # For each source point, find the closest point in target
        correspondences_src = []
        correspondences_tgt = []

        for i, point in enumerate(source_points):
            [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
            # Only accept correspondences within max distance (outlier rejection)
            if np.sqrt(dist[0]) < max_correspondence_dist:
                correspondences_src.append(i)
                correspondences_tgt.append(idx[0])

        # Check if we have enough correspondences
        num_correspondences = len(correspondences_src)
        if num_correspondences < 10:
            if verbose:
                print("WARNING: Too few correspondences found!")
            break

        # Get corresponding points
        src_corr = source_points[correspondences_src]
        tgt_corr = target_points[correspondences_tgt]

        # ===== STEP 2: Compute Centroids =====
        source_centroid = np.mean(src_corr, axis=0)
        target_centroid = np.mean(tgt_corr, axis=0)

        # ===== STEP 3: Center the Point Clouds =====
        source_centered = src_corr - source_centroid
        target_centered = tgt_corr - target_centroid

        # ===== STEP 4: Compute Cross-Covariance Matrix H =====
        H = source_centered.T @ target_centered

        # ===== STEP 5: SVD Decomposition =====
        U, S, Vt = np.linalg.svd(H)

        # ===== STEP 6: Compute Rotation Matrix =====
        R = Vt.T @ U.T

        # Handle reflection case (ensure proper rotation matrix)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # ===== STEP 7: Compute Translation =====
        t = target_centroid - R @ source_centroid

        # ===== STEP 8: Apply Transformation =====
        source_points = (R @ source_points.T).T + t

        # Update cumulative transformation
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        transformation = T_iter @ transformation

        # ===== STEP 9: Compute Error =====
        new_src_corr = source_points[correspondences_src]
        error = np.mean(np.linalg.norm(new_src_corr - tgt_corr, axis=1))
        error_change = prev_error - error
        errors_history.append(error)

        # Progress indicator
        if verbose:
            progress = (iteration + 1) / max_iterations
            bar_length = 15
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)

            if error_change > 0:
                status = "↓ Improving"
            elif error_change < 0:
                status = "↑ Worsening"
            else:
                status = "→ Stable"

            delta_str = "N/A" if prev_error == float('inf') else f"{error_change:+.8f}"
            inlier_ratio = num_correspondences / len(source_points)
            inlier_str = f"{num_correspondences} ({100 * inlier_ratio:.1f}%)"

            print(f"{iteration + 1:<6} {error:<14.8f} {delta_str:<14} {inlier_str:<12} {status}")

            if (iteration + 1) % 25 == 0:
                print(f"Progress: [{bar}] {progress * 100:.1f}%")

        # ===== STEP 10: Check Convergence =====
        if abs(error_change) < tolerance:
            if verbose:
                print("-" * 70)
                print(f"✓ CONVERGED at iteration {iteration + 1}")
                print(f"  Final error: {error:.8f}")
                print(f"  Final inliers: {num_correspondences} / {len(source_points)}")
                print("=" * 70)
            break

        prev_error = error
    else:
        if verbose:
            print("-" * 70)
            print(f"✗ Reached maximum iterations ({max_iterations})")
            print(f"  Final error: {prev_error:.8f}")
            print("=" * 70)

    return transformation, errors_history


def draw_registration_result(source, target, transformation, title="Registration Result"):
    """
    Visualize registration result using Open3D visualization.

    Source is shown in orange/yellow, target in blue.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange/Yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name=title,
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def save_registration_visualization(source, target, transformation, filename, title="Registration Result"):
    """
    Save registration result as an image using matplotlib.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)

    src_points = np.asarray(source_temp.points)
    tgt_points = np.asarray(target_temp.points)

    # Downsample for visualization if too many points
    max_points = 5000
    if len(src_points) > max_points:
        idx = np.random.choice(len(src_points), max_points, replace=False)
        src_points = src_points[idx]
    if len(tgt_points) > max_points:
        idx = np.random.choice(len(tgt_points), max_points, replace=False)
        tgt_points = tgt_points[idx]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2],
               c='orange', s=1, alpha=0.6, label='Source (transformed)')
    ax.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2],
               c='blue', s=1, alpha=0.6, label='Target')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {filename}")


def compute_alignment_error(source, target, transformation):
    """Compute mean point-to-point distance after alignment."""
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)

    src_pts = np.asarray(source_temp.points)
    tgt_tree = o3d.geometry.KDTreeFlann(target)

    distances = []
    for pt in src_pts:
        [_, _, dist] = tgt_tree.search_knn_vector_3d(pt, 1)
        distances.append(np.sqrt(dist[0]))

    return np.mean(distances), np.std(distances), np.median(distances)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # PART A: Demo Point Clouds
    # ========================================================================

    print("\n" + "=" * 80)
    print("PART A: Open3D Demo Point Clouds")
    print("=" * 80 + "\n")

    # Load demo point clouds
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source_a = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target_a = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    print(f"Source point cloud: {len(source_a.points)} points")
    print(f"Target point cloud: {len(target_a.points)} points")

    # Initial transform for demo point clouds
    initial_transform_a = np.array([
        [0.862, 0.011, -0.507, 0.5],
        [-0.139, 0.967, -0.215, 0.7],
        [0.487, 0.255, 0.835, -1.4],
        [0.0, 0.0, 0.0, 1.0]
    ])

    print("\nInitial Transform for Part a:")
    print(initial_transform_a)

    # Run ICP for Part a
    print("\nRunning ICP on Demo Point Clouds...\n")
    transform_a, errors_a = icp_algorithm(
        source_a, target_a,
        max_iterations=25,
        tolerance=1e-8,
        initial_transform=initial_transform_a,
        max_correspondence_dist=0.05,
        verbose=True
    )

    print("\nFinal Transformation Matrix (Part a):")
    print(transform_a)

    # Visualize Part a result
    print("\nOpening visualization for Part a...")
    draw_registration_result(source_a, target_a, transform_a,
                             "Part a) Demo Point Clouds - ICP Result")

    # Save visualization
    save_registration_visualization(source_a, target_a, transform_a,
                                    "part_a_result.png",
                                    "Part a) Demo Point Clouds - ICP Registration Result")

    # ========================================================================
    # PART B: KITTI Dataset Point Clouds
    # ========================================================================

    print("\n" + "=" * 80)
    print("PART B: KITTI Dataset Point Clouds")
    print("=" * 80 + "\n")

    # Load KITTI point clouds
    # UPDATE THESE PATHS to match your file locations
    source_b = o3d.io.read_point_cloud("data/Task2/kitti_frame1.pcd")
    target_b = o3d.io.read_point_cloud("data/Task2/kitti_frame2.pcd")

    print(f"Source point cloud (KITTI frame1): {len(source_b.points)} points")
    print(f"Target point cloud (KITTI frame2): {len(target_b.points)} points")

    # Analyze point cloud characteristics
    src_points_b = np.asarray(source_b.points)
    tgt_points_b = np.asarray(target_b.points)

    print(f"\nSource extent: X[{src_points_b[:, 0].min():.2f}, {src_points_b[:, 0].max():.2f}], "
          f"Y[{src_points_b[:, 1].min():.2f}, {src_points_b[:, 1].max():.2f}], "
          f"Z[{src_points_b[:, 2].min():.2f}, {src_points_b[:, 2].max():.2f}]")
    print(f"Target extent: X[{tgt_points_b[:, 0].min():.2f}, {tgt_points_b[:, 0].max():.2f}], "
          f"Y[{tgt_points_b[:, 1].min():.2f}, {tgt_points_b[:, 1].max():.2f}], "
          f"Z[{tgt_points_b[:, 2].min():.2f}, {tgt_points_b[:, 2].max():.2f}]")

    # For KITTI sequential frames, start with identity transform
    initial_transform_b = np.eye(4)

    print("\nInitial Transform for Part b (Identity):")
    print(initial_transform_b)

    # Run ICP for Part b
    print("\nRunning ICP on KITTI Point Clouds...\n")
    transform_b, errors_b = icp_algorithm(
        source_b, target_b,
        max_iterations=25,
        tolerance=1e-8,
        initial_transform=initial_transform_b,
        max_correspondence_dist=2.0,  # Larger for outdoor LiDAR
        verbose=True
    )

    print("\nFinal Transformation Matrix (Part b):")
    print(transform_b)

    # Visualize Part b result
    print("\nOpening visualization for Part b...")
    draw_registration_result(source_b, target_b, transform_b,
                             "Part b) KITTI Point Clouds - ICP Result")

    # Save visualization
    save_registration_visualization(source_b, target_b, transform_b,
                                    "part_b_result.png",
                                    "Part b) KITTI Point Clouds - ICP Registration Result")

    # ========================================================================
    # COMPARISON AND ANALYSIS
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON: Part a vs Part b")
    print("=" * 80 + "\n")

    # Plot convergence comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(errors_a) + 1), errors_a, 'b-o', markersize=3)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Mean Error')
    axes[0].set_title('Part a) Demo Point Clouds - Convergence')
    axes[0].grid(True, alpha=0.3)
    if errors_a:
        axes[0].set_yscale('log')

    axes[1].plot(range(1, len(errors_b) + 1), errors_b, 'r-o', markersize=3)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Mean Error')
    axes[1].set_title('Part b) KITTI Point Clouds - Convergence')
    axes[1].grid(True, alpha=0.3)
    if errors_b:
        axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved convergence comparison to convergence_comparison.png")

    # Compute alignment quality metrics
    mean_a, std_a, median_a = compute_alignment_error(source_a, target_a, transform_a)
    mean_b, std_b, median_b = compute_alignment_error(source_b, target_b, transform_b)

    print("\n" + "=" * 60)
    print("FINAL ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n--- Part a (Demo Point Clouds) ---")
    print(f"  Number of iterations: {len(errors_a)}")
    print(f"  Initial error: {errors_a[0]:.8f}" if errors_a else "  N/A")
    print(f"  Final error: {errors_a[-1]:.8f}" if errors_a else "  N/A")
    print(f"  Mean alignment distance: {mean_a:.6f}")
    print(f"  Source points: {len(source_a.points)}")
    print(f"  Target points: {len(target_a.points)}")

    print("\n--- Part b (KITTI Point Clouds) ---")
    print(f"  Number of iterations: {len(errors_b)}")
    print(f"  Initial error: {errors_b[0]:.8f}" if errors_b else "  N/A")
    print(f"  Final error: {errors_b[-1]:.8f}" if errors_b else "  N/A")
    print(f"  Mean alignment distance: {mean_b:.6f}")
    print(f"  Source points: {len(source_b.points)}")
    print(f"  Target points: {len(target_b.points)}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)