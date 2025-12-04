import open3d as o3d
import numpy as np
import copy

demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])


def icp_algorithm(source, target, max_iterations=50, tolerance=1e-6):
    """
    Custom implementation of the Iterative Closest Point (ICP) algorithm.

    param: source - source point cloud
    param: target - target point cloud
    param: max_iterations - maximum number of iterations
    param: tolerance - convergence threshold

    return: 4x4 homogeneous transformation matrix
    """
    # Initialize transformation as identity matrix
    transformation = np.eye(4)

    # Get source points as numpy array
    source_points = np.asarray(source.points).copy()

    # Build KD-tree for target for efficient nearest neighbor search
    target_tree = o3d.geometry.KDTreeFlann(target)
    target_points = np.asarray(target.points)

    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Step 1: Find closest points in target for each source point
        correspondences = []
        for i, point in enumerate(source_points):
            # Query nearest neighbor
            [_, idx, _] = target_tree.search_knn_vector_3d(point, 1)
            correspondences.append(idx[0])

        # Get corresponding target points
        target_corr = target_points[correspondences]

        # Step 2: Compute centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_corr, axis=0)

        # Step 3: Center the point clouds
        source_centered = source_points - source_centroid
        target_centered = target_corr - target_centroid

        # Step 4: Compute cross-covariance matrix H
        H = source_centered.T @ target_centered

        # Step 5: SVD decomposition
        U, S, Vt = np.linalg.svd(H)

        # Step 6: Compute rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Step 7: Compute translation
        t = target_centroid - R @ source_centroid

        # Step 8: Apply transformation to source points
        source_points = (R @ source_points.T).T + t

        # Update cumulative transformation
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        transformation = T_iter @ transformation

        # Step 9: Compute mean squared error
        error = np.mean(np.linalg.norm(source_points - target_corr, axis=1))

        # Check for convergence
        if abs(prev_error - error) < tolerance:
            print(f"Converged at iteration {iteration + 1}")
            break

        prev_error = error

    return transformation


# Run ICP algorithm
transformation = icp_algorithm(source, target)
print("Final Transformation Matrix:")
print(transformation)


def draw_registration_result(source, target, transformation):
    """
    param: source - source point cloud
    param: target - target point cloud
    param: transformation - 4 X 4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


# Visualize result
draw_registration_result(source, target, transformation)