import open3d as o3d
import numpy as np
from typing import Tuple


def ransac_plane_fitting(pcd: o3d.geometry.PointCloud,
                         distance_threshold: float = 0.01,
                         num_iterations: int = 1000,
                         num_samples: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom RANSAC implementation for plane fitting in 3D point clouds.

    Parameters:
    -----------
    pcd : o3d.geometry.PointCloud
        Input point cloud
    distance_threshold : float
        Maximum distance a point can be from the plane to be considered an inlier
    num_iterations : int
        Number of RANSAC iterations
    num_samples : int
        Number of points to sample for plane estimation (minimum 3 for a plane)

    Returns:
    --------
    best_plane : np.ndarray
        Plane equation coefficients [a, b, c, d] where ax + by + cz + d = 0
    best_inliers : np.ndarray
        Indices of inlier points
    """

    points = np.asarray(pcd.points)
    n_points = len(points)

    if n_points < num_samples:
        raise ValueError(f"Point cloud has {n_points} points, need at least {num_samples}")

    best_plane = None
    best_inliers = np.array([])
    best_num_inliers = 0

    for iteration in range(num_iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(n_points, num_samples, replace=False)
        sample_points = points[sample_indices]

        # Compute plane from 3 points
        # Using two vectors in the plane
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]

        # Normal vector (cross product)
        normal = np.cross(v1, v2)

        # Skip if points are collinear (normal vector is zero)
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-6:
            continue

        # Normalize the normal vector
        normal = normal / normal_length

        # Plane equation: ax + by + cz + d = 0
        # where (a, b, c) is the normal vector
        # d = -(ax0 + by0 + cz0) for point (x0, y0, z0) on the plane
        d = -np.dot(normal, sample_points[0])

        plane = np.append(normal, d)

        # Calculate distances from all points to the plane
        # Distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
        # Since normal is already normalized, sqrt(a^2 + b^2 + c^2) = 1
        distances = np.abs(np.dot(points, normal) + d)

        # Find inliers
        inlier_mask = distances < distance_threshold
        inliers = np.where(inlier_mask)[0]
        num_inliers = len(inliers)

        # Update best plane if this one has more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_plane = plane
            best_inliers = inliers

    print(f"RANSAC completed: {best_num_inliers}/{n_points} inliers found")
    print(
        f"Plane equation: {best_plane[0]:.4f}x + {best_plane[1]:.4f}y + {best_plane[2]:.4f}z + {best_plane[3]:.4f} = 0")

    return best_plane, best_inliers


def visualize_plane_fitting(pcd: o3d.geometry.PointCloud,
                            plane: np.ndarray,
                            inliers: np.ndarray):
    """
    Visualize the point cloud with the fitted plane highlighted.

    Parameters:
    -----------
    pcd : o3d.geometry.PointCloud
        Input point cloud
    plane : np.ndarray
        Plane equation coefficients [a, b, c, d]
    inliers : np.ndarray
        Indices of inlier points
    """

    # Create a copy of the point cloud
    pcd_with_plane = o3d.geometry.PointCloud(pcd)

    # Color all points gray
    colors = np.asarray(pcd_with_plane.colors)
    if len(colors) == 0:
        colors = np.ones((len(pcd_with_plane.points), 3)) * 0.5
    else:
        colors = np.ones_like(colors) * 0.5

    # Color inliers red
    colors[inliers] = [1, 0, 0]  # Red for plane
    pcd_with_plane.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd_with_plane],
                                      zoom=1,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


# Demo usage
if __name__ == "__main__":
    # Read demo point cloud provided by Open3D
    pcd_point_cloud = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

    print("Original point cloud:")
    print(pcd)

    # Apply custom RANSAC
    plane, inliers = ransac_plane_fitting(pcd,
                                          distance_threshold=0.02,
                                          num_iterations=500,
                                          num_samples=3)

    # Visualize results
    visualize_plane_fitting(pcd, plane, inliers)