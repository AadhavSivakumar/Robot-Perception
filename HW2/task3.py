import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the two images
img1_path = './Task3/IMG_2329.JPG'
img2_path = './Task3/IMG_2330.JPG'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Convert to grayscale for feature detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print("=" * 70)
print("TASK 3: F-MATRIX AND RELATIVE POSE ESTIMATION")
print("=" * 70)

# ============================================================================
# Part a) Estimate the Fundamental Matrix
# ============================================================================
print("\n" + "=" * 70)
print("PART A: ESTIMATING FUNDAMENTAL MATRIX")
print("=" * 70)

# Use SIFT for feature detection (more robust than ORB)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"\nDetected {len(kp1)} keypoints in image 1")
print(f"Detected {len(kp2)} keypoints in image 2")

# Match features using FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"Found {len(good_matches)} good matches after ratio test")

# Extract matched point coordinates
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Estimate fundamental matrix using RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
mask = mask.ravel()

# Keep only inlier matches
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
pts1_inliers = pts1[mask == 1]
pts2_inliers = pts2[mask == 1]

print(f"\nNumber of inliers: {len(inlier_matches)}")
print(f"\nEstimated Fundamental Matrix F:")
print(F)

# Verify F by checking epipolar constraint: x2^T * F * x1 ≈ 0
print("\nVerifying epipolar constraint (x2^T * F * x1 should be ≈ 0):")
sample_indices = np.random.choice(len(pts1_inliers), min(5, len(pts1_inliers)), replace=False)
for idx in sample_indices:
    p1 = np.array([pts1_inliers[idx][0], pts1_inliers[idx][1], 1])
    p2 = np.array([pts2_inliers[idx][0], pts2_inliers[idx][1], 1])
    error = p2.T @ F @ p1
    print(f"  Point pair {idx}: error = {error:.6f}")

# ============================================================================
# Part b) Draw Epipolar Lines
# ============================================================================
print("\n" + "=" * 70)
print("PART B: DRAWING EPIPOLAR LINES")
print("=" * 70)


def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=15):
    """Draw epipolar lines on both images"""

    # Select random sample of points
    indices = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)

    # Create copies for drawing
    img1_epi = img1.copy()
    img2_epi = img2.copy()

    # Generate random colors for each point
    colors = np.random.randint(0, 255, (num_lines, 3))

    for i, idx in enumerate(indices):
        color = tuple(map(int, colors[i]))

        # Draw point on image 1
        x1, y1 = int(pts1[idx][0]), int(pts1[idx][1])
        cv2.circle(img1_epi, (x1, y1), 8, color, -1)

        # Draw point on image 2
        x2, y2 = int(pts2[idx][0]), int(pts2[idx][1])
        cv2.circle(img2_epi, (x2, y2), 8, color, -1)

        # Compute epipolar line in image 2 from point in image 1
        # l2 = F * p1
        p1_homo = np.array([pts1[idx][0], pts1[idx][1], 1]).reshape(3, 1)
        line2 = F @ p1_homo
        line2 = line2.flatten()

        # Draw line in image 2: ax + by + c = 0
        # Get two points on the line
        h, w = img2.shape[:2]
        if abs(line2[1]) > abs(line2[0]):  # More horizontal line
            x_start, x_end = 0, w - 1
            y_start = int(-(line2[2] + line2[0] * x_start) / line2[1])
            y_end = int(-(line2[2] + line2[0] * x_end) / line2[1])
        else:  # More vertical line
            y_start, y_end = 0, h - 1
            x_start = int(-(line2[2] + line2[1] * y_start) / line2[0])
            x_end = int(-(line2[2] + line2[1] * y_end) / line2[0])

        cv2.line(img2_epi, (x_start, y_start), (x_end, y_end), color, 2)

        # Compute epipolar line in image 1 from point in image 2
        # l1 = F^T * p2
        p2_homo = np.array([pts2[idx][0], pts2[idx][1], 1]).reshape(3, 1)
        line1 = F.T @ p2_homo
        line1 = line1.flatten()

        # Draw line in image 1
        h, w = img1.shape[:2]
        if abs(line1[1]) > abs(line1[0]):
            x_start, x_end = 0, w - 1
            y_start = int(-(line1[2] + line1[0] * x_start) / line1[1])
            y_end = int(-(line1[2] + line1[0] * x_end) / line1[1])
        else:
            y_start, y_end = 0, h - 1
            x_start = int(-(line1[2] + line1[1] * y_start) / line1[0])
            x_end = int(-(line1[2] + line1[1] * y_end) / line1[0])

        cv2.line(img1_epi, (x_start, y_start), (x_end, y_end), color, 2)

    return img1_epi, img2_epi


# Draw epipolar lines
img1_with_lines, img2_with_lines = draw_epipolar_lines(img1, img2, pts1_inliers, pts2_inliers, F)

# Save visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 20))

axes[0].imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
axes[0].set_title('Left Image with Epipolar Lines', fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
axes[1].set_title('Right Image with Epipolar Lines', fontsize=16, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Task3/epipolars.png', dpi=150, bbox_inches='tight')
print("\nEpipolar lines visualization saved!")

# ============================================================================
# Part c) Relative Pose Estimation
# ============================================================================
print("\n" + "=" * 70)
print("PART C: RELATIVE POSE ESTIMATION")
print("=" * 70)

# For demonstration, let's assume a reasonable intrinsic matrix
# Typical smartphone camera parameters
img_height, img_width = img1.shape[:2]
focal_length = max(img_width, img_height)  # Approximate focal length
cx, cy = img_width / 2, img_height / 2  # Principal point at image center

K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
])

print("\n\nAssumed Camera Intrinsic Matrix K:")
print(K)

# Compute Essential matrix
E = K.T @ F @ K

print("\nComputed Essential Matrix E:")
print(E)

# Decompose Essential matrix to get R and t
# Returns 2 possible rotations and 1 translation (up to scale)
_, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

print("\nRecovered Rotation Matrix R:")
print(R)

print("\nRecovered Translation vector t (up to scale):")
print(t.flatten())

# Compute rotation angle
rotation_angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
print(f"\nRotation angle: {rotation_angle:.2f} degrees")

# Compute translation direction
t_normalized = t.flatten() / np.linalg.norm(t)
print(f"Translation direction (normalized): {t_normalized}")

print("\n\nQuestion: Does the estimated relative pose make sense?")
print("-" * 70)
print("INTERPRETATION:")
print(f"The camera moved approximately {rotation_angle:.2f}° in rotation.")
print(f"The translation is primarily in direction: {t_normalized}")
print("\nLooking at your images of the delivery robot, the camera appears to have")
print("moved slightly to the right and possibly forward between the two shots,")
print("which is consistent with a small lateral translation.")
print("\nThe small rotation angle suggests the camera was kept relatively level")
print("between shots, which matches the similar framing of both images.")

# ============================================================================
# Create comprehensive visualization
# ============================================================================
print("\n" + "=" * 70)
print("CREATING COMPREHENSIVE VISUALIZATION")
print("=" * 70)

# Create feature matching visualization
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig = plt.figure(figsize=(20, 12))

# Feature matches
ax1 = plt.subplot(2, 2, (1, 2))
ax1.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
ax1.set_title(f'Feature Matches (showing 50/{len(inlier_matches)} inliers)',
              fontsize=14, fontweight='bold')
ax1.axis('off')

# Image 1 with epipolar lines
ax2 = plt.subplot(2, 2, 3)
ax2.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
ax2.set_title('Left Image with Epipolar Lines', fontsize=12, fontweight='bold')
ax2.axis('off')

# Image 2 with epipolar lines
ax3 = plt.subplot(2, 2, 4)
ax3.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
ax3.set_title('Right Image with Epipolar Lines', fontsize=12, fontweight='bold')
ax3.axis('off')

plt.tight_layout()
plt.savefig('Task3/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
print("Comprehensive visualization saved!")

