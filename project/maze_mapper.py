import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

class MazeMapper:
    def __init__(self, images_folder, grid_size=100, cell_size=20):
        """
        Initialize the maze mapper for a grid-based maze.

        Args:
            images_folder: Path to folder containing robot POV images
            grid_size: Size of the grid (number of cells)
            cell_size: Size of each cell in pixels
        """
        self.images_folder = Path(images_folder)
        self.grid_size = grid_size
        self.cell_size = cell_size

        # Grid-based occupancy map: 0=unknown, 1=free, 2=wall
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Wall grids for each direction
        self.horizontal_walls = np.zeros((grid_size + 1, grid_size), dtype=np.uint8)
        self.vertical_walls = np.zeros((grid_size, grid_size + 1), dtype=np.uint8)

        # Robot state (grid coordinates)
        self.robot_grid_x = grid_size // 2
        self.robot_grid_y = grid_size // 2
        self.robot_dir = 0  # 0=North, 1=East, 2=South, 3=West

        # Movement tracking
        self.prev_frame = None
        self.trajectory = [(self.robot_grid_x, self.robot_grid_y)]

        # Direction vectors for grid movement
        self.dir_vectors = {
            0: (0, -1),  # North (up)
            1: (1, 0),   # East (right)
            2: (0, 1),   # South (down)
            3: (-1, 0)   # West (left)
        }

    def get_sorted_image_files(self):
        """Get list of image files sorted by number."""
        image_files = list(self.images_folder.glob("*.jpg"))
        # Sort by numeric value in filename
        image_files.sort(key=lambda x: int(x.stem))
        return image_files

    def detect_walls(self, frame):
        """
        Detect walls in the current frame.
        Returns: left_wall, front_wall, right_wall (boolean)
        """
        h, w = frame.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Define regions for wall detection
        # Left: left third of image
        # Front: lower center portion (walls appear closer at bottom)
        # Right: right third of image
        left_region = edges[:, :w//3]
        center_region = edges[h//2:, w//3:2*w//3]  # Lower half center for front wall
        right_region = edges[:, 2*w//3:]

        # Check wall presence based on edge density
        left_wall = np.sum(left_region) > 8000
        front_wall = np.sum(center_region) > 10000
        right_wall = np.sum(right_region) > 8000

        return left_wall, front_wall, right_wall

    def estimate_motion(self, curr_frame):
        """
        Estimate robot motion between frames using optical flow.
        Returns: (moved_forward, turned_left, turned_right)
        """
        if self.prev_frame is None:
            return False, False, False

        # Convert to grayscale
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        h, w = flow.shape[:2]

        # Analyze flow patterns
        center_flow = flow[h//3:2*h//3, w//3:2*w//3]
        avg_vertical_flow = np.mean(center_flow[:, :, 1])

        left_flow = np.mean(flow[:, :w//3, 0])
        right_flow = np.mean(flow[:, 2*w//3:, 0])
        horizontal_diff = right_flow - left_flow

        # Detect discrete movements
        moved_forward = avg_vertical_flow < -2.0  # Negative flow = forward
        turned_left = horizontal_diff < -3.0      # Left turn pattern
        turned_right = horizontal_diff > 3.0      # Right turn pattern

        return moved_forward, turned_left, turned_right

    def update_map(self, left_wall, front_wall, right_wall):
        """Update grid-based map with detected walls."""
        x, y = self.robot_grid_x, self.robot_grid_y

        # Mark current cell as free
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1

        # Add walls based on robot direction
        # Direction: 0=North, 1=East, 2=South, 3=West

        if self.robot_dir == 0:  # Facing North
            if left_wall and x > 0:
                self.vertical_walls[y, x] = 1  # Wall to the left (West)
            if front_wall and y > 0:
                self.horizontal_walls[y, x] = 1  # Wall in front (North)
            if right_wall and x < self.grid_size:
                self.vertical_walls[y, x + 1] = 1  # Wall to the right (East)

        elif self.robot_dir == 1:  # Facing East
            if left_wall and y > 0:
                self.horizontal_walls[y, x] = 1  # Wall to the left (North)
            if front_wall and x < self.grid_size - 1:
                self.vertical_walls[y, x + 1] = 1  # Wall in front (East)
            if right_wall and y < self.grid_size:
                self.horizontal_walls[y + 1, x] = 1  # Wall to the right (South)

        elif self.robot_dir == 2:  # Facing South
            if left_wall and x < self.grid_size:
                self.vertical_walls[y, x + 1] = 1  # Wall to the left (East)
            if front_wall and y < self.grid_size:
                self.horizontal_walls[y + 1, x] = 1  # Wall in front (South)
            if right_wall and x > 0:
                self.vertical_walls[y, x] = 1  # Wall to the right (West)

        elif self.robot_dir == 3:  # Facing West
            if left_wall and y < self.grid_size:
                self.horizontal_walls[y + 1, x] = 1  # Wall to the left (South)
            if front_wall and x > 0:
                self.vertical_walls[y, x] = 1  # Wall in front (West)
            if right_wall and y > 0:
                self.horizontal_walls[y, x] = 1  # Wall to the right (North)

    def update_robot_pose(self, moved_forward, turned_left, turned_right):
        """Update robot position and orientation with discrete movements."""
        # Handle rotation first
        if turned_left:
            self.robot_dir = (self.robot_dir - 1) % 4
        elif turned_right:
            self.robot_dir = (self.robot_dir + 1) % 4

        # Handle forward movement
        if moved_forward:
            dx, dy = self.dir_vectors[self.robot_dir]
            new_x = self.robot_grid_x + dx
            new_y = self.robot_grid_y + dy

            # Only move if within bounds
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                self.robot_grid_x = new_x
                self.robot_grid_y = new_y
                self.trajectory.append((self.robot_grid_x, self.robot_grid_y))

    def process_images(self, skip_frames=2):
        """
        Process all images to build the map.

        Args:
            skip_frames: Process every Nth frame to speed up
        """
        image_files = self.get_sorted_image_files()
        print(f"Found {len(image_files)} images")

        for i, img_path in enumerate(image_files):
            if i % skip_frames != 0:
                continue

            if i % 100 == 0:
                print(f"Processing image {i}/{len(image_files)}")

            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Detect walls
            left_wall, front_wall, right_wall = self.detect_walls(frame)

            # Estimate motion
            moved_forward, turned_left, turned_right = self.estimate_motion(frame)

            # Update robot pose
            self.update_robot_pose(moved_forward, turned_left, turned_right)

            # Update map with wall detections
            self.update_map(left_wall, front_wall, right_wall)

            # Store current frame for next iteration
            self.prev_frame = frame.copy()

        print("Processing complete!")

    def visualize_map(self, output_path="maze_map.png"):
        """Create and save visualization of the grid-based maze map."""
        # Create image with proper size
        img_size = self.grid_size * self.cell_size
        map_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        # Draw grid cells
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                x_start = x * self.cell_size
                y_start = y * self.cell_size

                if self.grid[y, x] == 1:  # Free space
                    cv2.rectangle(map_img,
                                (x_start, y_start),
                                (x_start + self.cell_size, y_start + self.cell_size),
                                (255, 255, 255), -1)
                else:  # Unknown space
                    cv2.rectangle(map_img,
                                (x_start, y_start),
                                (x_start + self.cell_size, y_start + self.cell_size),
                                (220, 220, 220), -1)

        # Draw horizontal walls (walls between rows)
        for y in range(self.horizontal_walls.shape[0]):
            for x in range(self.horizontal_walls.shape[1]):
                if self.horizontal_walls[y, x] == 1:
                    x_start = x * self.cell_size
                    y_pos = y * self.cell_size
                    cv2.line(map_img,
                           (x_start, y_pos),
                           (x_start + self.cell_size, y_pos),
                           (0, 0, 0), 3)

        # Draw vertical walls (walls between columns)
        for y in range(self.vertical_walls.shape[0]):
            for x in range(self.vertical_walls.shape[1]):
                if self.vertical_walls[y, x] == 1:
                    x_pos = x * self.cell_size
                    y_start = y * self.cell_size
                    cv2.line(map_img,
                           (x_pos, y_start),
                           (x_pos, y_start + self.cell_size),
                           (0, 0, 0), 3)

        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                x1, y1 = self.trajectory[i]
                x2, y2 = self.trajectory[i + 1]
                pt1 = (x1 * self.cell_size + self.cell_size // 2,
                       y1 * self.cell_size + self.cell_size // 2)
                pt2 = (x2 * self.cell_size + self.cell_size // 2,
                       y2 * self.cell_size + self.cell_size // 2)
                cv2.line(map_img, pt1, pt2, (255, 100, 100), 2)

        # Mark start position (green)
        if len(self.trajectory) > 0:
            x, y = self.trajectory[0]
            center = (x * self.cell_size + self.cell_size // 2,
                     y * self.cell_size + self.cell_size // 2)
            cv2.circle(map_img, center, self.cell_size // 3, (0, 255, 0), -1)

        # Mark end position (red)
        if len(self.trajectory) > 0:
            x, y = self.trajectory[-1]
            center = (x * self.cell_size + self.cell_size // 2,
                     y * self.cell_size + self.cell_size // 2)
            cv2.circle(map_img, center, self.cell_size // 3, (0, 0, 255), -1)

        # Save the map
        cv2.imwrite(output_path, map_img)
        print(f"Map saved to {output_path}")

        # Create matplotlib visualization
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))
        plt.title("Grid-Based Maze Map\n(Green=Start, Red=End, Pink=Path, Black=Walls)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_annotated.png'), dpi=150)
        print(f"Annotated map saved to {output_path.replace('.png', '_annotated.png')}")

        return map_img


def main():
    """Main function to run the maze mapper."""
    # Set up paths
    images_folder = "images_subsample"

    # Create mapper with grid-based parameters
    print("Initializing Grid-Based Maze Mapper...")
    print("Maze assumptions: 90-degree walls only, grid-based layout")
    mapper = MazeMapper(images_folder, grid_size=100, cell_size=20)

    # Process images
    print("\nProcessing images...")
    mapper.process_images(skip_frames=2)  # Process every 2nd frame

    # Generate and save map
    print("\nGenerating map visualization...")
    mapper.visualize_map("maze_map_grid.png")

    print("\n" + "="*50)
    print("Done! Check the output files:")
    print("  - maze_map_grid.png")
    print("  - maze_map_grid_annotated.png")
    print("="*50)


if __name__ == "__main__":
    main()
