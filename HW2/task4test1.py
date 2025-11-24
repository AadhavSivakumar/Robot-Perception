"""
Object Tracking Implementation
==============================

This script implements multi-object tracking for video sequences using:
1. MOG2 Background Subtraction for object detection
2. IoU-based data association for tracking

Author: Claude
Date: November 2024

Requirements:
- opencv-python (pip install opencv-python)
- numpy (pip install numpy)

Usage:
    python object_tracking.py --input video.mp4 --output tracked_video.mp4

Or modify the paths in the main section at the bottom.
"""

import cv2
import numpy as np
import os
import argparse


class SimpleTracker:
    """
    Multi-object tracker using IoU-based data association.

    Maintains track identities across frames by matching new detections
    with existing tracks based on bounding box overlap (IoU).
    """

    def __init__(self, iou_threshold=0.2, max_age=10, min_hits=2):
        """
        Initialize the tracker.

        Args:
            iou_threshold (float): Minimum IoU to match detection with track
            max_age (int): Maximum frames to keep track without detection
            min_hits (int): Minimum detections before track is displayed
        """
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.colors = self._generate_colors(100)

    def _generate_colors(self, n):
        """Generate n distinct colors using HSV color space."""
        colors = []
        np.random.seed(42)
        for i in range(n):
            h = (i * 360 / n) % 360
            # Convert HSV to RGB
            c = 1
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = 0

            if 0 <= h < 60:
                r, g, b = c, x, 0
            elif 60 <= h < 120:
                r, g, b = x, c, 0
            elif 120 <= h < 180:
                r, g, b = 0, c, x
            elif 180 <= h < 240:
                r, g, b = 0, x, c
            elif 240 <= h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))

        return colors

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            box1: [x, y, width, height]
            box2: [x, y, width, height]

        Returns:
            float: IoU value between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: List of bounding boxes [x, y, w, h]

        Returns:
            dict: Dictionary of active tracks with their information
        """
        # Handle edge cases
        if len(self.tracks) == 0:
            # No existing tracks, create new ones for all detections
            for detection in detections:
                self.tracks[self.next_id] = {
                    'bbox': detection,
                    'age': 0,
                    'total_visible': 1,
                    'color': self.colors[self.next_id % len(self.colors)],
                    'time_since_update': 0
                }
                self.next_id += 1
            return self.tracks

        if len(detections) == 0:
            # No detections, age all tracks and remove old ones
            to_delete = []
            for track_id in self.tracks:
                self.tracks[track_id]['time_since_update'] += 1
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['time_since_update'] > self.max_age:
                    to_delete.append(track_id)

            for track_id in to_delete:
                del self.tracks[track_id]

            return self.tracks

        # Calculate IoU matrix between all tracks and detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for t, track_id in enumerate(track_ids):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(
                    self.tracks[track_id]['bbox'], detection
                )

        # Greedy matching: repeatedly select highest IoU pair
        matched_tracks = set()
        matched_detections = set()
        matches = []

        while True:
            if iou_matrix.size == 0:
                break

            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break

            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

            matches.append((track_ids[t_idx], d_idx))
            matched_tracks.add(track_ids[t_idx])
            matched_detections.add(d_idx)

            # Zero out matched row and column
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, d_idx] = 0

        # Update matched tracks
        for track_id, detection_idx in matches:
            self.tracks[track_id]['bbox'] = detections[detection_idx]
            self.tracks[track_id]['age'] += 1
            self.tracks[track_id]['total_visible'] += 1
            self.tracks[track_id]['time_since_update'] = 0

        # Create new tracks for unmatched detections
        for d_idx, detection in enumerate(detections):
            if d_idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection,
                    'age': 0,
                    'total_visible': 1,
                    'color': self.colors[self.next_id % len(self.colors)],
                    'time_since_update': 0
                }
                self.next_id += 1

        # Age and remove old unmatched tracks
        to_delete = []
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]['time_since_update'] += 1
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['time_since_update'] > self.max_age:
                    to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

        return self.tracks


def detect_objects_background_sub(frame, bg_subtractor, min_area=600):
    """
    Detect moving objects using background subtraction.

    Args:
        frame: Input video frame
        bg_subtractor: OpenCV background subtractor object
        min_area: Minimum area for valid detection (pixels)

    Returns:
        list: List of bounding boxes [x, y, w, h]
    """
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply threshold to remove shadows
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to bounding boxes
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very thin detections
            if w > 10 and h > 10:
                detections.append([x, y, w, h])

    return detections


def draw_tracks(frame, tracks, min_hits=2):
    """
    Draw tracking visualization on frame.

    Args:
        frame: Video frame to draw on
        tracks: Dictionary of track information
        min_hits: Minimum visibility count to display track

    Returns:
        frame: Frame with tracking visualization
    """
    for track_id, track in tracks.items():
        # Only draw confirmed tracks
        if track['total_visible'] < min_hits:
            continue

        x, y, w, h = [int(v) for v in track['bbox']]
        color = track['color']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Prepare ID label
        label = f"ID: {track_id}"

        # Calculate text size for background
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width + 10, y),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

        # Draw centroid marker
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 2)

    return frame


def process_video(input_path, output_path, min_detection_area=600):
    """
    Process video with object tracking.

    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        min_detection_area: Minimum area for object detection (pixels)
    """
    print("=" * 60)
    print("Object Tracking - Processing Video")
    print("=" * 60)

    # Open video
    print(f"\nOpening video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Frame rate: {fps} fps")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {total_frames / fps:.2f} seconds")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracker and background subtractor
    tracker = SimpleTracker(iou_threshold=0.2, max_age=10, min_hits=2)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=50,
        detectShadows=False
    )

    frame_count = 0

    print(f"\nProcessing frames...")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect objects using background subtraction
        detections = detect_objects_background_sub(frame, bg_subtractor, min_area=min_detection_area)

        # Update tracker with detections
        tracks = tracker.update(detections)

        # Draw tracking visualization
        output_frame = draw_tracks(frame.copy(), tracks, min_hits=2)

        # Add frame counter
        cv2.putText(
            output_frame,
            f"Frame: {frame_count}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Write processed frame
        out.write(output_frame)

        # Progress update every 50 frames
        if frame_count % 50 == 0 or frame_count == total_frames:
            percent = frame_count * 100 // total_frames
            print(f"Progress: {frame_count}/{total_frames} frames ({percent}%) - Active tracks: {len(tracks)}")

    # Cleanup
    cap.release()
    out.release()

    print("-" * 60)
    print(f"\n✓ Tracking complete!")
    print(f"  - Output saved to: {output_path}")
    print(f"  - Total unique objects tracked: {tracker.next_id}")
    print("=" * 60)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Multi-Object Tracking for Video Sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --input video.mp4 --output tracked.mp4
  python %(prog)s --input video.mp4 --output tracked.mp4 --min-area 800
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output video file')
    parser.add_argument('--min-area', type=int, default=600,
                        help='Minimum detection area in pixels (default: 600)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return

    # Process video
    process_video(args.input, args.output, args.min_area)


if __name__ == "__main__":
    # You can either use command-line arguments (uncomment line below)
    # main()

    # Or set paths directly here:
    input_video = "Task4/tracking.mp4"  # Change this to your input video
    output_video = "Task4/tracked_output.mp4"  # Change this to desired output path

    if os.path.exists(input_video):
        process_video(input_video, output_video)
    else:
        print(f"Error: Video file '{input_video}' not found!")
        print("Please update the input_video path in the script.")