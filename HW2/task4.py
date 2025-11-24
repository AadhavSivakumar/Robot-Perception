"""
YOLO-Based Object Tracking
===========================

This implementation uses YOLO for object detection combined with
IoU-based tracking for maintaining object identities.

Advantages over background subtraction:
- Works with moving cameras
- Better detection accuracy
- Provides object class information
- More robust to lighting changes

Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    python yolo_tracking.py --input video.mp4 --output tracked.mp4
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os


class YOLOTracker:
    """
    Multi-object tracker using YOLO detections and IoU-based association.
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        """
        Initialize YOLO-based tracker.

        Args:
            iou_threshold: Minimum IoU to match detection with track
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is displayed
        """
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.colors = self._generate_colors(100)

    def _generate_colors(self, n):
        """Generate n distinct colors."""
        colors = []
        np.random.seed(42)
        for i in range(n):
            h = (i * 360 / n) % 360
            c = 1
            x = c * (1 - abs((h / 60) % 2 - 1))

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

            colors.append((int(r * 255), int(g * 255), int(b * 255)))

        return colors

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        """
        Update tracks with YOLO detections.

        Args:
            detections: List of [x, y, w, h, confidence, class_id, class_name]

        Returns:
            dict: Active tracks
        """
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks[self.next_id] = {
                    'bbox': detection[:4],
                    'confidence': detection[4],
                    'class_id': detection[5],
                    'class_name': detection[6],
                    'age': 0,
                    'total_visible': 1,
                    'time_since_update': 0,
                    'color': self.colors[self.next_id % len(self.colors)]
                }
                self.next_id += 1
            return self.tracks

        if len(detections) == 0:
            to_delete = []
            for track_id in self.tracks:
                self.tracks[track_id]['time_since_update'] += 1
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['time_since_update'] > self.max_age:
                    to_delete.append(track_id)

            for track_id in to_delete:
                del self.tracks[track_id]

            return self.tracks

        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for t, track_id in enumerate(track_ids):
            for d, detection in enumerate(detections):
                # Only match same class
                if self.tracks[track_id]['class_id'] == detection[5]:
                    iou_matrix[t, d] = self._calculate_iou(
                        self.tracks[track_id]['bbox'], detection[:4]
                    )

        # Greedy matching
        matched_tracks = set()
        matched_detections = set()

        while True:
            if iou_matrix.size == 0:
                break

            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break

            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

            matched_tracks.add(track_ids[t_idx])
            matched_detections.add(d_idx)

            # Update track
            detection = detections[d_idx]
            self.tracks[track_ids[t_idx]]['bbox'] = detection[:4]
            self.tracks[track_ids[t_idx]]['confidence'] = detection[4]
            self.tracks[track_ids[t_idx]]['age'] += 1
            self.tracks[track_ids[t_idx]]['total_visible'] += 1
            self.tracks[track_ids[t_idx]]['time_since_update'] = 0

            # Zero out
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, d_idx] = 0

        # Create new tracks
        for d_idx, detection in enumerate(detections):
            if d_idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection[:4],
                    'confidence': detection[4],
                    'class_id': detection[5],
                    'class_name': detection[6],
                    'age': 0,
                    'total_visible': 1,
                    'time_since_update': 0,
                    'color': self.colors[self.next_id % len(self.colors)]
                }
                self.next_id += 1

        # Age unmatched tracks
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


def draw_yolo_tracks(frame, tracks, min_hits=3):
    """
    Draw YOLO tracking visualization.

    Args:
        frame: Video frame
        tracks: Dictionary of tracks
        min_hits: Minimum hits to display

    Returns:
        frame: Annotated frame
    """
    for track_id, track in tracks.items():
        if track['total_visible'] < min_hits:
            continue

        x, y, w, h = [int(v) for v in track['bbox']]
        color = track['color']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Create label with ID and class
        label = f"ID:{track_id} {track['class_name']} {track['confidence']:.2f}"

        # Get text size
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (x, y - text_h - 10),
            (x + text_w + 10, y),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

        # Draw centroid
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 2)

    return frame


def process_video_yolo(input_path, output_path, model_name='yolov8n.pt',
                       conf_threshold=0.3, classes=None):
    """
    Process video with YOLO-based tracking.

    Args:
        input_path: Input video path
        output_path: Output video path
        model_name: YOLO model ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
        conf_threshold: Confidence threshold for detections
        classes: List of class IDs to track (None = all classes)
                 Common: [2, 3, 5, 7] for car, motorcycle, bus, truck
    """
    print("=" * 70)
    print("YOLO-Based Object Tracking")
    print("=" * 70)

    # Load YOLO model
    print(f"\nLoading YOLO model: {model_name}")
    model = YOLO(model_name)

    # Open video
    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open video")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f}s")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracker
    tracker = YOLOTracker(iou_threshold=0.3, max_age=30, min_hits=3)

    frame_count = 0

    print(f"\nProcessing with YOLO...")
    print("-" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame, conf=conf_threshold, classes=classes, verbose=False)

        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Convert to [x, y, w, h] format
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                detections.append([x, y, w, h, conf, cls, class_name])

        # Update tracker
        tracks = tracker.update(detections)

        # Draw visualization
        output_frame = draw_yolo_tracks(frame.copy(), tracks, min_hits=3)

        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(
            output_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Write frame
        out.write(output_frame)

        # Progress update
        if frame_count % 30 == 0 or frame_count == total_frames:
            percent = frame_count * 100 // total_frames
            print(f"Progress: {frame_count}/{total_frames} ({percent}%) | "
                  f"Detections: {len(detections)} | Active tracks: {len(tracks)}")

    cap.release()
    out.release()

    print("-" * 70)
    print(f"\n✓ Tracking complete!")
    print(f"  Output: {output_path}")
    print(f"  Total unique objects: {tracker.next_id}")
    print("=" * 70)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='YOLO-based Object Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track all objects
  python yolo_tracking.py --input video.mp4 --output tracked.mp4

  # Track only vehicles (cars, motorcycles, buses, trucks)
  python yolo_tracking.py --input video.mp4 --output tracked.mp4 --classes 2 3 5 7

  # Use larger model for better accuracy
  python yolo_tracking.py --input video.mp4 --output tracked.mp4 --model yolov8m.pt

YOLO Models (speed vs accuracy):
  yolov8n.pt - Nano (fastest, least accurate)
  yolov8s.pt - Small
  yolov8m.pt - Medium
  yolov8l.pt - Large
  yolov8x.pt - Extra large (slowest, most accurate)

Common COCO Classes:
  0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
  """
    )

    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                        help='YOLO model (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='Class IDs to track (default: all)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return

    process_video_yolo(args.input, args.output, args.model, args.conf, args.classes)


if __name__ == "__main__":
    # Command-line mode (uncomment to use)
    # main()

    # Direct execution mode
    input_video = "Task4/tracking.mp4"
    output_video = "Task4/yolo_tracked_output.mp4"

    # Track only vehicles: cars(2), motorcycles(3), buses(5), trucks(7)
    vehicle_classes = [2, 3, 5, 7]

    if os.path.exists(input_video):
        process_video_yolo(
            input_video,
            output_video,
            model_name='yolov8n.pt',  # Fast model
            conf_threshold=0.3,
            classes=vehicle_classes  # Only track vehicles
        )
    else:
        print(f"Error: Video file '{input_video}' not found!")
        print("\nUsage:")
        print("  1. Update 'input_video' path in the script")
        print("  2. Or use command line: python yolo_tracking.py --input video.mp4 --output out.mp4")