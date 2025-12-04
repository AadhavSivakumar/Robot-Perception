import cv2
import os
import re
from pathlib import Path

def natural_sort_key(s):
    """Sort strings containing numbers in natural order"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_video_from_images(input_folder, output_video, fps=30):
    """
    Create an MP4 video from all images in a folder.

    Args:
        input_folder: Path to folder containing images
        output_video: Output video file path
        fps: Frames per second for the output video
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No image files found in {input_folder}")
        return

    # Sort files naturally (0.jpg, 1.jpg, ..., 10.jpg, etc.)
    image_files = sorted([str(f) for f in image_files], key=natural_sort_key)

    print(f"Found {len(image_files)} images")
    print(f"First image: {os.path.basename(image_files[0])}")
    print(f"Last image: {os.path.basename(image_files[-1])}")

    # Read the first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error: Could not read first image {image_files[0]}")
        return

    height, width, layers = first_frame.shape
    print(f"Video dimensions: {width}x{height}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to video
    for i, image_path in enumerate(image_files):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue

        # Resize if dimensions don't match
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")

    video_writer.release()
    print(f"\nVideo created successfully: {output_video}")
    print(f"Total frames: {len(image_files)}")
    print(f"Duration: {len(image_files) / fps:.2f} seconds")

if __name__ == "__main__":
    # Set paths relative to script location
    script_dir = Path(__file__).parent
    input_folder = script_dir / "images_subsample_final"
    output_video = script_dir / "output_video_final.mp4"
    fps = 30  # Adjust this value to change video speed

    print(f"Looking for images in: {input_folder}")
    print(f"Output video will be: {output_video}\n")

    # Create video
    create_video_from_images(str(input_folder), str(output_video), fps)
