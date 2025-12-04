#!/usr/bin/env python3
"""
Skip-Trace Image Matching Solution
Task 5: Find similar images from database using multiple similarity metrics

File structure expected:
  task5.py (this file)
  vpr_dataset/
    ├── query/
    │   ├── query_1.jpg
    │   ├── query_1_1.jpg
    │   ├── query_2.jpg
    │   └── query_3.jpg
    └── smaller_database/
        ├── image001.jpg
        ├── image002.jpg
        └── ... (all database images)
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import heapq


class ImageMatcher:
    """
    Multi-metric image similarity matcher
    Combines perceptual hashing, color histograms, and ORB features
    """

    def __init__(self, n_features=300):
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def compute_phash(self, image_path: str) -> np.ndarray:
        """Compute perceptual hash using DCT"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            img_small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            dct = cv2.dct(np.float32(gray))
            dct_low = dct[:8, :8]
            avg = dct_low.mean()
            hash_val = (dct_low > avg).flatten()

            del img, img_small, gray, dct
            return hash_val
        except Exception as e:
            print(f"Error computing phash for {image_path}: {e}")
            return None

    def compute_histogram(self, image_path: str) -> np.ndarray:
        """Compute normalized HSV color histogram"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                                [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            del img, img_resized, hsv
            return hist
        except Exception as e:
            print(f"Error computing histogram for {image_path}: {e}")
            return None

    def compute_orb_descriptors(self, image_path: str) -> np.ndarray:
        """Compute ORB feature descriptors"""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            img_resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            _, descriptors = self.orb.detectAndCompute(img_resized, None)

            del img, img_resized
            return descriptors
        except Exception as e:
            print(f"Error computing ORB for {image_path}: {e}")
            return None

    def hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Compute Hamming distance between binary hashes"""
        if hash1 is None or hash2 is None:
            return float('inf')
        return np.sum(hash1 != hash2)

    def histogram_correlation(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute correlation coefficient between histograms"""
        if hist1 is None or hist2 is None:
            return 0.0

        mean1 = hist1.mean()
        mean2 = hist2.mean()

        numerator = np.sum((hist1 - mean1) * (hist2 - mean2))
        denominator = np.sqrt(np.sum((hist1 - mean1) ** 2) * np.sum((hist2 - mean2) ** 2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> int:
        """Match ORB descriptors and return number of good matches"""
        if desc1 is None or desc2 is None:
            return 0

        try:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            return len(matches)
        except:
            return 0

    def compute_similarity(self, query_path: str, db_path: str) -> float:
        """
        Compute weighted multi-metric similarity score

        Returns:
            float: Similarity score in [0, 1], where 1 = identical
        """
        scores = []

        # 1. Perceptual hash similarity (40% weight)
        try:
            hash_q = self.compute_phash(query_path)
            hash_db = self.compute_phash(db_path)
            hamming_dist = self.hamming_distance(hash_q, hash_db)
            phash_sim = 1.0 / (1.0 + hamming_dist)
            scores.append(phash_sim * 0.4)
        except:
            scores.append(0.0)

        # 2. Histogram similarity (30% weight)
        try:
            hist_q = self.compute_histogram(query_path)
            hist_db = self.compute_histogram(db_path)
            hist_corr = self.histogram_correlation(hist_q, hist_db)
            hist_sim = (hist_corr + 1.0) / 2.0
            scores.append(hist_sim * 0.3)
        except:
            scores.append(0.0)

        # 3. ORB feature matching (30% weight)
        try:
            desc_q = self.compute_orb_descriptors(query_path)
            desc_db = self.compute_orb_descriptors(db_path)
            num_matches = self.match_descriptors(desc_q, desc_db)
            orb_sim = min(num_matches / 50.0, 1.0)
            scores.append(orb_sim * 0.3)
        except:
            scores.append(0.0)

        return sum(scores)

    def find_top_k_matches(self, query_path: str, database_files: List[str], k: int = 1) -> List[Tuple[float, str]]:
        """
        Find top-k matches using min-heap for memory efficiency

        Args:
            query_path: Path to query image
            database_files: List of database image paths
            k: Number of top matches to return

        Returns:
            List of (similarity_score, filename) tuples, sorted by score descending
        """
        if not database_files:
            print(f"Error: No database images provided!")
            return []

        print(f"Processing query: {Path(query_path).name}")
        print(f"Database contains {len(database_files)} images")
        print("Computing similarities...")

        # Min-heap to efficiently track top-k
        top_k_heap = []

        for i, db_img_path in enumerate(database_files):
            if (i + 1) % 50 == 0 or i == len(database_files) - 1:
                print(f"  Processed {i + 1}/{len(database_files)} images...")

            sim = self.compute_similarity(query_path, db_img_path)

            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (sim, Path(db_img_path).name))
            else:
                if sim > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, (sim, Path(db_img_path).name))

        # Sort results by similarity (descending)
        results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)

        print(f"  Completed!\n")

        return results


def get_database_images(database_folder: Path) -> List[str]:
    """
    Get all database images from the database folder
    """
    if not database_folder.exists():
        print(f"Error: Database folder not found at {database_folder}")
        return []

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    db_images = []
    for file in database_folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            db_images.append(str(file))

    return sorted(db_images)


def format_results(query_name: str, matches: List[Tuple[float, str]], top_k: int = 1):
    """Pretty print results"""
    print(f"{query_name}:")
    if not matches:
        print("  No matches found!")
        return

    for i, (score, img_name) in enumerate(matches[:top_k], 1):
        if top_k == 1:
            print(f"  Best match: {img_name}")
            print(f"  Similarity: {score:.4f}")
        else:
            print(f"  {i}. {img_name} (similarity: {score:.4f})")


def main():
    """Main execution function"""

    print("=" * 80)
    print("SKIP-TRACE IMAGE MATCHING - Task 5")
    print("=" * 80)
    print()

    # Get script directory and construct paths
    SCRIPT_DIR = Path(__file__).parent.resolve() if '__file__' in globals() else Path.cwd()
    QUERY_DIR = SCRIPT_DIR / "vpr_dataset" / "query"
    DATABASE_DIR = SCRIPT_DIR / "vpr_dataset" / "smaller_database"

    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Query directory: {QUERY_DIR}")
    print(f"Database directory: {DATABASE_DIR}")
    print()

    # Check if paths exist
    if not QUERY_DIR.exists():
        print(f"ERROR: Query folder not found!")
        print(f"Expected: {QUERY_DIR}")
        print("\nMake sure you run this script from the directory containing vpr_dataset/")
        print("Your directory structure should be:")
        print("  task5.py")
        print("  vpr_dataset/")
        print("    ├── query/")
        print("    └── smaller_database/")
        return

    if not DATABASE_DIR.exists():
        print(f"ERROR: Database folder not found!")
        print(f"Expected: {DATABASE_DIR}")
        print("\nMake sure you run this script from the directory containing vpr_dataset/")
        return

    # Get database images
    print("Loading database images...")
    db_images = get_database_images(DATABASE_DIR)

    if not db_images:
        print("ERROR: No database images found!")
        print(f"Checked: {DATABASE_DIR}")
        return

    print(f"Found {len(db_images)} database images")
    if len(db_images) <= 10:
        for img in db_images:
            print(f"  - {Path(img).name}")
    else:
        for img in db_images[:5]:
            print(f"  - {Path(img).name}")
        print(f"  ... and {len(db_images) - 5} more")
    print()

    # Initialize matcher
    print("Initializing image matcher...")
    matcher = ImageMatcher(n_features=300)
    print()

    # Task 5a: Find top-1 matches for query_1, query_2, query_3
    print("=" * 80)
    print("TASK 5a: Finding Top-1 Matches")
    print("=" * 80)
    print()

    queries_a = ["query_1.jpg", "query_2.jpg", "query_3.jpg"]
    results_a = {}

    for query_name in queries_a:
        query_path = QUERY_DIR / query_name
        if not query_path.exists():
            print(f"{query_name}: File not found at {query_path}!\n")
            continue

        matches = matcher.find_top_k_matches(str(query_path), db_images, k=1)
        results_a[query_name] = matches
        format_results(query_name, matches, top_k=1)
        print()

    # Task 5b: Find top-5 matches for query_1_1
    print("=" * 80)
    print("TASK 5b: Finding Top-5 Matches for query_1_1.jpg")
    print("=" * 80)
    print()

    query_1_1_path = QUERY_DIR / "query_1_1.jpg"
    if query_1_1_path.exists():
        matches = matcher.find_top_k_matches(str(query_1_1_path), db_images, k=5)
        format_results("query_1_1.jpg", matches, top_k=5)
        results_b = matches
    else:
        print(f"query_1_1.jpg: File not found at {query_1_1_path}!")
        results_b = []

    print()

    # Print mathematical explanation
    print("=" * 80)
    print("MATHEMATICAL EXPLANATION OF SIMILARITY")
    print("=" * 80)
    print("""
The similarity between two images is computed using THREE complementary metrics:

1. PERCEPTUAL HASH SIMILARITY (40% weight):

   Process:
   - Resize image to 32×32 pixels and convert to grayscale
   - Apply Discrete Cosine Transform (DCT)
   - Extract 8×8 low-frequency DCT coefficients
   - Create binary hash: h[i] = 1 if DCT[i] > mean(DCT), else 0

   Formula:
       S_phash = 1 / (1 + Hamming_Distance(h₁, h₂))

   where Hamming_Distance counts the number of differing bits.
   This captures overall image structure and layout.


2. COLOR HISTOGRAM SIMILARITY (30% weight):

   Process:
   - Convert image to HSV color space
   - Compute 8×8×8 3D histogram across H, S, V channels
   - Normalize the histogram to unit magnitude

   Formula (Pearson Correlation):
       Corr = Σ[(H₁(i) - μ₁)(H₂(i) - μ₂)] / √[Σ(H₁(i) - μ₁)² · Σ(H₂(i) - μ₂)²]
       S_hist = (Corr + 1) / 2

   Normalized to [0, 1] range.
   This captures color distribution, robust to structural changes.


3. ORB FEATURE MATCHING (30% weight):

   Process:
   - Detect keypoints using FAST (Features from Accelerated Segment Test)
   - Compute 256-bit binary ORB descriptors at each keypoint
   - Match descriptors between images using Hamming distance
   - Count number of cross-checked matches

   Formula:
       S_orb = min(N_matches / 50, 1.0)

   where N_matches is the count of valid feature correspondences.
   This captures local structural patterns, invariant to rotation and scale.


COMBINED SIMILARITY SCORE:

   S_total = 0.4 × S_phash + 0.3 × S_hist + 0.3 × S_orb

where S_total ∈ [0, 1]:
   - 1.0 = identical images
   - 0.8-1.0 = very similar
   - 0.5-0.8 = moderately similar
   - 0.0-0.5 = different images

This multi-metric approach provides robustness to:
   - Lighting/exposure variations (histogram comparison)
   - Geometric transformations (ORB features)
   - Overall structure and layout (perceptual hash)
   - Occlusions and partial views (weighted combination)
""")

    # Save results summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\nTask 5a - Top-1 Matches:")
    for query_name, matches in results_a.items():
        if matches:
            print(f"  {query_name} → {matches[0][1]}")

    print("\nTask 5b - Top-5 Matches for query_1_1.jpg:")
    if results_b:
        for i, (score, img_name) in enumerate(results_b[:5], 1):
            print(f"  {i}. {img_name}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return results_a, results_b


if __name__ == "__main__":
    main()