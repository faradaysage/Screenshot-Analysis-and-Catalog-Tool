#!/usr/bin/env python3

import cv2
import pytesseract
import numpy as np
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Set, List, Optional
import logging
from rich.console import Console
from rich.table import Table
from PIL import Image
import re
from dataclasses import dataclass, asdict
from typing import NamedTuple
import pickle

# Global configuration
TEXT_MATCH_REGEX = r'\#[A-Za-z]+' # Set to regex that you want to match
TEMPLATE_PATH = "image_template.png"  # Set this to the path for your template image to match
DATABASE_PATH = "screenshot_db.json"
FINGERPRINT_SIZE = (32, 32)  # Size to resize target images for fingerprinting
SIMILARITY_THRESHOLD = 0.95  # Threshold for fingerprint similarity

@dataclass
class ImageFingerprint:
    features: np.ndarray
    histogram: np.ndarray

@dataclass
class ScreenshotEntry:
    path: str
    checksum: str
    has_image: bool = False
    image_fingerprint: Optional[List[float]] = None  # Store as list for JSON serialization
    matching_text: Optional[str] = None
    
class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.entries: Dict[str, ScreenshotEntry] = {}
        self.matching_text_index: Dict[str, List[str]] = {}  # matching_text -> list of file paths

    def save(self):
        """Save database to file."""
        with open(self.db_path, 'w') as f:
            # Convert entries to a JSON-serializable format
            data = {
                'entries': {k: asdict(v) for k, v in self.entries.items()},
                'matching_text_index': self.matching_text_index
            }
            # Convert ImageFingerprint to dictionary format for JSON
            for entry in data['entries'].values():
                if entry['image_fingerprint']:
                    entry['image_fingerprint'] = {
                        'features': entry['image_fingerprint']['features'],
                        'histogram': entry['image_fingerprint']['histogram']
                    }
            json.dump(data, f, indent=2)

    def load(self):
        """Load database from file or create new if doesn't exist."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                self.entries = {}
                for k, v in data['entries'].items():
                    # Reconstruct ScreenshotEntry and ensure image_fingerprint is ImageFingerprint
                    if isinstance(v['image_fingerprint'], dict):
                        v['image_fingerprint'] = ImageFingerprint(
                            features=v['image_fingerprint']['features'],
                            histogram=v['image_fingerprint']['histogram']
                        )
                    else:
                        v['image_fingerprint'] = None  # Handle cases where image_fingerprint is not properly formatted
                    self.entries[k] = ScreenshotEntry(**v)
                self.matching_text_index = data['matching_text_index']

            
    def add_entry(self, path: str, entry: ScreenshotEntry):
        """Add or update an entry and update indexes."""
        # Check that the entry has a valid ImageFingerprint object with histogram data before adding
        if isinstance(entry.image_fingerprint, ImageFingerprint):
            if entry.image_fingerprint.histogram:
                self.entries[path] = entry
                if entry.matching_text:
                    if entry.matching_text not in self.matching_text_index:
                        self.matching_text_index[entry.matching_text] = []
                    self.matching_text_index[entry.matching_text].append(path)
                #logging.info(f"Added entry with valid histogram: {path}")
            else:
                logging.warning(f"Attempted to add entry without a valid histogram: {path}")
        else:
            logging.warning(f"Attempted to add entry without a valid ImageFingerprint: {path}")
        self.save()  # Save after each update


class ImageProcessor:
    def __init__(self, template_path: str):
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError("Could not load template image")
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        

    def extract_image_region(self, image_path: str, debug: bool = False, find_best_match: bool = False, debug_output_dir: Path = Path("debug_matches")) -> Optional[np.ndarray]:
        """Extract the target region from an image, using a mask to exclude transparent areas.
        
        Parameters:
        - image_path: Path to the input image
        - debug: If True, save debug images to disk
        - find_best_match: If True, finds the best match; if False, returns the first match that meets the threshold
        - debug_output_dir: Directory to save debug images
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.error("Image could not be loaded.")
                return None

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img_gray.shape

            # Load the template with transparency and create the mask
            template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            
            # Mask: 255 where opaque, 0 where fully transparent
            mask = (template[:, :, 3] > 0).astype(np.uint8) * 255  # Use alpha channel for mask
            template_gray = template_gray * (mask > 0)  # Apply mask to template

            # Scale range to handle minor variations in target size
            base_scale = 477 / template_gray.shape[1]
            scales = [base_scale * (1 - 0.05), base_scale, base_scale * (1 + 0.05)]
            
            # Known x offsets and tolerance range
            x_offsets = [72, 99]
            offset_tolerance = 10

            best_match = None
            best_val = -1
            best_loc = None
            best_size = (0, 0)

            # Iterate over scales and x offsets with tolerance
            for scale in scales:
                # Resize the template and mask according to the current scale
                template_height = int(template_gray.shape[0] * scale)
                template_width = int(template_gray.shape[1] * scale)

                # Ensure resized template fits within the image dimensions
                if template_height > h or template_width > w:
                    continue

                template_resized = cv2.resize(template_gray, (template_width, template_height))
                mask_resized = cv2.resize(mask, (template_width, template_height))

                for x_offset in x_offsets:
                    # Allow ±10 pixels around each x offset
                    for x in range(x_offset - offset_tolerance, x_offset + offset_tolerance + 1):
                        if x < 0 or x + template_width > w:
                            continue  # Ensure search region is within image bounds

                        # Define search region for the current x offset and perform template matching
                        search_region = img_gray[:, x:x + template_width]
                        result = cv2.matchTemplate(search_region, template_resized, cv2.TM_CCORR_NORMED, mask=mask_resized)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        # Save each match result to disk with score for debug
                        if debug:
                            display_img = img.copy()
                            top_left = (max_loc[0] + x, max_loc[1])
                            bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
                            cv2.rectangle(display_img, top_left, bottom_right, (0, 255, 0), 2)
                            debug_file = debug_output_dir / f"match_scale_{scale}_x_{x}_score_{max_val:.2f}.png"
                            cv2.imwrite(str(debug_file), display_img)

                        # Check if the current match meets the threshold (match above 0.87 is pretty good, just slightly offset... above 0.95 is near perfect)
                        if max_val > 0.85:
                            # If we're only looking for the first match, return immediately
                            if not find_best_match:
                                top_left = (max_loc[0] + x, max_loc[1])
                                return img[top_left[1]:top_left[1] + template_height, top_left[0]:top_left[0] + template_width]

                            # If finding the best match, keep track of the highest score
                            if max_val > best_val:
                                best_match = template_resized
                                best_val = max_val
                                best_loc = (max_loc[0] + x, max_loc[1])
                                best_size = (template_width, template_height)

            # If finding the best match and a match was found, return the best one
            if find_best_match and best_match is not None and best_loc is not None:
                x, y = best_loc
                template_width, template_height = best_size
                return img[y:y + template_height, x:x + template_width]

            return None

        except Exception as e:
            logging.error(f"Error extracting image region from {image_path}: {e}")
            return None


    def create_fingerprint(self, image_image: np.ndarray, debug: bool = False) -> Optional[ImageFingerprint]:
        """Create a fingerprint from only the fully transparent regions of a image image, based on a bounding box."""
        try:
            # Load the template with alpha channel to create the transparency mask
            template_with_alpha = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
            if template_with_alpha is None or template_with_alpha.shape[2] != 4:
                logging.error("Template with alpha channel is missing or invalid.")
                return None

            # Extract the alpha channel and create a mask for fully transparent regions
            template_alpha = template_with_alpha[:, :, 3]
            transparent_mask = (template_alpha == 0).astype(np.uint8)  # Mask is 1 where alpha is 0

            # Resize transparent_mask to match the dimensions of image_image
            image_image_height, image_image_width = image_image.shape[:2]
            transparent_mask_resized = cv2.resize(transparent_mask, (image_image_width, image_image_height), interpolation=cv2.INTER_NEAREST)

            # Find the bounding box of the transparent area in the resized mask
            x, y, w, h = cv2.boundingRect(transparent_mask_resized)

            # Crop the image image to the bounding box of the transparent area
            cropped_image_image = image_image[y:y + h, x:x + w]

            # Check if the cropped region is empty
            if cropped_image_image.size == 0:
                logging.error("Cropped image image is empty.")
                return None

            # Output the cropped area for debugging
            if debug:
                debug_dir = Path("debug_fingerprint")
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"cropped_area_{hash(image_image.tobytes())}.png"
                cv2.imwrite(str(debug_path), cropped_image_image)
                logging.info(f"Saved cropped area for fingerprint to: {debug_path}")

            # Resize the cropped area to FINGERPRINT_SIZE
            resized_cropped_area = cv2.resize(cropped_image_image, FINGERPRINT_SIZE)

            # Calculate histogram for the resized cropped area
            hist = cv2.calcHist([resized_cropped_area], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            if hist is None:
                logging.error("Histogram calculation failed; hist is None.")
                return None

            # Normalize and convert histogram to float32 type for compatibility
            hist = cv2.normalize(hist, hist).astype('float32').flatten()
            #logging.info(f"Generated histogram shape from cropped area: {hist.shape}")

            # Convert the resized cropped area to grayscale for SIFT feature detection
            gray_cropped_area = cv2.cvtColor(resized_cropped_area, cv2.COLOR_BGR2GRAY)
            
            # Initialize SIFT and detect keypoints and descriptors
            sift = cv2.SIFT_create()
            keypoints, features = sift.detectAndCompute(gray_cropped_area, None)

            # Handle the case where no features were detected
            features_list = features.flatten().tolist() if features is not None else []

            return ImageFingerprint(features=features_list, histogram=hist.tolist())

        except Exception as e:
            logging.error(f"Error creating fingerprint: {e}")
            return None






    def compare_fingerprints(self, fp1: ImageFingerprint, fp2: ImageFingerprint, debug: bool = False, file1: str = "", file2: str = "") -> float:
        """Compare two fingerprints and return similarity score."""
        try:
            # Convert lists back to numpy arrays and check histogram integrity
            features1 = np.array(fp1.features, dtype=np.float32).reshape(-1, 128) if fp1.features and len(fp1.features) % 128 == 0 else None
            features2 = np.array(fp2.features, dtype=np.float32).reshape(-1, 128) if fp2.features and len(fp2.features) % 128 == 0 else None
            hist1 = np.array(fp1.histogram, dtype=np.float32) if fp1.histogram else None
            hist2 = np.array(fp2.histogram, dtype=np.float32) if fp2.histogram else None

            # Check histograms are valid and have the same shape
            if hist1 is None or hist2 is None or hist1.shape != hist2.shape:
                logging.error(f"Histogram dimensions do not match or are missing: hist1={hist1}, hist2={hist2}")
                return 0.0

            # Histogram comparison
            hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # Feature matching only if both feature vectors are non-empty and compatible
            feature_score = 0.0
            if features1 is not None and features2 is not None and features1.shape == features2.shape:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(features1, features2, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                feature_score = len(good_matches) / max(len(features1), len(features2))
                
            # Combine histogram and feature scores
            similarity_score = 0.7 * hist_score + 0.3 * feature_score

            # Print debug information if enabled
            if debug:
                logging.info(f"Comparing {file1} to {file2} - Similarity score: {similarity_score * 100:.2f}% (hist_score: {hist_score}; feature_score: {feature_score})")

            return similarity_score

        except Exception as e:
            logging.error(f"Error comparing fingerprints: {e}")
            return 0.0




    def find_nearest_matching_text(self, image_path: str, image_region: np.ndarray) -> Optional[str]:
        """Find matching_text nearest to the image region."""
        try:
            # Extract text and positions
            image = Image.open(image_path)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            nearest_matching_text = None
            min_distance = float('inf')
            
            # Process each word
            for i, text in enumerate(data['text']):
                if re.match(TEXT_MATCH_REGEX, text):
                    text_x = data['left'][i] + (data['width'][i] // 2)
                    text_y = data['top'][i] + (data['height'][i] // 2)
                    
                    # Calculate distance to image center
                    image_center_x = image_region.shape[1] // 2
                    image_center_y = image_region.shape[0] // 2
                    distance = ((text_x - image_center_x) ** 2 + (text_y - image_center_y) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_matching_text = text
            
            return nearest_matching_text
            
        except Exception as e:
            logging.error(f"Error finding matching_text: {e}")
            return None


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with file_path.open('rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def rename_with_suffix(source_file: Path, dest_file: Path) -> Path:
    """Rename the source file to the destination file, adding a suffix if needed."""
    new_dest_file = get_unique_filename(dest_file)
    source_file.rename(new_dest_file)
    return new_dest_file


def copy_with_suffix(source_file: Path, dest_file: Path) -> Path:
    """Copy the source file to the destination file, adding a suffix if needed."""
    from shutil import copy2
    new_dest_file = get_unique_filename(dest_file)
    copy2(source_file, new_dest_file)
    return new_dest_file


def get_unique_filename(dest_file: Path) -> Path:
    """Generate a unique filename by adding a numeric suffix if needed."""
    suffix = 1
    new_dest_file = dest_file
    while new_dest_file.exists():
        new_dest_file = dest_file.with_stem(f"{dest_file.stem}({suffix})")
        suffix += 1
    return new_dest_file


def process_files(source_dir: Path, dest_dir: Path, db: Database) -> Dict[str, int]:
    """Process files according to the specified workflow."""
    stats = {
        'thumbnails_removed': 0,
        'size_matches': 0,
        'checksum_matches': 0,
        'no_image': 0,
        'similar_image': 0,
        'duplicate_matching_text': 0,
        'new_unique': 0,
        'duplicate_matching_text_array': []
    }
    
    processor = ImageProcessor(TEMPLATE_PATH)
    
    # Process each file in source directory
    for source_file in source_dir.rglob('*'):
        if not source_file.is_file():
            continue
            
        # Skip non-image files
        if not source_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            continue
            
        # Remove thumbnails
        if '_thumb' in source_file.stem:
            source_file.unlink()
            stats['thumbnails_removed'] += 1
            continue
            
        relative_path = source_file.relative_to(source_dir)
        dest_file = dest_dir / relative_path
        
        # Check filename/size matches
        if dest_file.exists() and dest_file.stat().st_size == source_file.stat().st_size:
            source_file.unlink()
            stats['size_matches'] += 1
            continue
            
        # Calculate checksum
        checksum = calculate_checksum(source_file)
        
        # Check for checksum matches
        checksum_match = False
        for entry in db.entries.values():
            if entry.checksum == checksum:
                #source_file.unlink()
                stats['checksum_matches'] += 1
                checksum_match = True
                break
                
        if checksum_match:
            continue
            
        # Extract image region
        image_region = processor.extract_image_region(str(source_file))
        
        # Handle case with no image
        if image_region is None:
            logging.warning(f"No image region found or image invalid for file: {source_file}")
            entry = ScreenshotEntry(
                path=str(relative_path),
                checksum=checksum,
                has_image=False
            )
            db.add_entry(str(relative_path), entry)
            copy_with_suffix(source_file, dest_file)
            stats['no_image'] += 1
            continue
            
        # Create fingerprint
        try:
            fingerprint = processor.create_fingerprint(image_region)
        except Exception as e:
            logging.error(f"Error creating fingerprint from {source_file}: {e}")
            stats['no_image'] += 1
            continue
                            
        # Check for similar images
        try:
            for entry in db.entries.values():
                if entry.image_fingerprint and fingerprint:
                    similarity = processor.compare_fingerprints(
                        entry.image_fingerprint,
                        fingerprint,
                        debug=False,  # Enable debug logging
                        file1=str(entry.path),  # Current database file name
                        file2=str(relative_path)  # New source file name
                    )
                    if similarity > SIMILARITY_THRESHOLD:
                        #source_file.unlink()
                        print(f"The two files are too similiar ({matching_text}): '{entry.path}' ~= '{relative_path}'")
                        stats['similar_image'] += 1
                        stats['duplicate_matching_text_array'].append(matching_text)
                        continue
        except Exception as e:
            logging.error(f"Error comparing fingerprints for {source_file}: {e}")
            continue
                    
        # Find nearest matching_text
        try:
            matching_text = processor.find_nearest_matching_text(str(source_file), image_region)
            if matching_text:
                matching_text = matching_text.strip()
        except Exception as e:
            logging.error(f"Error finding matching_text in {source_file}: {e}")
            continue
        
        
        # Check for duplicate matching texts
        if matching_text and matching_text in db.matching_text_index:
            #source_file.unlink()
            print(f"Duplicate matching_text: {matching_text}")
            stats['duplicate_matching_text'] += 1
            stats['duplicate_matching_text_array'].append(matching_text)
            continue
            
        # New unique file
        entry = ScreenshotEntry(
            path=str(relative_path),
            checksum=checksum,
            has_image=True,
            image_fingerprint=fingerprint if isinstance(fingerprint, ImageFingerprint) else None,
            matching_text=matching_text
        )
        db.add_entry(str(relative_path), entry)
        rename_with_suffix(source_file, dest_file)
        stats['new_unique'] += 1
        
    return stats
    

def initialize_database(dest_dir: Path, db: Database):
    """Initialize the database by iterating over images in the destination directory."""
    processor = ImageProcessor(TEMPLATE_PATH)
    
    for image_file in dest_dir.rglob('*'):
        if not image_file.is_file() or image_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        checksum = calculate_checksum(image_file)
        image_region = processor.extract_image_region(str(image_file))
        fingerprint = processor.create_fingerprint(image_region) if image_region is not None else None
        
        matching_text = ""
        try:
            matching_text = processor.find_nearest_matching_text(str(image_file), image_region) if image_region is not None else None
            if matching_text:
                matching_text = matching_text.strip()
        except Exception as e:
            logging.error(f"Error finding matching text in {image_file}: {e}")
            continue

        entry = ScreenshotEntry(
            path=str(image_file.relative_to(dest_dir)),
            checksum=checksum,
            has_image=image_region is not None,
            image_fingerprint=fingerprint if isinstance(fingerprint, ImageFingerprint) else None,
            matching_text=matching_text
        )
        
        db.add_entry(str(image_file.relative_to(dest_dir)), entry)



def main():
    # Setup logging and parse arguments
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str)
    parser.add_argument('dest_dir', type=str)
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    if not source_dir.exists():
        logging.error("Source directory does not exist")
        return 1
        
    # Create dest directory if needed
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    db = Database(DATABASE_PATH)
    if not os.path.exists(DATABASE_PATH):
        initialize_database(dest_dir, db)
    db.load()  # This will create new if doesn't exist
    
    # Process files
    try:
        stats = process_files(source_dir, dest_dir, db)
        
        # Display report
        console = Console()
        table = Table(title="Screenshot Processing Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
            
        console.print(table)
        
        # Display matching text index
        if db.matching_text_index:
            console.print("\n[bold cyan]Matching Text Index:[/bold cyan]")
            matching_text_table = Table()
            matching_text_table.add_column("matching_text", style="yellow")
            matching_text_table.add_column("Files", style="green")
            
            for tag, files in db.matching_text_index.items():
                matching_text_table.add_row(tag, "\n".join(files))
                
            console.print(matching_text_table)
            
        duplicate_tickers = ', '.join(stats['duplicate_matching_text_array'])
        console.print(f"Duplicate tickers: {duplicate_tickers}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())