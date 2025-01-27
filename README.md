# Screenshot Cataloging Tool

## Overview
This is a Python-based tool designed to catalog screenshots by:
1. Detecting a specific template image within screenshots using computer vision techniques.
2. Extracting text patterns (e.g., hashtags, specific keywords) using Optical Character Recognition (OCR).
3. Storing and managing screenshot metadata in a database for efficient indexing and retrieval.

This project demonstrates advanced usage of computer vision and AI technologies, making it a valuable showcase of practical AI applications.

---

## Features
- **Template Matching**:
  - Detects regions in images that match a given template using OpenCV's `cv2.matchTemplate` function.
  - Employs techniques like grayscale conversion, image resizing, and multi-scale matching for robust detection.

- **Image Fingerprinting**:
  - Generates unique fingerprints for detected template regions using a combination of:
    - **Histogram-based color analysis**.
    - **SIFT (Scale-Invariant Feature Transform)** for feature extraction and keypoint matching.
  - Enables similarity scoring between images to avoid duplicate entries in the database.

- **Optical Character Recognition (OCR)**:
  - Utilizes Tesseract OCR via the `pytesseract` library to extract text from images.
  - Regex-powered text filtering (e.g., hashtags) for pattern-based text extraction.

- **Database Management**:
  - Stores image metadata, fingerprints, and extracted text in a JSON-based database for persistent storage.
  - Tracks duplicate entries and supports efficient querying.

---

## Libraries and Techniques Used
### 1. Computer Vision (OpenCV)
- **Template Matching**:
  - Used `cv2.matchTemplate` for locating template regions in images.
  - Grayscale conversion and mask-based template matching for improved accuracy.
- **Feature Detection and Matching**:
  - Leveraged `cv2.SIFT_create` for detecting and describing keypoints in images.
  - Used brute-force matching (BFMatcher) for comparing feature descriptors.
- **Histogram Analysis**:
  - Calculated 3D histograms (`cv2.calcHist`) and normalized them for robust color-based fingerprinting.

### 2. Optical Character Recognition (Tesseract)
- Used `pytesseract` for OCR to extract text data from images.
- Enhanced text detection with regular expressions for customizable matching (e.g., hashtags).

### 3. Data Management
- Implemented a JSON-based database for efficient storage and retrieval of image metadata.
- Indexed matching text (e.g., hashtags) to allow quick lookups.

---

## Getting Started

### Prerequisites
- **Python 3.8+**
- Required Python packages:
  ```bash
  pip install opencv-python numpy pytesseract rich pillow
  ```
- Install Tesseract OCR:
  - **Ubuntu**:
    ```bash
    sudo apt install tesseract-ocr
    ```
  - **macOS**:
    ```bash
    brew install tesseract
    ```
  - **Windows**:
    Download and install from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

### Usage
#### Command-Line Arguments
- **Basic Usage**:
  ```bash
  python script.py <source_dir> <dest_dir>
  ```
  - `<source_dir>`: Directory containing screenshots to process.
  - `<dest_dir>`: Directory where processed files will be stored.

### Example
```bash
python script.py ./screenshots ./processed
```

---

## Project Structure
- **`script.py`**: Main script for processing screenshots.
- **`screenshot_db.json`**: JSON-based database for storing metadata.
- **`template.png`**: Example template image for matching.

---

## Future Improvements
- Integrate advanced deep learning-based object detection models (e.g., YOLO, Faster R-CNN).
- Add support for more complex text extraction using NLP techniques.
- Implement a GUI for easier interaction and visualization of results.

---

## Author
Feel free to reach out for more details about this project.

---

## Acknowledgments
- OpenCV and Tesseract teams for their excellent open-source libraries.
- The Python community for making projects like this possible.
