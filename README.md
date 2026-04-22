# Dynamic Panoramic Image Stitching Using OpenCV

This repository contains a Python script that dynamically creates a panoramic image by stitching together input images using OpenCV. The script handles various input sizes and resolutions to effectively combine overlapping photos.

## Features

* **Dynamic Image Input:** Accepts and processes between 2 and 8 input images from a designated local folder.
* **Feature Detection:** Utilizes SIFT (Scale-Invariant Feature Transform) to detect keypoints and compute descriptors for each input image.
* **Feature Matching:** Applies BFMatcher (Brute-Force Matcher) combined with Lowe's ratio test to find reliable, high-quality matches between consecutive images.
* **Homography & Warping:** Uses RANSAC to compute the homography matrix, enabling accurate perspective warping to align each image with the previous one.
* **Seamless Blending:** Dynamically creates a canvas large enough to hold the growing panorama, applying a custom linear interpolation formula to blend images seamlessly in overlapping regions.
* **Automated Cropping:** Implements grayscale thresholding and bounding box detection to automatically eliminate black areas and unneeded borders from the final stitched image.

## Requirements

To run this script, you will need Python 3.x and the following libraries:
* `opencv-python` (`cv2`)
* `numpy`
* `os` (Standard Python library)

You can install the required external dependencies using pip:
`pip install opencv-python numpy`

## How to Run

1. Clone this repository to your local machine.
2. Ensure you have a folder named `images` in the same directory as the script.
3. Place **2 to 8 images** (`.jpg`, `.jpeg`, or `.png`) inside the `images` folder. 
4. Run the script:
   ```bash
   python cv_main.py
