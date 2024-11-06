# ComiteSuivi
## Overview

This repository contains a Python script, `cst.py`, designed to analyze the vein network structures of **Physarum polycephalum** from a sequence of binary images. The script performs several key functions:

- Skeletonization of the binary images to extract the vein network structure.
- Detection and labeling of nodes (endpoints and junctions) within the skeletonized network.
- Identification and analysis of segments (veins) between nodes.
- Measurement of vein widths at various points along each segment.
- Calculation of the fractal dimension for each vein segment using the box-counting method.
- Generation of visualizations for verification and analysis.
- Saving of results, including images and detailed CSV files, for further study and calibration of theoretical mathematical models.

The script is optimized for parallel processing, utilizing multiple CPU cores to efficiently handle large datasets.

## Features

### 1. Parallel Processing

- **Efficiency**: Leveraging Python's `multiprocessing` module, the script processes images in parallel, significantly reducing computation time.
- **Scalability**: Designed to utilize systems with multiple CPU cores (e.g., 40 cores), making it suitable for handling large datasets.

### 2. Comprehensive Vein Network Analysis

- **Skeletonization**: Converts binary images of the vein network into skeletal representations, preserving the essential structure.
- **Node Detection**: Identifies and labels nodes, which are critical points such as endpoints and junctions in the network.
- **Segment Identification**: Finds segments between nodes, representing individual vein paths.
- **Width Measurement**: Measures the width of veins at multiple points along each segment, providing detailed structural information.
- **Fractal Dimension Calculation**: Computes the fractal dimension of each vein segment, offering insights into the complexity and scaling behavior of the network.

### 3. Organized Output

- **Images**: Saves key visualizations for each frame, including the original image, skeletonized image, and skeleton with detected nodes, facilitating visual verification.
- **CSV Files**: Generates detailed CSV files for each frame, containing comprehensive data about nodes, segments, widths, and fractal dimensions.
- **Output Directory**: All outputs are systematically saved in a directory named `HoussamAnalyse` for easy access and organization.

## Requirements

- **Python Version**: Python 3.6 or higher.
- **Python Packages**:
  - `numpy`
  - `matplotlib`
  - `opencv-python`
  - `scikit-image`
  - `scipy`
  - `pandas`

## Installation

Install the required Python packages using `pip`:

```bash
pip install numpy matplotlib opencv-python scikit-image scipy pandas
