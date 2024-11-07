# ComiteSuivi
## Overview

This repository contains a Python script, `cst2.py`, designed to analyze the vein network structures of **Physarum polycephalum** from a sequence of binary images. The script performs several key functions:
- It skeletonizes the binary images to extract the vein network structure.
- It detects and labels nodes (endpoints and junctions) within the skeletonized network.
- It identifies and analyzes segments (veins) between nodes.
- It measures the widths of the veins at various points along each segment.
- It computes the fractal dimension of each segment and saves this information in separate CSV files.
- It generates visualizations for verification and analysis.
- It saves the results, including images and detailed CSV files, for further study and for calibrating theoretical mathematical models.
The script is optimized for parallel processing, utilizing multiple CPU cores to efficiently handle large datasets.
## Features
### 1. Parallel Processing
- **Efficiency**: By leveraging Python's `multiprocessing` module, the script processes images in parallel, significantly reducing computation time.
- **Scalability**: Designed to utilize systems with multiple CPU cores (e.g., 40 cores), making it suitable for handling large datasets.
### 2. Comprehensive Vein Network Analysis
- **Skeletonization**: Converts binary images of the vein network into skeletal representations, preserving the essential structure.
- **Node Detection**: Identifies and labels nodes, which are critical points such as endpoints and junctions in the network.
- **Segment Identification**: Finds segments between nodes, representing individual vein paths.
- **Width Measurement**: Measures the widths of veins at multiple points along each segment, providing detailed structural information.
- **Fractal Dimension Calculation**: Computes the fractal dimension of each segment using the box-counting method, and saves this information in separate CSV files.
### 3. Organized Output
- **Images**: Saves key visualizations for each frame, including the original image, the skeletonized image, and the skeleton with detected nodes, facilitating visual verification.
- **CSV Files**: Generates detailed CSV files for each frame, containing comprehensive data about nodes, segments, widths, and fractal dimensions.
- **Output Directory**: All outputs are systematically saved in a directory named `HoussamAnalyse` for easy access and organization.
## Requirements
- **Python Version**: Python 3.6 (or higher)
- **Python Packages**:
  - `numpy`
  - `matplotlib`
  - `opencv-python`
  - `scikit-image`
  - `scipy`
  - `pandas`
## Installation
Install the required Python packages using `pip`:
pip install numpy matplotlib opencv-python scikit-image scipy pandas
## Usage
### 1. Preparing the Data
Place the `network2.npy` file in the same directory as the script `cst2.py`. This file should be a NumPy array containing the sequence of binary images of *Physarum polycephalum*, with the vein network highlighted. Each frame in the array represents a time-lapse image to be processed.
### 2. Running the Script
Execute the script using the following command:
python3 cst2.py
### 3. Output
Upon completion, the script creates a directory named `HoussamAnalyse` containing:
- **Images** (saved in `HoussamAnalyse/images`):
  - `Image{frame_number}_original.png`: Original binary image.
  - `Image{frame_number}_skeleton.png`: Skeletonized image of the vein network.
  - `Image{frame_number}_skeleton_nodes.png`: Skeleton with detected nodes highlighted.
- **CSV Files** (saved in `HoussamAnalyse`):
  - `nodes_info_frame_{frame_number}.csv`: Contains detailed information about each node, including:
    - `Frame`: Frame number.
    - `Node_ID`: Unique identifier for the node.
    - `X`, `Y`: Coordinates of the node in the image.
    - `Degree`: Number of segments connected to the node.
  - `segments_info_frame_{frame_number}.csv`: Contains detailed information about each segment (vein), including:
    - `Frame`: Frame number.
    - `Segment_ID`: Unique identifier for the segment.
    - `Degree_A`, `Degree_B`: Degrees of the start and end nodes.
    - `Start_X`, `Start_Y`: Coordinates of the start position of the segment.
    - `End_X`, `End_Y`: Coordinates of the end position of the segment.
    - `Average_Width`: Mean width calculated along the segment.
    - `Width_Node_A`, `Width_Node_B`: Widths measured at the start and end nodes.
    - `Nb_Measurements`: Number of width measurements taken along the segment.
    - `Distance_from_Center`: Euclidean distance of the segment's mean position from the center of the image.
  - `fractal_info_frame_{frame_number}.csv`: Contains the fractal dimension for each segment:
    - `Segment_ID`: Unique identifier for the segment.
    - `Fractal_Dimension`: Fractal dimension calculated for the segment.
## Script Details
The script consists of several key functions and classes:
### `CompareNeighborsWithValue`
**Purpose**: Calculates the number of neighboring pixels with a specific value for each pixel in the skeletonized image.
**Functionality**:
- Supports 8-connectivity for neighbor detection.
- Generates shifted matrices to compare neighboring pixel values efficiently.
- Essential for identifying nodes based on the number of connected neighbors.
### `detect_nodes`
**Purpose**: Detects and labels nodes (endpoints and junctions) in the skeletonized image.
**Methodology**:
- Utilizes the `CompareNeighborsWithValue` class to compute neighbor counts.
- Identifies nodes as pixels with either one neighbor (endpoints) or more than two neighbors (junctions).
- Labels nodes and records their positions for further analysis.
### `find_segments`
**Purpose**: Identifies segments (veins) between nodes in the skeletonized network.
**Methodology**:
- Removes nodes from the skeleton to isolate segments.
- Finds connected components representing individual segments.
- Associates each segment with its start and end positions, even if the nodes are not detected.
- Compiles coordinates of all points in each segment.
### `extract_node_degrees`
**Purpose**: Calculates the degree of each node based on the number of connected segments.
**Importance**:
- Node degrees provide insight into the network's topology.
- Differentiates between simple endpoints (degree 1) and junctions (degree >2).
### `measure_segment_width`
**Purpose**: Measures the width of each segment at various points along its length.
**Methodology**:
- Samples perpendicular profiles across the segment at multiple points.
- Interpolates pixel values using sub-pixel precision for accurate measurements.
- Calculates various width metrics, including:
  - Average width
  - Width at the start node (Node A)
  - Width at the end node (Node B)
  - Number of measurements taken
### `compute_fractal_dimension`
**Purpose**: Computes the fractal dimension of each segment using the box-counting method.
**Methodology**:
- Applies box-counting to the binary mask of each segment.
- Uses logarithmic scaling to compute the slope, representing the fractal dimension.
- Captures the complexity and scaling behavior of the vein patterns.
### `process_frame`
**Purpose**: Processes a single frame (image) in the dataset.
**Steps**:
- Binarizes the current frame and performs skeletonization.
- Detects nodes and segments.
- Measures widths and computes fractal dimensions for each segment.
- Saves visualizations:
  - Original image
  - Skeletonized image
  - Skeleton with detected nodes
- Collects data for nodes, segments, and fractal dimensions, and saves them to CSV files.
### `main`
**Purpose**: Manages the overall processing of the dataset.
**Functionality**:
- Loads the dataset from `network2.npy`.
- Initializes the output directory `HoussamAnalyse`.
- Utilizes multiprocessing to process frames in parallel.
- Ensures all frames are processed efficiently, and results are saved appropriately.
## Data Outputs
### Images (per frame)
- **Original Image**: Visual reference of the raw binary image.
- **Skeletonized Image**: Depicts the simplified structure of the vein network.
- **Skeleton with Detected Nodes**: Highlights the nodes on the skeleton for visual verification.
### CSV Files (per frame)
1. **`nodes_info_frame_{frame_number}.csv`**
   Contains detailed information about each node:
   - `Frame`: Frame number.
   - `Node_ID`: Unique identifier for the node.
   - `X`, `Y`: Coordinates of the node in the image.
   - `Degree`: Number of segments connected to the node.
2. **`segments_info_frame_{frame_number}.csv`**
   Contains detailed information about each segment:
   - `Frame`: Frame number.
   - `Segment_ID`: Unique identifier for the segment.
   - `Degree_A`, `Degree_B`: Degrees of the start and end nodes.
   - `Start_X`, `Start_Y`: Coordinates of the start position of the segment.
   - `End_X`, `End_Y`: Coordinates of the end position of the segment.
   - `Average_Width`: Mean width calculated along the segment.
   - `Width_Node_A`, `Width_Node_B`: Widths measured at the start and end nodes.
   - `Nb_Measurements`: Number of width measurements taken along the segment.
   - `Distance_from_Center`: Euclidean distance of the segment's mean position from the center of the image.
3. **`fractal_info_frame_{frame_number}.csv`**
   Contains the fractal dimension for each segment:
   - `Segment_ID`: Unique identifier for the segment.
   - `Fractal_Dimension`: Fractal dimension calculated for the segment.
## Important Notes
- **Precision of Measurements**:
  - The widths are calculated using interpolation, allowing for sub-pixel precision. This means width values may be non-integer, offering more precise measurements of the vein structures.
- **Fractal Dimension**:
  - The fractal dimension provides insight into the complexity and scaling properties of each vein segment.
  - This information is saved in separate CSV files.
## Contact
For any questions, feedback, or further information, please contact:
- **Name**: Houssam Henni
- **Email**: [hennihoussam99@gmail.com]
