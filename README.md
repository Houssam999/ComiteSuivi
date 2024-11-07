ComiteSuivi

Overview

This repository contains a Python script, cst2.py, designed to analyze the vein network structures of Physarum polycephalum from a sequence of binary images. The script performs several key functions:

Skeletonization: It skeletonizes the binary images to extract the vein network structure.
Node Detection and Labeling: It detects and labels nodes (endpoints and junctions) within the skeletonized network.
Segment Identification and Analysis: It identifies and analyzes segments (veins) between nodes.
Width Measurement: It measures the widths of the veins at various points along each segment.
Fractal Dimension Calculation: It computes the fractal dimension of each segment.
Node Hierarchy Calculation: It calculates the hierarchical distance of each node from the central node in the network graph.
Visualization: It generates visualizations for verification and analysis.
Data Export: It saves the results, including images and detailed CSV files, for further study and for calibrating theoretical mathematical models.
The script is optimized for parallel processing, utilizing multiple CPU cores to efficiently handle large datasets.

Features

1. Parallel Processing
Efficiency: By leveraging Python's multiprocessing module, the script processes images in parallel, significantly reducing computation time.
Scalability: Designed to utilize systems with multiple CPU cores (e.g., 40 cores), making it suitable for handling large datasets.
2. Comprehensive Vein Network Analysis
Skeletonization: Converts binary images of the vein network into skeletal representations, preserving the essential structure.
Node Detection: Identifies and labels nodes, which are critical points such as endpoints and junctions in the network.
Segment Identification: Finds segments between nodes, representing individual vein paths.
Width Measurement: Measures the widths of veins at multiple points along each segment, providing detailed structural information.
Fractal Dimension Calculation: Computes the fractal dimension of each segment using the box-counting method, saving this information in the node data.
Node Hierarchy Calculation: Calculates the hierarchical distance of each node from the central node in the graph, adding this information directly to the node data.
3. Organized Output
Images: Saves key visualizations for each frame, including the original image, the skeletonized image, and the skeleton with detected nodes, facilitating visual verification.
CSV Files: Generates detailed CSV files for each frame, containing comprehensive data about nodes, segments, widths, fractal dimensions, and node hierarchy.
Output Directory: All outputs are systematically saved in a directory named HoussamAnalyse for easy access and organization.

Requirements

Python Version: Python 3.3 or higher
Python Packages:
numpy
matplotlib
opencv-python
scikit-image
scipy
pandas
networkx

Installation

Install the required Python packages using pip:
pip install numpy matplotlib opencv-python scikit-image scipy pandas networkx

Usage

1. Preparing the Data
Place the network2.npy file in the same directory as the script cst5.py. This file should be a NumPy array containing the sequence of binary images of Physarum polycephalum, with the vein network highlighted. Each frame in the array represents a time-lapse image to be processed.
2. Running the Script
Execute the script using the following command:
python3 cst2.py
3. Output
Upon completion, the script creates a directory named HoussamAnalyse containing:

Images (saved in HoussamAnalyse/images):
Image{frame_number}_original.png: Original binary image.
Image{frame_number}_skeleton.png: Skeletonized image of the vein network.
Image{frame_number}_skeleton_nodes.png: Skeleton with detected nodes highlighted.
CSV Files (saved in HoussamAnalyse):
1. nodes_info_frame_{frame_number}.csv: Contains detailed information about each node, including:
Frame: Frame number.
Node_ID: Unique identifier for the node.
X, Y: Coordinates of the node in the image.
Degree: Number of segments connected to the node.
Distance_to_Image_Center: Distance of the node from the center of the image.
Hierarchical_Distance: Hierarchical distance of the node from the central node in the network graph.
2. segments_info_frame_{frame_number}.csv: Contains detailed information about each segment (vein), including:
Frame: Frame number.
Segment_ID: Unique identifier for the segment.
Start_Node, End_Node: Node IDs of the start and end nodes.
Degree_A, Degree_B: Degrees of the start and end nodes.
Start_X, Start_Y: Coordinates of the start position of the segment.
End_X, End_Y: Coordinates of the end position of the segment.
Average_Width: Mean width calculated along the segment.
Width_Node_A, Width_Node_B: Widths measured at the start and end nodes.
Nb_Measurements: Number of width measurements taken along the segment.
Distance_from_Center: Euclidean distance of the segment's mean position from the center of the image.
3. fractal_info_frame_{frame_number}.csv: Contains the fractal dimension for each segment:
Segment_ID: Unique identifier for the segment.
Fractal_Dimension: Fractal dimension calculated for the segment.
Script Details

The script consists of several key functions and classes:

CompareNeighborsWithValue

Purpose: Calculates the number of neighboring pixels with a specific value for each pixel in the skeletonized image.
Functionality:
Supports 8-connectivity for neighbor detection.
Generates shifted matrices to compare neighboring pixel values efficiently.
Essential for identifying nodes based on the number of connected neighbors.

detect_nodes

Purpose: Detects and labels nodes (endpoints and junctions) in the skeletonized image.
Methodology:
Utilizes the CompareNeighborsWithValue class to compute neighbor counts.
Identifies nodes as pixels with either one neighbor (endpoints) or more than two neighbors (junctions).
Labels nodes and records their positions for further analysis.

find_segments

Purpose: Identifies segments (veins) between nodes in the skeletonized network.
Methodology:
Removes nodes from the skeleton to isolate segments.
Finds connected components representing individual segments.
Associates each segment with its start and end positions, even if the nodes are not detected.
Records the node IDs for start and end nodes (Start_Node, End_Node).

extract_node_degrees

Purpose: Calculates the degree of each node based on the number of connected segments.
Importance:
Node degrees provide insight into the network's topology.
Differentiates between simple endpoints (degree 1) and junctions (degree >2).

measure_segment_width

Purpose: Measures the width of each segment at various points along its length.
Methodology:
Samples perpendicular profiles across the segment at multiple points.
Interpolates pixel values using sub-pixel precision for accurate measurements.
Calculates various width metrics, including:
Average width
Width at the start node (Node A)
Width at the end node (Node B)
Number of measurements taken

compute_fractal_dimension

Purpose: Computes the fractal dimension of each segment using the box-counting method.
Methodology:
Applies box-counting to the binary mask of each segment.
Uses logarithmic scaling to compute the slope, representing the fractal dimension.
Captures the complexity and scaling behavior of the vein patterns.

process_frame

Purpose: Processes a single frame (image) in the dataset.
Steps:
Binarization and Skeletonization: Binarizes the current frame and performs skeletonization.
Node Detection: Detects and labels nodes in the skeletonized image.
Segment Identification: Identifies segments between nodes.
Width Measurement: Measures widths and computes fractal dimensions for each segment.
Graph Construction: Builds a graph using the nodes and segments.
Node Hierarchy Calculation: Calculates the hierarchical distance of each node from the central node in the graph.
Visualization: Saves visualizations:
Original image
Skeletonized image
Skeleton with detected nodes
Data Collection: Collects data for nodes, segments, fractal dimensions, and hierarchy, and saves them to CSV files.

main

Purpose: Manages the overall processing of the dataset.
Functionality:
Loads the dataset from network2.npy.
Initializes the output directory HoussamAnalyse.
Utilizes multiprocessing to process frames in parallel, using up to 40 CPU cores.
Ensures all frames are processed efficiently, and results are saved appropriately.
Data Outputs
Images (per frame)
Original Image: Visual reference of the raw binary image.
Skeletonized Image: Depicts the simplified structure of the vein network.
Skeleton with Detected Nodes: Highlights the nodes on the skeleton for visual verification.
CSV Files (per frame)
nodes_info_frame_{frame_number}.csv
Contains detailed information about each node:

Frame: Frame number.
Node_ID: Unique identifier for the node.
X, Y: Coordinates of the node in the image.
Degree: Number of segments connected to the node.
Distance_to_Image_Center: Distance of the node from the center of the image.
Hierarchical_Distance: Hierarchical distance from the central node in the network graph.
segments_info_frame_{frame_number}.csv
Contains detailed information about each segment:

Frame: Frame number.
Segment_ID: Unique identifier for the segment.
Start_Node, End_Node: Node IDs of the start and end nodes.
Start_X, Start_Y: Coordinates of the start position of the segment.
End_X, End_Y: Coordinates of the end position of the segment.
Degree_A, Degree_B: Degrees of the start and end nodes.
Average_Width: Mean width calculated along the segment.
Width_Node_A, Width_Node_B: Widths measured at the start and end nodes.
Nb_Measurements: Number of width measurements taken along the segment.
Distance_from_Center: Euclidean distance of the segment's mean position from the center of the image.
fractal_info_frame_{frame_number}.csv
Contains the fractal dimension for each segment:

Segment_ID: Unique identifier for the segment.
Fractal_Dimension: Fractal dimension calculated for the segment.
Important Notes
Precision of Measurements
Sub-pixel Accuracy: The widths are calculated using interpolation, allowing for sub-pixel precision. This means width values may be non-integer, offering more precise measurements of the vein structures.
Multiple Measurements: Widths are sampled at multiple points along each segment, providing a comprehensive profile.
Fractal Dimension
Insight into Complexity: The fractal dimension provides insight into the complexity and scaling properties of each vein segment.
Separate CSV Files: This information is saved in separate CSV files for detailed analysis.
Node Hierarchy
Central Node Determination: The central node is determined as the node closest to the center of the image.
Hierarchical Distance: The hierarchical distance represents the shortest path length from the central node to each node in the graph.
Integrated Data: Hierarchical distances are included directly in the node data CSV files, simplifying data analysis.
Parallel Processing
Optimized Performance: Utilizing up to 40 CPU cores, the script can process large datasets efficiently.
Resource Management: Ensure your system can support multiple parallel processes to avoid overloading.

Contact

For any questions, feedback, or further information, please contact:

Name: Houssam Henni
Email: hennihoussam99@gmail.com
