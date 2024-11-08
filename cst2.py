#!/usr/bin/env python3
"""
This script processes a sequence of binary images of Physarum polycephalum.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import morphology
from scipy import ndimage
from collections import defaultdict
import pandas as pd
from multiprocessing import Pool
import os
import networkx as nx  

# Definition of the class CompareNeighborsWithValue
class CompareNeighborsWithValue:
    def __init__(self, matrix, connectivity, data_type=np.int32):
        self.matrix = matrix.astype(data_type)
        self.connectivity = connectivity
        # Create shifted matrices with padding to keep dimensions
        # Right neighbor
        self.on_the_right = np.zeros_like(self.matrix)
        self.on_the_right[:, :-1] = self.matrix[:, 1:]
        # Left neighbor
        self.on_the_left = np.zeros_like(self.matrix)
        self.on_the_left[:, 1:] = self.matrix[:, :-1]
        # Bottom neighbor
        self.on_the_bottom = np.zeros_like(self.matrix)
        self.on_the_bottom[:-1, :] = self.matrix[1:, :]
        # Top neighbor
        self.on_the_top = np.zeros_like(self.matrix)
        self.on_the_top[1:, :] = self.matrix[:-1, :]
        if self.connectivity == 8:
            # Top-left neighbor
            self.on_the_topleft = np.zeros_like(self.matrix)
            self.on_the_topleft[1:, 1:] = self.matrix[:-1, :-1]
            # Top-right neighbor
            self.on_the_topright = np.zeros_like(self.matrix)
            self.on_the_topright[1:, :-1] = self.matrix[:-1, 1:]
            # Bottom-left neighbor
            self.on_the_bottomleft = np.zeros_like(self.matrix)
            self.on_the_bottomleft[:-1, 1:] = self.matrix[1:, :-1]
            # Bottom-right neighbor
            self.on_the_bottomright = np.zeros_like(self.matrix)
            self.on_the_bottomright[:-1, :-1] = self.matrix[1:, 1:]

    def is_equal(self, value, and_itself=False):
        """Compute the number of neighboring pixels equal to a given value."""
        neighbors = [
            self.on_the_right,
            self.on_the_left,
            self.on_the_bottom,
            self.on_the_top
        ]
        if self.connectivity == 8:
            neighbors.extend([
                self.on_the_topleft,
                self.on_the_topright,
                self.on_the_bottomleft,
                self.on_the_bottomright
            ])
        self.equal_neighbor_nb = np.zeros_like(self.matrix, dtype=np.uint8)
        for neighbor in neighbors:
            self.equal_neighbor_nb += (neighbor == value).astype(np.uint8)
        if and_itself:
            self.equal_neighbor_nb *= (self.matrix == value).astype(np.uint8)

def detect_nodes(skeleton):
    """
    Detect nodes in the skeletonized image.
    Nodes are pixels with one neighbor (endpoints) or more than two neighbors (junctions).
    """
    # Calculate the number of neighbors for each pixel in the skeleton
    cnv = CompareNeighborsWithValue(skeleton, 8)
    cnv.is_equal(1, and_itself=True)
    neighbor_counts = cnv.equal_neighbor_nb

    # Identify nodes: pixels with 1 or more than 2 neighbors
    nodes = ((neighbor_counts == 1) | (neighbor_counts > 2)) & skeleton

    # Label nodes
    labeled_nodes, num_labels = ndimage.label(nodes, structure=np.ones((3, 3), dtype=np.uint8))
    labeled_nodes = labeled_nodes.astype(np.int32)

    # Get node positions
    node_positions = ndimage.center_of_mass(nodes, labeled_nodes, range(1, num_labels + 1))
    node_positions = [tuple(map(int, pos)) for pos in node_positions]
    label_to_position = {label: pos for label, pos in zip(range(1, num_labels + 1), node_positions)}

    return labeled_nodes, label_to_position

def find_segments(skeleton, labeled_nodes, label_to_position):
    """
    Find segments in the skeleton by removing nodes and finding connected components.
    Each segment is associated with its coordinates and node labels.
    """
    # Remove nodes from the skeleton to get segments without nodes
    skeleton_wo_nodes = skeleton.copy()
    skeleton_wo_nodes[labeled_nodes > 0] = 0

    # Detect segments (connected components without nodes)
    num_labels, labels = cv2.connectedComponents(skeleton_wo_nodes.astype(np.uint8))
    labels = labels.astype(np.int32)
    segments = []

    for label in range(1, num_labels):
        segment_mask = (labels == label)
        coords = np.column_stack(np.where(segment_mask))

        # Dilate the segment to find adjacent nodes
        dilated_segment = morphology.binary_dilation(segment_mask, morphology.disk(2))
        overlapping_nodes = labeled_nodes * dilated_segment
        node_labels = np.unique(overlapping_nodes[overlapping_nodes > 0])

        if len(node_labels) >= 2:
            # If at least two nodes are connected, find the two farthest apart
            node_positions = [label_to_position[n_label] for n_label in node_labels]
            distances = np.sum((np.array(node_positions)[:, None] - np.array(node_positions)[None, :]) ** 2, axis=2)
            idx_max = np.unravel_index(np.argmax(distances), distances.shape)
            start_label = node_labels[idx_max[0]]
            end_label = node_labels[idx_max[1]]
            start_pos = label_to_position[start_label]
            end_pos = label_to_position[end_label]
        elif len(node_labels) == 1:
            # If only one node is connected
            start_label = node_labels[0]
            start_pos = label_to_position[start_label]
            distances = np.sum((coords - np.array(start_pos)) ** 2, axis=1)
            idx_max = np.argmax(distances)
            end_pos = tuple(coords[idx_max])
            end_label = None
        else:
            # Isolated segment without connected nodes
            distances = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            idx_max = np.unravel_index(np.argmax(distances), distances.shape)
            start_pos = tuple(coords[idx_max[0]])
            end_pos = tuple(coords[idx_max[1]])
            start_label = None
            end_label = None

        # Include node positions in segment coordinates
        coords = np.vstack([coords, start_pos])
        coords = np.vstack([coords, end_pos])
        coords = np.unique(coords, axis=0)

        segments.append({
            'start_label': start_label,
            'end_label': end_label,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'coords': coords
        })
    return segments

def extract_node_degrees(segments):
    """
    Calculate the degree of each node based on the number of connected segments.
    """
    node_degrees = defaultdict(int)
    for segment in segments:
        start_label = segment['start_label']
        end_label = segment['end_label']
        if start_label is not None:
            node_degrees[start_label] += 1
        if end_label is not None:
            node_degrees[end_label] += 1
    return node_degrees

def measure_segment_width(binary_image, segment, distance_map):
    """
    Measure the width of a segment by sampling perpendicular profiles along its length.
    """
    coords = segment['coords']
    distances = []
    for i in range(len(coords)):
        y, x = coords[i]
        if distance_map[y, x] == 0:
            continue
        # Handle special cases for dy, dx
        if len(coords) == 1:
            dy, dx = 0.0, 0.0
        elif i == 0:
            dy = coords[i+1][0] - y
            dx = coords[i+1][1] - x
        elif i == len(coords) - 1:
            dy = y - coords[i-1][0]
            dx = x - coords[i-1][1]
        else:
            dy = coords[i+1][0] - coords[i-1][0]
            dx = coords[i+1][1] - coords[i-1][1]
        dy = float(dy)
        dx = float(dx)
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        perp_dx = -dy / norm
        perp_dy = dx / norm
        length = distance_map[y, x] * 2.0  # Ensure length is a float
        if length < 1:
            continue
        r0 = y - perp_dy * length / 2.0
        c0 = x - perp_dx * length / 2.0
        r1 = y + perp_dy * length / 2.0
        c1 = x + perp_dx * length / 2.0
        line_length = int(np.hypot(r1 - r0, c1 - c0))
        if line_length == 0:
            continue
        line_coords = np.linspace(0, 1, line_length)
        rr = r0 + line_coords * (r1 - r0)
        cc = c0 + line_coords * (c1 - c0)
        # Check indices are within image bounds
        valid_idx = (rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])
        rr = rr[valid_idx]
        cc = cc[valid_idx]
        # Interpolate profile values
        from scipy.ndimage import map_coordinates
        profile = map_coordinates(binary_image.astype(float), [rr, cc], order=1, mode='constant', cval=0.0)
        # Determine width from interpolated profile
        threshold = 0.5  # Threshold to determine the presence of the vein
        width = np.sum(profile > threshold) * (length / line_length)  # Adjust based on actual length
        distances.append(width)
    if distances:
        widths = {
            'average_width': np.mean(distances),
            'width_node_A': distances[0],
            'width_node_B': distances[-1],
            'nb_widths_measured': len(distances)  # Number of width measurements
        }
        return widths
    else:
        return None

def compute_fractal_dimension(segment_mask):
    """
    Compute the fractal dimension of a binary image (segment) using the box-counting method.
    """
    def boxcount(Z, k):
        # Count the number of k x k boxes that contain at least one foreground pixel
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k),
            axis=1
        )
        return len(np.where(S > 0)[0])

    Z = (segment_mask > 0)
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    n = int(n)
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    if len(counts) > 1:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        fractal_dimension = -coeffs[0]
    else:
        fractal_dimension = np.nan
    return fractal_dimension

def process_frame(args):
    """
    Process a single frame: compute the skeleton, detect nodes and segments, measure segment widths,
    compute fractal dimensions, build the graph, calculate node hierarchy, and generate visualizations.
    """
    frame_idx, frame_data, output_dir = args
    print(f"Processing image {frame_idx + 1}")

    # Binarize the current frame
    binary_image = (frame_data == 2).astype(bool)

    # Create the directory for images if it does not exist
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Save the original image
    fig_original, ax_original = plt.subplots(figsize=(8, 8))
    ax_original.imshow(frame_data, cmap='gray')
    ax_original.set_title(f'Image {frame_idx + 1}: Original Image')
    ax_original.axis('off')
    fig_original.savefig(os.path.join(img_dir, f'Image{frame_idx + 1}_original.png'))
    plt.close(fig_original)

    # Skeletonization
    skeleton = morphology.skeletonize(binary_image)

    # Save the skeleton image
    fig_skeleton, ax_skeleton = plt.subplots(figsize=(8, 8))
    ax_skeleton.imshow(skeleton, cmap='gray')
    ax_skeleton.set_title(f'Image {frame_idx + 1}: Skeleton')
    ax_skeleton.axis('off')
    fig_skeleton.savefig(os.path.join(img_dir, f'Image{frame_idx + 1}_skeleton.png'))
    plt.close(fig_skeleton)

    # Node detection
    labeled_nodes, label_to_position = detect_nodes(skeleton)

    # Save skeleton with detected nodes
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(skeleton, cmap='gray')
    node_positions = np.array(list(label_to_position.values()))
    if node_positions.size > 0:
        ax2.scatter(node_positions[:, 1], node_positions[:, 0], c='red', s=1, label='Detected Nodes')
        ax2.legend()
    ax2.set_title(f'Image {frame_idx + 1}: Skeleton with Detected Nodes')
    ax2.axis('off')
    fig2.savefig(os.path.join(img_dir, f'Image{frame_idx + 1}_skeleton_nodes.png'))
    plt.close(fig2)

    # Find segments
    segments = find_segments(skeleton, labeled_nodes, label_to_position)

    # Compute node degrees
    node_degrees = extract_node_degrees(segments)

    # Center of the image
    center_y, center_x = np.array(binary_image.shape) // 2

    # Prepare lists to store information
    nodes_info = []
    segments_info = []
    fractal_info = []

    # Node information
    for label, pos in label_to_position.items():
        degree = node_degrees.get(label, 0)
        node_info = {
            'Frame': frame_idx + 1,
            'Node_ID': int(label),
            'X': float(pos[1]),
            'Y': float(pos[0]),
            'Degree': int(degree)
        }
        nodes_info.append(node_info)

    nodes_df = pd.DataFrame(nodes_info)

    # Process each segment
    for idx_segment, segment in enumerate(segments, start=1):
        start_label = segment['start_label']
        end_label = segment['end_label']
        start_pos = segment['start_pos']
        end_pos = segment['end_pos']
        # Obtain degrees, set to 0 if node label is None
        degree1 = node_degrees.get(start_label, 0) if start_label is not None else 0
        degree2 = node_degrees.get(end_label, 0) if end_label is not None else 0
        coords = segment['coords']

        # Create mask for the segment
        segment_mask = np.zeros_like(binary_image, dtype=bool)
        segment_mask[coords[:, 0], coords[:, 1]] = True
        vein_mask = morphology.binary_dilation(segment_mask, morphology.disk(3)) & binary_image
        distance_map = ndimage.distance_transform_edt(vein_mask)

        # Measure segment width
        width_measure = measure_segment_width(binary_image, segment, distance_map)

        # Compute fractal dimension of the segment
        fractal_dimension = compute_fractal_dimension(segment_mask)

        # Compute distance from center
        mean_y = np.mean(coords[:, 0])
        mean_x = np.mean(coords[:, 1])
        distance_from_center = np.hypot(mean_y - center_y, mean_x - center_x)

        # Add segment information, including start and end node IDs
        segment_info = {
            'Frame': frame_idx + 1,
            'Segment_ID': idx_segment,
            'Start_Node': int(start_label) if start_label is not None else np.nan,
            'End_Node': int(end_label) if end_label is not None else np.nan,
            'Start_X': start_pos[1],
            'Start_Y': start_pos[0],
            'End_X': end_pos[1],
            'End_Y': end_pos[0],
            'Degree_A': degree1,
            'Degree_B': degree2,
            'Average_Width': width_measure['average_width'] if width_measure else np.nan,
            'Width_Node_A': width_measure['width_node_A'] if width_measure else np.nan,
            'Width_Node_B': width_measure['width_node_B'] if width_measure else np.nan,
            'Nb_Measurements': width_measure['nb_widths_measured'] if width_measure else 0,
            'Distance_from_Center': distance_from_center
        }
        segments_info.append(segment_info)

        # Add fractal dimension information
        fractal_info.append({
            'Segment_ID': idx_segment,
            'Fractal_Dimension': fractal_dimension
        })

    segments_df = pd.DataFrame(segments_info)
    fractal_df = pd.DataFrame(fractal_info)

    # Build the graph
    G = nx.Graph()

    # Add nodes
    node_ids = nodes_df['Node_ID'].tolist()
    G.add_nodes_from(node_ids)

    # Add edges (segments)
    for idx, row in segments_df.iterrows():
        start_node = row['Start_Node']
        end_node = row['End_Node']
        # Check that nodes exist and are valid
        if pd.notna(start_node) and pd.notna(end_node):
            start_node = int(start_node)
            end_node = int(end_node)
            if start_node in G.nodes and end_node in G.nodes:
                G.add_edge(start_node, end_node)

    # Determine the central node (nearest to the center of the image)
    center_x_nodes = nodes_df['X'].mean()
    center_y_nodes = nodes_df['Y'].mean()
    nodes_df['Distance_to_Image_Center'] = np.hypot(nodes_df['X'] - center_x_nodes, nodes_df['Y'] - center_y_nodes)
    if not nodes_df.empty:
        center_node_id = nodes_df.loc[nodes_df['Distance_to_Image_Center'].idxmin(), 'Node_ID']
    else:
        print(f"No nodes found in image {frame_idx + 1}")
        return

    # Calculate hierarchical distances from the central node
    try:
        lengths = nx.single_source_shortest_path_length(G, center_node_id)
    except Exception as e:
        print(f"Error computing shortest paths in image {frame_idx + 1}: {e}")
        nodes_df['Hierarchical_Distance'] = np.nan
        # Save nodes_df even in case of error
        nodes_df.to_csv(os.path.join(output_dir, f'nodes_info_frame_{frame_idx + 1}.csv'), index=False)
        segments_df.to_csv(os.path.join(output_dir, f'segments_info_frame_{frame_idx + 1}.csv'), index=False)
        fractal_df.to_csv(os.path.join(output_dir, f'fractal_info_frame_{frame_idx + 1}.csv'), index=False)
        return

    # Add the hierarchical distance to the nodes
    lengths = {int(k): v for k, v in lengths.items()}  # Ensure keys are integers
    nodes_df['Hierarchical_Distance'] = nodes_df['Node_ID'].map(lengths)

    # Save DataFrames to CSV
    nodes_df.to_csv(os.path.join(output_dir, f'nodes_info_frame_{frame_idx + 1}.csv'), index=False)
    segments_df.to_csv(os.path.join(output_dir, f'segments_info_frame_{frame_idx + 1}.csv'), index=False)
    fractal_df.to_csv(os.path.join(output_dir, f'fractal_info_frame_{frame_idx + 1}.csv'), index=False)

    return

def main():
    """
    Main function to process all frames in the dataset in parallel.
    """
    # Load the complete data
    data = np.load('network2.npy')
    num_frames = data.shape[0]

    # Directory to save outputs
    output_dir = 'HoussamAnalyse'
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    frames = [(i, data[i], output_dir) for i in range(num_frames)]

    # Number of processes (adjusted to 40 cores)
    num_processes = 40  # Adjust this number as needed

    # Use multiprocessing to process images in parallel
    with Pool(processes=num_processes) as pool:
        pool.map(process_frame, frames)

    print("Analysis completed. The CSV files and images have been saved for each image.")

if __name__ == "__main__":
    main()
