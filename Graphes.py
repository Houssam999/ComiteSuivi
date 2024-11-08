#!/usr/bin/env python3
"""
Ce script traite les données des nœuds et des segments pour chaque frame afin de générer :
- Histogrammes des degrés des nœuds.
- Histogrammes des largeurs moyennes des segments.
- Graphiques de la largeur moyenne des segments en fonction de la distance au centre.
- Histogrammes du nombre de mesures de largeur par segment.


Le script utilise le multiprocessing pour traiter les frames en parallèle.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import networkx as nx

def process_frame(frame_idx):
    """
    Traite une seule frame :
    - Charge les données des nœuds et des segments.
    - Génère les histogrammes des degrés des nœuds et des largeurs des segments.
    - Calcule le plus court chemin du nœud central à tous les autres nœuds.
    """
    frame_number = frame_idx + 1
    print(f"Traitement de la frame {frame_number}")

    # Répertoires
    input_dir = 'HoussamAnalyse'  # Dossier contenant les données d'entrée
    output_dir = 'HenniGraphs'    # Dossier où les sorties seront enregistrées
    img_dir = os.path.join(output_dir, 'plots')

    # Créer les répertoires de sortie avec exist_ok=True
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Chemins des fichiers
    nodes_file = os.path.join(input_dir, f'nodes_info_frame_{frame_number}.csv')
    segments_file = os.path.join(input_dir, f'segments_info_frame_{frame_number}.csv')

    # Vérifier si les fichiers existent
    if not os.path.exists(nodes_file) or not os.path.exists(segments_file):
        print(f"Fichiers de données pour la frame {frame_number} non trouvés.")
        return

    # Charger les données des nœuds et des segments
    nodes_df = pd.read_csv(nodes_file)
    segments_df = pd.read_csv(segments_file)

    # Vérifier que Node_ID est unique et de type int
    nodes_df['Node_ID'] = nodes_df['Node_ID'].astype(int)
    if nodes_df['Node_ID'].duplicated().any():
        print(f"Avertissement : Des Node_ID dupliqués ont été trouvés dans la frame {frame_number}.")
        nodes_df = nodes_df.drop_duplicates(subset=['Node_ID'])

    # Générer l'histogramme des degrés des nœuds
    degrees = nodes_df['Degree']
    plt.figure()
    plt.hist(degrees, bins=range(int(degrees.min()), int(degrees.max()) + 2), edgecolor='black', align='left')
    plt.title(f'Frame {frame_number} : Distribution des degrés des nœuds')
    plt.xlabel('Degré')
    plt.ylabel('Nombre de nœuds')
    plt.savefig(os.path.join(img_dir, f'frame_{frame_number}_node_degree_distribution.png'))
    plt.close()

    # Générer l'histogramme des largeurs moyennes des segments
    widths = segments_df['Average_Width'].dropna()
    if not widths.empty:
        plt.figure()
        plt.hist(widths, bins=30, edgecolor='black')
        plt.title(f'Frame {frame_number} : Distribution des largeurs moyennes des segments')
        plt.xlabel('Largeur moyenne')
        plt.ylabel('Nombre de segments')
        plt.savefig(os.path.join(img_dir, f'frame_{frame_number}_average_segment_width_distribution.png'))
        plt.close()

    # Calculer Mean_X et Mean_Y pour chaque segment
    segments_df['Mean_X'] = (segments_df['Start_X'] + segments_df['End_X']) / 2
    segments_df['Mean_Y'] = (segments_df['Start_Y'] + segments_df['End_Y']) / 2

    # Graphe de la largeur moyenne des segments en fonction de la distance au centre
    if not segments_df.empty:
        # Calculer le centre de l'image
        center_y, center_x = segments_df['Mean_Y'].mean(), segments_df['Mean_X'].mean()
        # Calculer la distance au centre pour chaque segment
        segments_df['Distance_to_Center'] = np.hypot(segments_df['Mean_Y'] - center_y, segments_df['Mean_X'] - center_x)
        plt.figure()
        plt.scatter(segments_df['Distance_to_Center'], segments_df['Average_Width'])
        plt.title(f'Frame {frame_number} : Largeur moyenne vs Distance au centre')
        plt.xlabel('Distance au centre')
        plt.ylabel('Largeur moyenne du segment')
        plt.savefig(os.path.join(img_dir, f'frame_{frame_number}_width_vs_distance.png'))
        plt.close()

    # Générer l'histogramme du nombre de mesures de largeur par segment
    nb_measurements = segments_df['Nb_Measurements'].dropna()
    if not nb_measurements.empty:
        plt.figure()
        plt.hist(nb_measurements, bins=range(int(nb_measurements.min()), int(nb_measurements.max()) + 2), edgecolor='black', align='left')
        plt.title(f'Frame {frame_number} : Nombre de mesures de largeur par segment')
        plt.xlabel('Nombre de mesures')
        plt.ylabel('Nombre de segments')
        plt.savefig(os.path.join(img_dir, f'frame_{frame_number}_number_of_width_measurements_distribution.png'))
        plt.close()

  
    print(f"Frame {frame_number} traitée.")

def main():
    """
    Fonction principale pour traiter toutes les frames en parallèle.
    """
    # Dossiers d'entrée et de sortie
    input_dir = 'HoussamAnalyse'
    output_dir = 'HenniGraphs'

    # S'assurer que le dossier de sortie existe
    os.makedirs(output_dir, exist_ok=True)

    # Déterminer le nombre de frames en fonction des fichiers existants
    files = os.listdir(input_dir)
    nodes_files = [f for f in files if f.startswith('nodes_info_frame_')]
    num_frames = len(nodes_files)

    # Préparer les indices des frames
    frame_indices = list(range(num_frames))

    # Nombre de processus (ajuster en fonction des ressources disponibles)
    num_processes = 40  

    # Utiliser multiprocessing pour traiter les frames en parallèle
    with Pool(processes=num_processes) as pool:
        pool.map(process_frame, frame_indices)

    print("Toutes les frames ont été traitées.")

if __name__ == "__main__":
    main()