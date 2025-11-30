import numpy as np
import pandas as pd
import scipy.io
import os
import networkx as nx
from nilearn.connectome import ConnectivityMeasure

# thresholds
t1 = 0.1
t2 = 0.75
Delta = 0.01
features_gesamt = 580 * int((t2 - t1) / Delta)
features_array = np.zeros((1, features_gesamt))
labels = []

for j, file_name in enumerate(filtered_list):
    
    degree_list = []
    pagerank_list = []
    closeness_list = []
    betweenness_list = []
    clustering_list = []
    mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
    matrix = mat['ROISignals']
    data = pd.DataFrame(matrix)

    aal_array = matrix[:, :116]  # 1~116: Automated Anatomical Labeling (AAL) atlas (Tzourio-Mazoyer et al., 2002)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([aal_array])[0]

    threshold = t1
    while threshold < t2:
        # apply threshold
        thresholded_matrix = np.where(correlation_matrix >= threshold, 1, 0)
        np.fill_diagonal(thresholded_matrix, 0)
        graph = nx.from_numpy_array(thresholded_matrix)

        # Degree Centrality
        degree_list.extend([degree for node, degree in nx.degree(graph)])

        # PageRank Centrality
        pagerank_list.extend([pagerank for node, pagerank in nx.pagerank(graph).items()])

        # Closeness Centrality
        closeness_list.extend([closeness for node, closeness in nx.closeness_centrality(graph).items()])

        # Betweenness Centrality
        betweenness_list.extend([betweenness for node, betweenness in nx.betweenness_centrality(graph).items()])

        # Clustering Coefficient
        clustering_list.extend([clustering for node, clustering in nx.clustering(graph).items()])

        # increase threshold
        threshold += Delta

    big_list = [degree_list, pagerank_list, closeness_list, betweenness_list, clustering_list]
    array = np.concatenate([np.array(list_).reshape(1, -1) for list_ in big_list], axis=1)

    features_array = np.concatenate((features_array, array[:, 1:]), axis=0)
    zahl = int(file_name.split('-')[1])
    labels.append(zahl)

    print(f"Verarbeitet: {j}/{len(filtered_list)}")

features_array = features_array[1:, :] # remove first row with zeros