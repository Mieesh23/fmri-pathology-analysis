from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import community
import scipy.io
import re
import os
import pickle

def filter_strings_not_in_list(strings_list, substrings_list):
    filtered_list = []
    for s in strings_list:
        if not any(substring in s for substring in substrings_list):
            filtered_list.append(s)
    return filtered_list

folder_path = '/path_to/Results/ROISignals_FunImgARCWF'  # Pfad zum Ordner
#folder_path = '/path_to/ROISignals_FunImgARglobalCWF'
strings_list = []
for file_name in os.listdir(folder_path):
    strings_list.append(file_name)
with open('/path_to/first_id_list.pkl', 'rb') as file:
    data_list_substrings = pickle.load(file)
filtered_list = filter_strings_not_in_list(strings_list, data_list_substrings)
print(len(filtered_list))

labels = []
for k, file_name in enumerate(filtered_list[:1300]):

    mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
    matrix = mat['ROISignals']

    aal_array = matrix[:, :116]  # 1~116: Automated Anatomical Labeling (AAL) atlas (Tzourio-Mazoyer et al., 2002)
    #hoac_array = matrix[:, 116:212]   # 117~212: Harvard-Oxford atlas (Kennedy et al., 1998)– cortical areas
    #hoas_array = matrix[:, 212:228]   # 213~228: Harvard-Oxford atlas (Kennedy et al., 1998)– subcortical areas
    #ccl_array = matrix[:, 228:428]    # 229~428: Craddock’s clustering 200 ROIs (Craddock et al., 2012)
    #zrp_array = matrix[:, 428:1408]   # 429~1408: Zalesky’s random parcelations (compact version: 980 ROIs) (Zalesky et al., 2010)
    #dbf_array = matrix[:, 1408:1568]  # 1409~1568: Dosenbach’s 160 functional ROIs (Dosenbach et al., 2010)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([aal_array])[0]
    threshold = 0.35  
    thresholded_matrix = np.where(correlation_matrix < threshold, 0, correlation_matrix)
    np.fill_diagonal(thresholded_matrix, 0)
    graph = nx.Graph()  # direct Graph
    num_nodes = thresholded_matrix.shape[0]
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = thresholded_matrix[i][j]  # weights of edges
            if weight != 0:
                graph.add_edge(i, j, weight=weight)

    big_list =[]
    zahl = int(file_name.split('-')[1])
    big_list=np.array([list(dict(nx.degree(graph)).values()), list(nx.pagerank(graph).values()), list(nx.closeness_centrality(graph).values()), list(nx.betweenness_centrality(graph).values()), list(nx.clustering(graph).values())]).flatten()

    if k == 0:
        features_array = big_list
    else:
        features_array = np.vstack((features_array, big_list))

    labels.append(int(file_name.split('-')[1]))
    if (k+1) % 200 == 0:
        print(k+1)  

