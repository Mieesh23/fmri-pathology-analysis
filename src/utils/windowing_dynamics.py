from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import community
import scipy.io
import pickle
import re
import os

def filter_strings_not_in_list(strings_list, substrings_list):
    filtered_list = []
    for s in strings_list:
        if not any(substring in s for substring in substrings_list):
            filtered_list.append(s)
    return filtered_list
folder_path = '/path_to/ROISignals_FunImgARCWF'  # path to data
#folder_path = '/path_to/ROISignals_FunImgARglobalCWF'
strings_list = []
for i, file_name in enumerate(os.listdir(folder_path)):
    if i % 2 == 0:
        strings_list.append(file_name)
    
with open('/path_to/first_id_list.pkl', 'rb') as file:
    data_list_substrings = pickle.load(file)

filtered_list = filter_strings_not_in_list(strings_list, data_list_substrings)

corrs_dic = {}
num_windows = 5

for j, file_name in enumerate(filtered_list[:1300]):
    print(file_name)
    pattern = r"S\d+"
    match = re.search(pattern, file_name)
    s_part = match.group()[1:]
    
    if int(s_part) != 16:
        if int(s_part) != 25:
            #if int(file_name.split('-')[1]) == 1:
            mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
            matrix = mat['ROISignals']

            aal_array = matrix[:, :116]       # 1~116: Automated Anatomical Labeling (AAL) atlas (Tzourio-Mazoyer et al., 2002)
            #hoac_array = matrix[:, 116:212]   # 117~212: Harvard-Oxford atlas (Kennedy et al., 1998)– cortical areas
            #hoas_array = matrix[:, 212:228]   # 213~228: Harvard-Oxford atlas (Kennedy et al., 1998)– subcortical areas
            #ccl_array = matrix[:, 228:428]    # 229~428: Craddock’s clustering 200 ROIs (Craddock et al., 2012)
            #zrp_array = matrix[:, 428:1408]   # 429~1408: Zalesky’s random parcelations (compact version: 980 ROIs) (Zalesky et al., 2010)
            #dbf_array = matrix[:, 1408:1568]  # 1409~1568: Dosenbach’s 160 functional ROIs (Dosenbach et al., 2010)

            correlation_measure = ConnectivityMeasure(kind='correlation')
            corrs_list = []

            for t_wind in range(num_windows):
                one_length = int(aal_array.shape[0]*(1/num_windows))
                correlation_matrix=correlation_measure.fit_transform([aal_array[t_wind*one_length:(t_wind+1)*one_length,:]])[0]
                np.fill_diagonal(correlation_matrix, 0)
                if t_wind == 0:
                    five_corrs_array = correlation_matrix.reshape((-1,1))
                else:
                    five_corrs_array = np.hstack((five_corrs_array,correlation_matrix.reshape((-1,1))))

            five_corrs_correlation_matrix = correlation_measure.fit_transform([five_corrs_array])[0]
            np.fill_diagonal(five_corrs_correlation_matrix, 0)
            
            num_type = int(file_name.split('-')[1])
            if num_type == 1:
                sub_type = "MDD"
            else:
                sub_type = "HC"
                
            # Heatmap 
            plotting.plot_matrix(five_corrs_correlation_matrix, figure=(5, 5), colorbar=True, title=f"Heatmap für {sub_type}")
            plotting.show()
            
    if (j+1) % 200 == 0:
        print(j+1)