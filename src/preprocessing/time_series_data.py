import numpy as np
import scipy.io
import pickle
import os
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, plotting
import matplotlib.pyplot as plt

folder_path = '/path_to/ROISignals_FunImgARCWF' 
subject = 0
dict_correlation_matrices = {}

# loop over data
for file_name in os.listdir(folder_path):
    subject = subject + 1
    mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
    matrix = mat['ROISignals']
    data = pd.DataFrame(matrix)

    # choice of atlas
    aal_df = data.iloc[:, :116]       # 1~116: Automated Anatomical Labeling (AAL) atlas (Tzourio-Mazoyer et al., 2002)
    #hoac_df = data.iloc[:, 116:212]   # 117~212: Harvard-Oxford atlas (Kennedy et al., 1998)– cortical areas
    #hoas_df = data.iloc[:, 212:228]   # 213~228: Harvard-Oxford atlas (Kennedy et al., 1998)– subcortical areas
    #ccl_df = data.iloc[:, 228:428]    # 229~428: Craddock’s clustering 200 ROIs (Craddock et al., 2012)
    #zrp_df = data.iloc[:, 428:1408]   # 429~1408: Zalesky’s random parcelations (compact version: 980 ROIs) (Zalesky et al., 2010)
    #dbf_df = data.iloc[:, 1408:1568]  # 1409~1568: Dosenbach’s 160 functional ROIs (Dosenbach et al., 2010)
    #glob_df = pd.DataFrame(data.iloc[:, 1568])      # 1569: Global signal (Since DPARSF V4.2)
    #powf_df = pd.DataFrame(data.iloc[:, 1569:])     # 1570~1833: Power’s 264 functional ROIs (Power et al., 2011) (Since DPARSF V4.3)
    
    aal_array = aal_df.to_numpy()
    #hoac_array = hoac_df.to_numpy()
    #hoas_array = hoas_df.to_numpy()
    #ccl_array = ccl_df.to_numpy()
    #zrp_array = zrp_df.to_numpy()
    #dbf_array = dbf_df.to_numpy()
    #glob_array = glob_df.to_numpy()
    #powf_array = powf_df.to_numpy()

    dict_correlation_matrices[file_name] = aal_array
    # Heatmap 
    plotting.plot_matrix(correlation_matrix, figure=(5, 5), colorbar=True, title=f"Heatmap für {subject}. Datei")
    plotting.show()

# saving Dictionaries
#with open('dictionaries.pickle', 'wb') as file:
#    pickle.dump(dict_correlation_matrices, file)

dictionary = dict_correlation_matrices
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

for i, key in enumerate(list(dictionary.keys())[:3]):
    array = dictionary[key]

    #subset = array[:, :5]
    subset = array
    axs[i].plot(subset)

    axs[i].set_xlabel('Frames')
    axs[i].set_ylabel('Werte')
    axs[i].set_title('Array {}'.format(key))

plt.tight_layout()
plt.show()

dictionary = dict_correlation_matrices
mean_list = []
std_list = []
for i, key in enumerate(list(dictionary.keys())):
    mean_list.append(np.mean(dictionary[key]))
    std_list.append(np.std(dictionary[key]))
    
mittelwerte = mean_list
standardabweichungen = std_list

anzahl_regionen = len(mittelwerte)
ind = np.arange(anzahl_regionen)

breite = 0.35
fig, ax = plt.subplots(figsize=(10, 6)) 
balken_mittelwerte = ax.bar(ind, mittelwerte, breite, color='b', label='Mittelwerte')

for i in range(anzahl_regionen):
    ax.errorbar(ind[i]+breite/2, mittelwerte[i], yerr=standardabweichungen[i], color='r')

ax.set_xlabel('ROIs')
ax.set_ylabel('Werte')
ax.set_title('Mittelwerte und Standardabweichungen nach ROIs')
ax.set_xticks(ind)
ax.set_xticklabels([f'ROI {i+1}' for i in range(anzahl_regionen)], rotation=90)
ax.legend()
plt.tight_layout() 
plt.show()