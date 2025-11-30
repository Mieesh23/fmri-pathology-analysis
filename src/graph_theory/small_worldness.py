from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io
import pickle
import os

def filter_strings_not_in_list(strings_list, substrings_list):
    filtered_list = []
    for s in strings_list:
        if not any(substring in s for substring in substrings_list):
            filtered_list.append(s)
    return filtered_list

def average_shortest_path_length(G):
    try:
        return nx.average_shortest_path_length(G)
    except nx.NetworkXError:  
        return np.inf

def small_worldness(G):
    C = nx.average_clustering(G)
    L = average_shortest_path_length(G)
    random_graph = nx.erdos_renyi_graph(G.number_of_nodes(), nx.density(G))
    C_rand = nx.average_clustering(random_graph)
    L_rand = average_shortest_path_length(random_graph)
    if L_rand == 0: 
        return np.nan
    try:
        return (C / C_rand) / (L / L_rand)
    except:
        return np.nan

folder_path = '/path_to/ROISignals_FunImgARCWF'  # path to data
strings_list = []
for i, file_name in enumerate(os.listdir(folder_path)):
    if i % 2 == 0:
        strings_list.append(file_name)
    
with open('/path_to/WlllNeuro Quest/id_list_T3final.pkl', 'rb') as file:
    data_list_substrings = pickle.load(file)
filtered_list = filter_strings_not_in_list(strings_list, data_list_substrings)

mdd_values = []
hc_values = []

for i, file_name in enumerate(filtered_list):
    mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
    matrix = mat['ROISignals']
    aal_array = matrix[:, :116]
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([aal_array])[0]
    status = int(file_name.split('-')[1]) # 1 == Mdd, 2 == HC
    sw_values = []

    for factor in range(0, 65):
        threshold = factor/100
        thresholded_matrix = np.where(correlation_matrix < threshold, 0, 1)
        np.fill_diagonal(thresholded_matrix, 0)
        G = nx.from_numpy_array(thresholded_matrix)
        if len(G) > 0:
            sw = small_worldness(G)
            sw_values.append(sw)
        else:
            sw_values.append(np.nan)

    if status == 1:
        mdd_values.append(sw_values)
    else:
        hc_values.append(sw_values)
    print(f'{i} Done!')

t = 65
mdd_values = np.array(mdd_values)
hc_values = np.array(hc_values)

mdd_mean = np.nanmean(mdd_values, axis=0)[:t]
hc_mean = np.nanmean(hc_values, axis=0)[:t]

mdd_std = np.nanstd(mdd_values, axis=0)[:t]
hc_std = np.nanstd(hc_values, axis=0)[:t]

thresholds = np.arange(0, 0.65, 0.01)[:t]
plt.figure(figsize=(10,5),dpi=300)
plt.errorbar(thresholds, mdd_mean, yerr=mdd_std, color='#B60909', label='MDD', alpha=1, elinewidth=2.5)
plt.errorbar(thresholds, hc_mean, yerr=hc_std, color='#FF998C', label='HC', alpha=0.85,elinewidth=1.5)
#plt.errorbar(thresholds, mdd_mean, yerr=mdd_std, color='#FF998C', label='MDD', alpha=1)
#plt.errorbar(thresholds, hc_mean, yerr=hc_std, color='#B60909', label='HC', alpha=0.25)
plt.xlabel('Schwellwert')
plt.ylabel('Small Worldness')
plt.legend()
plt.grid(axis='y')
plt.title('compare Small-Worldness between MDD & HC over different thresholds')
plt.tight_layout()
plt.xlim(0, max(thresholds))
plt.axhline(y=1, color='black', linestyle='--')
plt.savefig('/path_to/data_tools/Abbts')
plt.show()