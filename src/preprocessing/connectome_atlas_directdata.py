import scipy.io
import pandas as pd
import numpy as np
from nilearn import datasets, plotting
from nilearn.connectome import ConnectivityMeasure

# load matrix from MATLAB-data
mat = scipy.io.loadmat('/path_to/ROISignals_patient_x.mat')
matrix = mat['ROISignals']
data = pd.DataFrame(matrix)

# choice of atlas from DirectDataset
aal_df = data.iloc[:, :116]       # 1~116: Automated Anatomical Labeling (AAL) atlas (Tzourio-Mazoyer et al., 2002)
hoac_df = data.iloc[:, 116:212]   # 117~212: Harvard-Oxford atlas (Kennedy et al., 1998)– cortical areas
hoas_df = data.iloc[:, 212:228]   # 213~228: Harvard-Oxford atlas (Kennedy et al., 1998)– subcortical areas
ccl_df = data.iloc[:, 228:428]    # 229~428: Craddock’s clustering 200 ROIs (Craddock et al., 2012)
zrp_df = data.iloc[:, 428:1408]   # 429~1408: Zalesky’s random parcelations (compact version: 980 ROIs) (Zalesky et al., 2010)
dbf_df = data.iloc[:, 1408:1568]  # 1409~1568: Dosenbach’s 160 functional ROIs (Dosenbach et al., 2010)
glob_df = pd.DataFrame(data.iloc[:, 1568])      # 1569: Global signal (Since DPARSF V4.2)
powf_df = pd.DataFrame(data.iloc[:, 1569:])     # 1570~1833: Power’s 264 functional ROIs (Power et al., 2011) (Since DPARSF V4.3)

dfs = [aal_df, hoac_df, hoas_df, ccl_df, zrp_df, dbf_df, glob_df, powf_df]

for i, df in enumerate(dfs):
    df.columns = range(0, len(df.columns))
    
aal_array = aal_df.to_numpy()
hoac_array = hoac_df.to_numpy()
hoas_array = hoas_df.to_numpy()
ccl_array = ccl_df.to_numpy()
zrp_array = zrp_df.to_numpy()
dbf_array = dbf_df.to_numpy()
glob_array = glob_df.to_numpy()
powf_array = powf_df.to_numpy()

atlas_arrays = [aal_array, hoac_array, hoas_array, ccl_array, zrp_array, dbf_array, glob_array, powf_array]

# compute correlation
correlation_matrices = []
for atlas_array in atlas_arrays:
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([atlas_array])[0]
    correlation_matrices.append(correlation_matrix)

for i, correlation_matrix in enumerate(correlation_matrices):
    atlas_name = ['AAL', 'HOAC', 'HOAS', 'CCL', 'ZRP', 'DBF', 'GLOB', 'POWF'][i] 
    #print(f"Korrelationsmatrix für {atlas_name}:")
    #print(correlation_matrix)

correlation_matrix = correlation_measure.fit_transform([aal_array])[0]

# load AAL-Atlas-Labels
aal = datasets.fetch_atlas_aal()
#roi_names = aal.labels

# compute heatmaps
for i, correlation_matrix in enumerate(correlation_matrices):
    atlas_name = ['AAL', 'HOAC', 'HOAS', 'CCL', 'ZRP', 'DBF', 'GLOB', 'POWF'][i] 
    plotting.plot_matrix(correlation_matrix, figure=(5, 5), colorbar=True, title=f"Heatmap für {atlas_name}")
    plotting.show()
