from nilearn import plotting
import nibabel as nib
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
from nilearn.datasets import fetch_atlas_aal
from nilearn.maskers import NiftiLabelsMasker
import igraph as ig

# load AAL-masker from Nilearn
aal = datasets.fetch_atlas_aal()
aal_img = nib.load(aal.maps)
plotting.plot_roi(aal.maps)

# load Resting-State-fMRI-Data
resting_state = nib.load('/path_to/rfMRI_REST1_LR.nii.gz')
# create Masker
masker = NiftiLabelsMasker(labels_img=aal_img, standardize=True)
# ROI timeseries
aal_array = masker.fit_transform(resting_state)

# compute the correlation matrix
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([aal_array])[0]

# list of ROI names
aal = datasets.fetch_atlas_aal()
roi_names = aal.labels
aal = fetch_atlas_aal()

# get region labels and their corresponding label numbers
region_label = aal.indices
region_label_numbers = []
type(region_label_numbers)

for eintrag in region_label:
    integer_eintrag = int(eintrag)
    region_label_numbers.append(integer_eintrag)

# load the Nifti file
aal_img = nib.load('/path_to/ext-d000035_AAL1Atlas_pub-Release2018_SPM12/aal_for_SPM12/atlas/AAL.nii')
affine = aal_img.affine
aal_data = aal_img.get_fdata()

# extract coordinates of the 116 regions
regions = region_label_numbers # List of region IDs
region_coords = []
for region_id in regions:
    region_mask = aal_data == region_id
    region_voxels = np.array(np.where(region_mask)).T
    
    # convert voxel coordinates to physical coordinates
    region_coords_phys = nib.affines.apply_affine(affine, region_voxels)
    region_center = np.mean(region_coords_phys, axis=0)
    region_coords.append(region_center)

region_coords = np.array(region_coords)
# creating graph g
g = ig.Graph()

# adding edges
num_nodes = 116
g.add_vertices(num_nodes)
node_coords = region_coords

# nodes' coords
g.vs['x'] = node_coords[:, 0]
g.vs['y'] = node_coords[:, 1]
g.vs['z'] = node_coords[:, 2]

# edges
correlation_matrix
threshold = 0.6
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if abs(correlation_matrix[i, j]) >= threshold:
            g.add_edge(i, j)

# Heatmap
plotting.plot_matrix(correlation_matrix, figure=(15, 15), labels=roi_names, colorbar=True)
# Graph
plotting.view_connectome(correlation_matrix, edge_threshold=0.6,
                         node_coords=region_coords)