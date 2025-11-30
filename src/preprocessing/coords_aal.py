import nibabel as nib
import numpy as np

# load nifti-data
aal_img = nib.load('/path_to/aal_for_SPM12/atlas/AAL.nii')
aal_data = aal_img.get_fdata()

# coordinates: 116 regions
regions = region_label_numbers # list of ids
region_coords = []
for region_id in regions:
    region_mask = aal_data == region_id
    region_voxels = np.array(np.where(region_mask)).T
    region_center = np.mean(region_voxels, axis=0)
    region_coords.append(region_center)

region_coords = np.array(region_coords)
print(region_coords)

region_coords.shape