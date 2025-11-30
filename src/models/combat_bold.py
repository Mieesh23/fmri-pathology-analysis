from nilearn.connectome import ConnectivityMeasure
from neurocombat_sklearn import CombatModel
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import community
import scipy.io
import re
import os
import pickle

excel_file = '/path_to/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx'
sheet1_name = 'MDD'
sheet2_name = 'Controls'
df1 = pd.read_excel(excel_file, sheet_name=sheet1_name)
df2 = pd.read_excel(excel_file, sheet_name=sheet2_name)

data_dict1 = {row['ID']: (row['Sex'], row['Age']) for _, row in df1.iterrows()}
data_dict2 = {row['ID']: (row['Sex'], row['Age']) for _, row in df2.iterrows()}

def gender_and_age(string):
    gender = None
    num = None
    if string in data_dict1:
        sex, num = data_dict1[string]
    elif string in data_dict2:
        sex, num = data_dict2[string]
    else:
        return None, None
    if sex == 1:
        gender = 0  # male
    elif sex == 2:
        gender = 1  # female
    agegroup = min(9, num // 10)
    return gender, agegroup

# load information
folder_path = '/path_to/ROISignals_FunImgARCWF' 
strings_list = os.listdir(folder_path)
with open('/path_to/first_id_list.pkl', 'rb') as file:
    data_list_substrings = pickle.load(file)

filtered_list = [file_name for file_name in strings_list if not any(substring in file_name for substring in data_list_substrings)]
print(len(filtered_list))
labels = []
features_list_one = []
features_list_two = []
gender_Test = []
gender_Train = []
age_Test = []
age_Train = []
location_Test = []
location_Train = []

for j, file_name in enumerate(filtered_list[:2600]):
    if j % 2 == 0:
        mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
        matrix = mat['ROISignals']
        aal_array = matrix[:, :116]  # 1~116: Automated Anatomical Labeling (AAL) atlas (Tzourio-Mazoyer et al., 2002)
        #hoac_array = matrix[:, 116:212]   # 117~212: Harvard-Oxford atlas (Kennedy et al., 1998)– cortical areas
        #hoas_array = matrix[:, 212:228]   # 213~228: Harvard-Oxford atlas (Kennedy et al., 1998)– subcortical areas
        #ccl_array = matrix[:, 228:428]    # 229~428: Craddock’s clustering 200 ROIs (Craddock et al., 2012)
        #zrp_array = matrix[:, 428:1408]   # 429~1408: Zalesky’s random parcelations (compact version: 980 ROIs) (Zalesky et al., 2010)
        #dbf_array = matrix[:, 1408:1568]  # 1409~1568: Dosenbach’s 160 functional ROIs (Dosenbach et al., 2010)

        zahl = int(file_name.split('-')[1])
        pattern = r"S\d+"
        match = re.search(pattern, file_name)
        s_part = match.group()[1:]
        desired_part = file_name.split('_')[1].split('.')[0]
        gendergroup, agegroup = gender_and_age(desired_part)
        if int(s_part) != 16:
            if int(s_part) != 25:
                if zahl == 1:
                    features_list_one.append(aal_array[:140,:].flatten())
                    gender_Test.append(gendergroup)
                    age_Test.append(agegroup)
                    location_Test.append(int(s_part))
                else:
                    features_list_two.append(aal_array[:140,:].flatten())
                    gender_Train.append(gendergroup)
                    age_Train.append(agegroup)
                    location_Train.append(int(s_part))

        if (j+1)/2 % 200 == 0:
            print(f'----{(j+1)/2} Done!')
        print(j/2)

for i, array_ in enumerate(features_list_one):
    if i == 0:
        X_Test = array_.reshape((1,-1))
    else:
        X_Test = np.vstack((X_Test,array_.reshape((1,-1))))

for i, array_ in enumerate(features_list_two):
    if i == 0:
        X_Train = array_.reshape((1,-1))
    else:
        X_Train = np.vstack((X_Train,array_.reshape((1,-1))))

def from_matrix_to_correlation(Matrix):
    for row in range(Matrix.shape[0]):
        aal_array = Matrix[row,:].reshape((140,116))
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([aal_array])[0]
        np.fill_diagonal(correlation_matrix, 0)
        upper_triangle = np.triu(correlation_matrix, k=1)
        upper_triangle_vector = upper_triangle[np.triu_indices_from(upper_triangle)]
        if row == 0:
            feature_array = upper_triangle_vector
        else:
            feature_array = np.vstack((feature_array,upper_triangle_vector))
    return feature_array

# creating model
model = CombatModel()
# fitting the model and transforming the training set
X_train_harmonized = model.fit_transform(X_Train, np.array(location_Train).reshape((-1,1)))
# harmonize test set using training set fitted parameters
X_test_harmonized = model.transform(X_Test, np.array(location_Test).reshape((-1,1)))
feature_hc = from_matrix_to_correlation(X_train_harmonized)
label_hc = np.ones((feature_hc.shape[0],1))
feature_mdd = from_matrix_to_correlation(X_test_harmonized)
label_mdd = np.zeros((feature_mdd.shape[0],1))
feature_matrix_hc = np.concatenate((feature_hc,label_hc), axis = 1)
feature_matrix_mdd = np.concatenate((feature_mdd,label_mdd), axis = 1)
feature_matrix = np.vstack((feature_matrix_hc,feature_matrix_mdd))
