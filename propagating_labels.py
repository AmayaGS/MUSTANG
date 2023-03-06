# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:13:57 2023

@author: AmayaGS
"""


import os
import pandas as pd
import math


# %%

stains = ["CD138", "CD68", "CD20", "HE"]
df_all_stains = []

for stain in stains:

    main_dir = r"C:/Users/Amaya/Documents/PhD/Data/QuPath " + stain + "/tiles"
    csv_file = r"C:\Users/Amaya/Documents/PhD/Data/" + stain + "/" + stain + "_ID_labels_relabeling.csv"
    labels = r"C:\Users\Amaya\Documents\PhD\Data\patient_labels.csv"
    
    # %%

    folder_list = [f.path for f in os.scandir(main_dir)]
    df = pd.read_csv(csv_file, encoding='windows-1252')
    df_name = df[df.columns[:5]]
    df_labels = pd.read_csv(labels)
    
    # %%
    
    # patch_names = []
    # for location in folder_list:
    #    patch_names.append([location.split("\\")[-1].split(" ")[0].split("_")[0], location.split("\\")[-1][:-4], location])
            
    # %% 
    
    patch_names = []
    for location in folder_list:
        patch_list = os.listdir(location)
        for patch in patch_list:
            patch_names.append([patch.split(" ")[0].split("_")[0], patch[0:-4], os.path.join(location, patch)])
    
    # %%
    
    patch_names_df = pd.DataFrame(patch_names)
    patch_names_df.columns= ['Patient ID', 'Patch_name', 'Location']
        
    # %%
    
    merged_data = pd.merge(patch_names_df, df_name, how='outer', on='Patient ID')
    merged_data_label = pd.merge(merged_data, df_labels, how='outer', on='Patient ID')
    merged_data_label = merged_data_label.dropna(subset=["Patch_name"])
    df_all_stains.append(merged_data_label)
    
    #%%
    
merged_all_stains_df = pd.concat(df_all_stains,  keys='Patient ID')
patient_labels = set(zip(merged_all_stains_df['Patient ID'], merged_all_stains_df['Pathotype']))
patient_labels_dict = {k: v for k, v in patient_labels if not math.isnan(v)}

# %%

# patient_labels_df = pd.DataFrame(patient_labels_dict, index=["Pathotype"]).transpose()
# patient_labels_df.to_csv( r"C:/Users/Amaya/Documents/PhD/Data/patient_labels.csv")

#%%

merged_data.to_csv(r"C:/Users/Amaya/Documents/PhD/Data/" + stain + "/df_all_" + stain + "_patches_labels.csv", sep=',', index=False)
merged_all_stains_df.to_csv(r"C:/Users/Amaya/Documents/PhD/Data/df_all_stains_patches_labels.csv", sep=',', index=False)

# %%

