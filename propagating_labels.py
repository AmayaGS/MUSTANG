# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:15:45 2022

@author: AmayaGS
"""

import shutil
import os
import pandas as pd
import numpy as np

# %%
    
main_dir = r"C:/Users/Amaya/Documents/PhD/NECCESITY/Slides/Janssen/tiles"
csv_file = r"C:\Users\Amaya\Documents\PhD\NECCESITY\Slides\Janssen\Sjogren_biopsy_designTable_Jansen metadata shared.csv"
#print(main_dir, csv_file, stain)

folder_list = [f.path for f in os.scandir(main_dir)]
df = pd.read_csv(csv_file, encoding='windows-1252')

# %%

patch_names = []
for location in folder_list:
    patch_list = os.listdir(location)
    for patch in patch_list:
        patch_names.append([patch[0:-4], patch.split(" ")[0], os.path.join(location, patch)])

# %%

patch_names_df = pd.DataFrame(patch_names)
patch_names_df.columns= ['Patch_name', 'Patient ID', 'Location']
    
# %%

merged_data = pd.merge(patch_names_df, df, how='outer', on='Patient ID')
merged_data = merged_data.dropna(subset=["Patch_name"])

    #%%

merged_data.to_csv(r'C:\Users\Amaya\Documents\PhD\NECCESITY\Slides\Janssen\janssen_patch_labels.csv', sep=',', index=False)


# %%

#qmul = r"C:\Users\Amaya\Documents\PhD\NECCESITY\Slides\QMUL_patch_labels.csv"
janssen = r"C:\Users\Amaya\Documents\PhD\NECCESITY\Slides\Janssen\janssen_patch_labels.csv"

#df_qmul = pd.read_csv(qmul)
df_janssen = pd.read_csv(janssen)

# %%

patch_labels = r"C:/Users/Amaya/Documents/PhD/NECCESITY/all_slides_patch_labels.csv"
df = pd.read_csv(patch_labels, header=0)
df["Group label"] = df["Group label"].astype('Int64')
df["Patient ID"] = df["Patient ID"].astype('str')
df["Binary disease"] = df["Binary disease"].astype('Int64')

# df_qmul = pd.read_csv(qmul_csv_file, header=0)
# df_qmul["Group ID"] = df_qmul["Group ID"].astype('Int64')

# df_birm = pd.read_csv(birm_csv_file, header=0)
# df_birm["Group ID"] = df_birm["Group ID"].astype('Int64')

# %%

df = df[df["CENTER"] == "QMUL"]

# %%

jan_sub = df_janssen[["Patch_name", "Patient ID", "Location", "Binary disease"]]
qmul_sub = df[["Patch_name", "Patient ID", "Location", "Binary disease"]]

# %%

result = pd.concat([qmul_sub, jan_sub])

# %%

result.to_csv(r'C:\Users\Amaya\Documents\PhD\NECCESITY\Slides\qj_patch_labels.csv', sep=',', index=False)












# patch_names = []
# for location in folder_list:
#     patch_list = os.listdir(location)
#     for patch in patch_list:
#         if patch.startswith("20-OA"):
#             patch_names.append([patch[0:-4], location.split("\\")[-1], location.split("\\")[1].split("-")[-1].split("_")[0], os.path.join(location, patch)])
#         if patch.startswith("50."):
#             patch_names.append([patch[0:-4], location.split("\\")[-1], location.split("\\")[-1].split(".")[0] + "." + location.split("\\")[-1].split(".")[1], os.path.join(location, patch)])
#         if patch.startswith("PATHSSAI"):
#             patch_names.append([patch[0:-4], location.split("\\")[-1], location.split("\\")[-1].split(" ")[-1].split("-")[0] + "-" + location.split("\\")[-1].split(" ")[-1].split("-")[1], os.path.join(location, patch)])
            