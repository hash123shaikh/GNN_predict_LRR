import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
def save_pkl(pkl_dict,outPath):

  filehandler = open(outPath, 'wb')
  pickle.dump(pkl_dict, filehandler)
  filehandler.close()
  return

def load_pkl(pkl_path):
  MasterDict={}

  filehandler = open(pkl_path, 'rb')
  result_dict=pickle.load(filehandler)
  filehandler.close()
  return result_dict
def getFeatNamesFromPKL(pkl_folder):
  feats=[]
  for patient in os.listdir(pkl_folder):
    for images in os.listdir(os.path.join(pkl_folder,patient)):
      feat=images.split('__')[-1].split('.')[0]
      feats.append(feat)

  total_feats=list(set(feats))
  return total_feats

"""
This script creates a scaler for use in GraphGeneration.py. Used to normalize radiomic feature expression. 
"""
sheetpath="/path/to/radiomic/features" #This should be a csv with radiomic features for supervoxels used for normalization (GTV supervoxels in the paper)
outpath="/dir/to/save/scaler"
sheet=pd.read_csv(sheetpath)
desiredCols=[x for x in sheet.columns if x not in ['Patient','LR','DM','Unnamed: 0']] #Filters df to include only radiomoic feature columns
rad_data=sheet.loc[:,desiredCols]
print(desiredCols) #Sanity check radiomic features

scaler=StandardScaler() #z-score normalization. 
scaler.fit(rad_data)
scaled=scaler.transform(rad_data)
new_data=pd.DataFrame(scaled,columns=desiredCols)
sheet[desiredCols]=new_data[desiredCols]
out_d={'scaler':scaler}
save_pkl(out_d,outpath) #Save the scaler for use in GraphGeneration.py

