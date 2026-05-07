import radiomics
import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage.segmentation import slic
import numpy as np
import pickle
from glob import glob
import time
from pathlib import Path
import os
from tqdm import tqdm

radiomics.setVerbosity(40)

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



def FindBounds(struct,axial_padding=25,vertical_padding=10):
  ### This function finds the bounds of a mask with padding if desired
  r = np.any(struct, axis=(1, 2))
  c = np.any(struct, axis=(0, 2))
  z = np.any(struct, axis=(0, 1))

  rmin, rmax = np.where(r)[0][[0, -1]]
  cmin, cmax = np.where(c)[0][[0, -1]]
  zmin, zmax = np.where(z)[0][[0, -1]]



  if rmin-vertical_padding<=0:
    rmin=0
  else:
    rmin=rmin-vertical_padding
  if cmin-axial_padding<=0:
    cmin=0
  else:
    cmin=cmin-axial_padding
  if zmin-axial_padding<=0:
    zmin=0
  else:
    zmin=zmin-axial_padding
  if rmax+vertical_padding>=struct.shape[0]:
    rmax=struct.shape[0]-1
  else:
    rmax=rmax+vertical_padding
  if cmax+axial_padding>=struct.shape[1]:
    cmax=struct.shape[1]-1
  else:
    cmax=cmax+axial_padding
  if zmax+axial_padding>=struct.shape[2]:
    zmax=struct.shape[2]-1
  else:
    zmax=zmax+axial_padding   

  return rmin,rmax,cmin,cmax,zmin,zmax

if __name__=='__main__':

  log={} #Log errors
  n_segments=100
  compactness=.0001
  mm_bound=50
  # The below assummes directory contains CTs with naming {patient}__CT.nii.gz and associated GTV masks {patient}__mask.nii.gz
  imagepathlist=glob(os.path.join('/dir/containing/CTs','*CT.nii.gz')) 

  OutDir='/dir/for/output' #Used in GraphGeneration.py
  if not os.path.exists(OutDir):
    os.mkdir(OutDir)
  for image_path in tqdm(imagepathlist):
    mask_path=image_path.split('__CT')[0]+'__mask.nii.gz'
    patient=Path(image_path).name.split('__')[0] #See naming convention in Line 97

    dataset_name=image_path.split('/')[-2] #Assumes images are split into folders by dataset
    curr_out_path=os.path.join(OutDir,dataset_name)
    if not os.path.exists(curr_out_path):
      os.mkdir(curr_out_path)
    patient_outpath=os.path.join(curr_out_path,patient+'.pkl')

    try:
      image=sitk.ReadImage(image_path)
      mask=sitk.ReadImage(mask_path)
      mask_array=sitk.GetArrayFromImage(mask)
      array_3D=sitk.GetArrayFromImage(image)
      axial_padding=np.round(mm_bound/mask.GetSpacing()[0]).astype(int)
      vertical_padding=np.round(mm_bound/mask.GetSpacing()[-1]).astype(int)
      a,b,c,d,e,f=FindBounds(mask_array,axial_padding=axial_padding,vertical_padding=vertical_padding)
      boundingbox=np.zeros(mask_array.shape)
      boundingbox[a:b,c:d,e:f]=1
      boundingbox[mask_array>0]=0 #We want supervoxels not including the tumor region. We use the GTV as its own "supervoxel"

      labels_img_3D = slic(array_3D,mask=boundingbox,
                                        compactness=compactness, 
                                        n_segments=n_segments, 
                                        start_label=1,
                                        channel_axis=None,
                                        slic_zero=True)
   
      
      labels_img_3D[mask_array>0]=labels_img_3D.max()+1 ##GTV is now the node with largest label
      
      label_value_set=set(labels_img_3D.flatten())

      labels_img_3D=labels_img_3D.astype(np.float32)
      labels_img=sitk.GetImageFromArray(labels_img_3D)
      labels_img.CopyInformation(image)
      pkl_dict={}
      pkl_dict['label_img']=labels_img
      pkl_dict['features']={}

      #The below begins extracting radiomic features
      for label_val in label_value_set:
        if label_val==0:
          continue
        setting={}
        setting['setting']={}
        setting['setting']['label']=int(label_val)
        ext=radiomics.featureextractor.RadiomicsFeatureExtractor(setting)


        feats=ext.execute(image,labels_img)

        desired_keys=[]
        for key in feats.keys():
          if 'diagnostics' in key:
            continue
          if 'shape' in key:
            continue
          desired_keys.append(key)
        new_feat_dict={}
        for key in desired_keys:
          new_feat_dict[key]=feats[key]
        
        pkl_dict['features'][label_val]=new_feat_dict
      save_pkl(pkl_dict,patient_outpath) #Saves all supervoxel radiomic features into a dict on a per-patient basis for GraphGeneration.py
    except Exception as e:

      log[patient]=str(e)
      print(str(e))
  #Log Errors
  logdf=pd.DataFrame.from_dict(log,orient='index')
  logdf.to_csv(os.path.join(OutDir,exp_name+'_log.csv'))