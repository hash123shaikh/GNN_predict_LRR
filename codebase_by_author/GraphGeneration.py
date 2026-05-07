import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import pickle
from skimage.measure import regionprops
from scipy.spatial.distance import pdist,squareform
from glob import glob
import time
import os
from collections import defaultdict
import argparse
import torch
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

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


def make_rad_graph_gtv_readout(feature_dict, input_regions, desired_feats, connectivity_type='nearest', 
                   connectivity_thresh=None, connectivity=3,dimensions=3):

  regions=regionprops(input_regions)
  current_rad_graph=nx.Graph()

  expressions=[]
  label_value_set=np.array(list(set(input_regions.flatten())))
  label_value_set.sort()
  label_value_set=np.delete(label_value_set,0)
  #The below for loop iterates through our radiomic features for supervoxels and pre-processes them a little further (clipping)
  for label_index,label in enumerate(label_value_set):
    desired_region=regions[label_index]
    if desired_region.label!=label:
      raise(Exception("Node label and region label do not match"))
    exps=[]
    if type(desired_feats)==str: #This handles the case that we all radiomic feature for our nodes
      if desired_feats=='all':
        for feat_key in feature_dict[label].keys():
          feat_expression_val=feature_dict[label][feat_key]
          if feat_expression_val>5: #assume we performed z-score normalization, we are clipping outlier features
            feat_expression_val=5.0
          elif feat_expression_val<-5:
            feat_expression_val=-5.0
          exps.append(np.array(feat_expression_val))
    else: #In general we are passed a list of desired radiomic features
      for feat_key in desired_feats:
        feat_expression_val=feature_dict[label][feat_key]
        if feat_expression_val>5: #assume we performed z-score normalization, we are clipping outlier features
          feat_expression_val=5.0
        elif feat_expression_val<-5:
          feat_expression_val=-5.0
        exps.append(np.array(feat_expression_val))


    expressions.append(np.array(exps))

  expressions=np.array(expressions).astype(np.float64)

  #The below for loop builds our graph
  node_index=0
  for label_index,label in enumerate(label_value_set):
    desired_region=regions[label_index]
    if desired_region.label!=label:
      raise(Exception("Node label and region label do not match"))
    if dimensions==3:
      x,y,z=np.round(desired_region.centroid).astype(int)
    else:
      x,y=np.round(desired_region.centroid).astype(int)
    if dimensions==3:
      current_rad_graph.add_node(node_index,center=(x,y,z),orig_label=label,
                                mean=expressions[node_index],
                                )

      node_index+=1
    else:
      current_rad_graph.add_node(node_index,center=(x,y),orig_label=label,
                                mean=expressions[node_index],
                                )
      
      node_index+=1

  #Calculate Euclidean distance between radiomic feature supervoxels
  distances=pdist(expressions)
  square_dist=squareform(distances)

  #Can use different sparsification strategies. In the paper we use nearest. 
  if connectivity_thresh==None:
    connectivity_thresh=np.nanmedian(square_dist.ravel())
  i=square_dist.shape[0]-1
  current_dists=square_dist[i,:]
  if connectivity_type=='nearest':
    sorted_inds=np.argsort(current_dists)
    
    num_connects=0
    
    for j in sorted_inds:
      if num_connects>=connectivity: #Sets the number of nodes to retain
        break
      if j==i: #Do not want nodes with edges to themselves
        continue
      dist=square_dist[i,j]
      current_rad_graph.add_edge(i,j,
                                mean_dist=dist,
                                pts=(current_rad_graph.nodes('center')[i],
                                      current_rad_graph.nodes('center')[j])
                                )
      num_connects+=1
  else:
    for j in range(square_dist.shape[1]):
      if j==i:
        continue
      dist=square_dist[i,j]
      if dist<connectivity_thresh:
        current_rad_graph.add_edge(i,j,
                                mean_dist=dist,
                                pts=(current_rad_graph.nodes('center')[i],
                                      current_rad_graph.nodes('center')[j])
                                )
  current_rad_graph.remove_nodes_from(list(nx.isolates(current_rad_graph))) #Unconnected nodes are removed from the graph
  mapping={}

  for new_index,node in enumerate(current_rad_graph.nodes()): #Relabel nodes to be sequential after removing isolated nodes, but keep mapping to original labels for post-hoc viz
    mapping[node]=new_index
  relabeled_graph=nx.relabel_nodes(current_rad_graph,mapping)
    
  return relabeled_graph


if __name__=='__main__':

  parser=argparse.ArgumentParser()
  parser.add_argument('--neighbors', type=int, default = 10)
  parser.add_argument('--outcome', type=str, default = 'lr')

  args=parser.parse_args()
  
  neighbors=args.neighbors
  outcome_type=args.outcome
  failureKey=pd.read_csv("/path/to/outcomes") #Contains binarized DM and LR labels for all patients based on 2-year followups
  scaler=load_pkl("/path/to/scaler")['scaler'] #from ScalerGeneration.py. This is used to z-score normalize features

  #The below shoudl contain radiomci features generated in RadiomicSupervoxels.py
  RADCURE_dir='/path/to/radcure'
  HN1_dir='/path/to/Head_Neck_Radiomics_1/'
  HNSCC_dir='/path/to/HNSCC/'
  HNPET_dir='/path/to/Head-Neck-PET-CT'

  if outcome_type=='lr':
    desired_feats=load_pkl("/path/to/lr_features/")['features'] #A list of features based upon radiomic baseline experiments. 
  elif outcome_type=='dm':
    desired_feats=load_pkl("/path/to/dm_features")['features']

  out_dir='path/to/store/graphs'

  
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  
  start=time.time()
  # The below just makes subfolders for different combinations of outcome and neighbors (aka how many supervoxels to retain)
  ExpName=outcome_type+'__'+str(neighbors)
  if not os.path.exists(os.path.join(out_dir,ExpType)):
    os.mkdir(os.path.join(out_dir,ExpType))
  curr_out_dir=os.path.join(out_dir,ExpType,ExpName)
  if not os.path.exists(curr_out_dir):
    os.mkdir(curr_out_dir)

  statuses={} #Store exceptions and errors

  iter_dirs=(RADCURE_dir,HN1_dir,HNSCC_dir,HNPET_dir,)
  out_names=('RADCURE','HN1','HNSS','HeadNeckPETCT',)

  for rad_dir,split_name in zip(iter_dirs,out_names):
    patient_list_to_iterate=os.listdir(rad_dir) #remember, rad_dir contains the output of RadiomicSupervoxels. Every file should be of format {patient_name}.pkl
    graphs_for_pkl=[]
    out_patients=[]
    for count_index,patient in enumerate(patient_list_to_iterate):
      p=patient.split('.')[0]
      try:
        d=load_pkl(os.path.join(rad_dir,patient))
        for node in d['features'].keys():
          df_rep=pd.DataFrame.from_dict(d['features'][node],orient='index').T
          scaled_node=pd.DataFrame(scaler.transform(df_rep),columns=df_rep.columns)
          d['features'][node]=scaled_node.to_dict(orient='index')[0]
          
        feature_dict=d['features']
        label_img=d['label_img']
        input_regions=sitk.GetArrayFromImage(label_img).astype(int)   
        
        graph=make_rad_graph_gtv_readout(feature_dict,
          input_regions,desired_feats,
          dimensions=3,
          connectivity_type='nearest',
          connectivity=neighbors)

        geograph=from_networkx(graph)
        geograph.LR=failureKey.loc[failureKey['ID']==p,'binary locoregional failure'].values[0] #Both this and the next line are binarized based on 2 year follow-up
        geograph.DM=failureKey.loc[failureKey['ID']==p,'binary distant failure'].values[0]
        geograph.x=geograph.mean.double()
        edge_weights=torch.tensor(nx.adjacency_matrix(graph,dtype=np.float32, weight='mean_dist').toarray())
        geograph.edge_attr=edge_weights[edge_weights!=0][:,None]
        graphs_for_pkl.append(geograph)
        out_patients.append(p)
      except Exception as e:
        print(p)
        print(str(e))
        statuses[p]=str(e)
      
      print(('Finished patient %i out of %i in %.2f hours') % (count_index+1,len(patient_list_to_iterate),(time.time()-start)/3600))
      print(('Predicted Finish is in %.2f hours') % (((time.time()-start)/(count_index+1))*(len(patient_list_to_iterate)-count_index-1)/3600))
    
    status_sheet=pd.DataFrame.from_dict(statuses,orient='index')
    status_sheet.to_csv(os.path.join(curr_out_dir,split_name+' graph_creation_log.csv'))
    save_pkl({'graphs':graphs_for_pkl,'patients':out_patients},os.path.join(curr_out_dir,split_name+'.pkl'))














