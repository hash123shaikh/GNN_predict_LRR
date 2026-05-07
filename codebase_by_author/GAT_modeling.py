import numpy as np
import pandas as pd
import pickle
from glob import glob
import time
import os
from datetime import datetime
import numpy.random as random
from collections import defaultdict
import wandb
import argparse
from sklearn.metrics import confusion_matrix,roc_auc_score
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from torch.nn import Linear
import torch.nn.functional as F
torch.manual_seed(1)
np.random.seed(1)

class ReadoutGAT_clinical(torch.nn.Module):
  def __init__(self,layers,num_features,hidden_dim=32,num_heads=1,dropout=0):
    super(ReadoutGAT_clinical,self).__init__()
    torch.manual_seed(1)

    self.gat_layers = torch.nn.ModuleList()
    if layers==1:
      # First GAT layer
      self.gat_layers.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
      self.lin = Linear(hidden_dim*num_heads+25, 2)
    elif layers==2:
      self.gat_layers.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
      self.gat_layers.append(LayerNorm(hidden_dim * num_heads))
      self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)) 

      self.lin = Linear(hidden_dim+25, 2)
    else:
      self.gat_layers.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
      self.gat_layers.append(LayerNorm(hidden_dim * num_heads))
      # Middle GAT layers
      for _ in range(layers - 2):
        
        self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        self.gat_layers.append(LayerNorm(hidden_dim * num_heads))
      # Final GAT layer
      self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)) 
      self.lin = Linear(hidden_dim+25, 2)

    
  def obtain_token_rep(self,x,batch):
    token_rep=torch.empty(1,x.shape[1])
    for val_index in set(batch.numpy()):
      token_rep=torch.cat((token_rep,x[batch==val_index][-1,:][None,:]),dim=0)
    token_rep=token_rep[1::,:]
    return token_rep
  def forward(self,x,edge_index,batch,clinical_data):
    for gat_layer in self.gat_layers:
      if isinstance(gat_layer,LayerNorm):
        x = gat_layer(x, batch)
        x = torch.relu(x)
      else:
        x = gat_layer(x, edge_index)
        
    x=self.obtain_token_rep(x,batch)
    x = F.dropout(x, p=0.5, training=self.training)
    x = torch.cat((x,torch.tensor(clinical_data)),dim=1)
    x = self.lin(x)
    return x


class ReadoutGAT(torch.nn.Module):
  def __init__(self,layers,num_features,hidden_dim=32,num_heads=1,dropout=0):
    super(ReadoutGAT,self).__init__()
    torch.manual_seed(1)

    self.gat_layers = torch.nn.ModuleList()
    if layers==1:
      # First GAT layer
      self.gat_layers.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
      self.lin = Linear(hidden_dim*num_heads, 2)
    elif layers==2:
      self.gat_layers.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
      self.gat_layers.append(LayerNorm(hidden_dim * num_heads))
      self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)) 

      self.lin = Linear(hidden_dim, 2)
    else:
      self.gat_layers.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
      self.gat_layers.append(LayerNorm(hidden_dim * num_heads))
      # Middle GAT layers
      for _ in range(layers - 2):
        
        self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        self.gat_layers.append(LayerNorm(hidden_dim * num_heads))
      # Final GAT layer
      self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)) 
      self.lin = Linear(hidden_dim, 2)

    
  def obtain_token_rep(self,x,batch):
    token_rep=torch.empty(1,x.shape[1])
    for val_index in set(batch.numpy()):
      token_rep=torch.cat((token_rep,x[batch==val_index][-1,:][None,:]),dim=0)
    token_rep=token_rep[1::,:]
    return token_rep
  def forward(self,x,edge_index,batch):
    for gat_layer in self.gat_layers:
      if isinstance(gat_layer,LayerNorm):
        x = gat_layer(x, batch)
        x = torch.relu(x)
      else:
        x = gat_layer(x, edge_index)
        
    x=self.obtain_token_rep(x,batch)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin(x)
    return x



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
def are_items_increasing(lst,num_last=5):
    last_items = lst[-num_last:]
    return last_items == sorted(last_items) 
if __name__=='__main__':
  
  parser=argparse.ArgumentParser()
  parser.add_argument('--layers', type=int, default = 2)   
  parser.add_argument('--heads', type=int, default = 1)    
  parser.add_argument('--epochs', type=int, default = 1000)   
  parser.add_argument('--hidden_features', type=int, default = 16)   
  parser.add_argument('--weight_decay', type=float,default=0.00001)
  parser.add_argument('--lr', type=float,default=0.001)
  parser.add_argument('--dropout', type=float,default=0.5)  
  parser.add_argument('--outcome',type=str,default='lr')
  parser.add_argument('--sampling',type=str,default='mid')
  parser.add_argument('--batch_size',type=int,default=128)
  parser.add_argument('--project',type=str,default='WandB_Project_Name')
  parser.add_argument('--out_path',type=str,default='/out/dir')  
  parser.add_argument('--neighbors',type=int,default=10)
  parser.add_argument('--early_stopping',type=int,default=1) #Whether to use early stopping of training absed on validation
  parser.add_argument('--add_clinical',type=int,default=0)  
   
  args=parser.parse_args()
  train_names=['RADCURE_train']
  val_names=['RADCURE_val']
  test_names=['CHUM','HMR','HGJ','CHUS','HNSCC','RADCURE_test','Head_Neck_Radiomics_1'] 
  outcome_type=args.outcome
  out_dir=args.out_path
  add_clinical=args.add_clinical
  feat_type_dict={

    'lr':4,
    'dm':6

  }

  feat_type=feat_type_dict[outcome_type]
  now=datetime.now().strftime('%b_%d_%Y__%H_%M_%S')
  ExpName=now
  print('Starting at '+now)

  day_now=datetime.now().strftime('%b_%d_%Y')

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  out_dir=os.path.join(out_dir,outcome_type)
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  out_dir=os.path.join(out_dir,day_now)
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  


  early_stopping=args.early_stopping
  lr=args.lr
  weight_decay=args.weight_decay
  layers=args.layers
  heads=args.heads
  batch_size=args.batch_size
  sampling=args.sampling
  project_name=args.project
  epochs=args.epochs
  dropout=args.dropout
  neighbors=args.neighbors
  hidden_features=args.hidden_features

  if feat_type=='all':
    num_features=93
  else:
    num_features = feat_type
  wandb.init(project=project_name, entity="jaebae")
  RADCURE_split_d=load_pkl("/path/to/split_dictionary") #Contains a list of patients in train, validation, and testing for RADCURE
  clinical_data=pd.read_csv("/clinical/features/csv") #Contains rows corresponding to each patient and columns corresponding to clinical features
  clin_cols=[x for x in clinical_data.columns if x not in ['ID','Unnamed: 0','Patient']] #Filter out non-feature columns

  graph_dict={
    
    """
    Should contain paths to graphs generated by GraphGeneration.py
    You might have different directories for different outcomes because
    radiomic features used in GraphGeneration.py depend on the choice of radiomic features

    """


  }




  graph_gen_dir=graph_dict["InsertKeyHere"] #This is dependent on how you structure graph_dict. Ultimately, you want a directory from GraphGeneration.py here
  print(graph_gen_dir)
  config_dict={
        'layers':layers,
        'heads':heads,
        'hidden_features':hidden_features,
        'lr':lr,
        'sampling_type':sampling,
        'graph_dir':graph_gen_dir,
        'outcome':outcome_type,
        'batch_size':batch_size,
        'time':now
        }

  
  start=time.time()


  curr_out_dir=os.path.join(out_dir,ExpName)
  if not os.path.exists(curr_out_dir):
    os.mkdir(curr_out_dir)

  pkl_dicts={}
  print('Loading Graphs')
  for pkl_file_name in os.listdir(graph_gen_dir): #See GraphGeneration.py where pkl_file_name should be a dataset name
    if pkl_file_name.split('.')[-1]!='pkl':
      continue
    else:
      pkl_dicts[pkl_file_name]=load_pkl(os.path.join(graph_gen_dir,pkl_file_name)) #We have now loaded dictionaries containing all graphs for a dataset
 



  train_dataset=[]
  val_dataset=[]

  test_dataset_mdacc=[]
  test_dataset_hn1=[]
  test_dataset_radcure=[]
  test_dataset_hnpet=[]
  test_dataset_all=[]
  
  train_patient_list=[]
  val_patient_list=[]
  
  test_patient_list_mdacc=[]
  test_patient_list_hn1=[]
  test_patient_list_radcure=[]
  test_patient_list_hnpet=[]
  test_patient_list_all=[]
  print('Processing Data Splits')
  for dataset_name_d,graph_pkl_c in pkl_dicts.items():
    dataset_name=dataset_name_d.split('.')[0]
    current_graph_list=graph_pkl_c['graphs']
    current_patient_list=graph_pkl_c['patients']

    if dataset_name=='RADCURE':
      for patient_to_iterate,graph_to_iterate in zip(current_patient_list,current_graph_list):

        if add_clinical:
          clin_vector=clinical_data.loc[clinical_data['Patient']==patient_to_iterate,clin_cols].values[0]
          graph_to_iterate.clin_data=clin_vector
       

        if patient_to_iterate in RADCURE_split_d['train']:
          train_dataset.append(graph_to_iterate)
          train_patient_list.append(patient_to_iterate)
        elif patient_to_iterate in RADCURE_split_d['test']:
          test_dataset_radcure.append(graph_to_iterate)
          test_patient_list_radcure.append(patient_to_iterate)
          test_dataset_all.append(graph_to_iterate)
          test_patient_list_all.append(patient_to_iterate)          
        elif patient_to_iterate in RADCURE_split_d['val']:
          val_dataset.append(graph_to_iterate)
          val_patient_list.append(patient_to_iterate)    
    elif dataset_name=='HeadNeckPETCT':

      for patient_to_iterate,graph_to_iterate in zip(current_patient_list,current_graph_list):
        
        if add_clinical:
          clin_vector=clinical_data.loc[clinical_data['Patient']==patient_to_iterate,clin_cols].values[0]
          graph_to_iterate.clin_data=clin_vector        
        
        current_sub=patient_to_iterate.split('-')[1] #We could implement logic here to make other training datasets if the HeadNeckPETCT subsites were split between different train/val/test splits
        
        test_dataset_hnpet.append(graph_to_iterate)
        test_patient_list_hnpet.append(patient_to_iterate)
        test_dataset_all.append(graph_to_iterate)
        test_patient_list_all.append(patient_to_iterate)          
        else:
          continue
    else:
      for patient_to_iterate,graph_to_iterate in zip(current_patient_list,current_graph_list):

        if add_clinical:
          
          clin_vector=clinical_data.loc[clinical_data['Patient']==patient_to_iterate,clin_cols].values[0]
          graph_to_iterate.clin_data=clin_vector        
        
        
        if dataset_name in train_names:
          train_dataset.append(graph_to_iterate)
          train_patient_list.append(patient_to_iterate)
        elif dataset_name in val_names:
          val_dataset.append(graph_to_iterate)
          val_patient_list.append(patient_to_iterate)
        elif dataset_name in test_names:
          test_dataset_all.append(graph_to_iterate)
          test_patient_list_all.append(patient_to_iterate)
          if dataset_name=='Head_Neck_Radiomics_1':
            test_dataset_hn1.append(graph_to_iterate)
            test_patient_list_hn1.append(patient_to_iterate)
          else:
            test_dataset_mdacc.append(graph_to_iterate)
            test_patient_list_mdacc.append(patient_to_iterate)




  print('-----------------------------------')
  print('Length of train:',len(train_dataset), 'Length of val:',len(val_dataset),'Length of test:',len(test_dataset_all))
  
  #### PAY ATTENTION TO TESTLOADERS
  datasets=[
    train_dataset,
    val_dataset,
    test_dataset_hnpet,
    test_dataset_mdacc,
    test_dataset_hn1,
    test_dataset_radcure,
    test_dataset_all
  ]
  
  patient_sets=[
    train_patient_list,
    val_patient_list,
    test_patient_list_petct,
    test_patient_list_mdacc,hnpet
    test_patient_list_hn1,
    test_patient_list_radcure,
    test_patient_list_all
  ]
  

  if outcome_type=='dm':
    for dataset_split,patient_split in zip(datasets,patient_sets):
      for curr_graph,curr_patient in zip(dataset_split,patient_split):
        curr_graph.y=curr_graph.DM
  if outcome_type=='lr':
    for dataset_split,patient_split in zip(datasets,patient_sets):
      for curr_graph,curr_patient in zip(dataset_split,patient_split):
        curr_graph.y=curr_graph.LR

  
  #Balancing the training dataset based on sampling strategy. 
  positive_samples=0
  negative_samples=0
  for ind_graph in train_dataset:
    if ind_graph.y:
      positive_samples+=1
    else:
      negative_samples+=1


  if positive_samples>negative_samples:
    pos_is_majority=1
    factor=positive_samples/negative_samples
    majority_size=positive_samples
    minority_size=negative_samples
  else:
    pos_is_majority=0
    majority_size=negative_samples
    factor=negative_samples/positive_samples
    minority_size=positive_samples
  new_train=[]
  if sampling=='over':
    running_minority=0
    print('oversampling')
    for ind_graph in train_dataset:
      
      if ind_graph.y==(pos_is_majority==0):

        if np.random.randint(10)>7:
          iter_size=np.ceil(factor)
        else:
          iter_size=np.ceil(factor)+1
        for iijj in range(iter_size.astype(int)):

          if running_minority==majority_size:
            
            break
          new_train.append(ind_graph)
          running_minority+=1
      else:
        new_train.append(ind_graph)
  elif sampling=='under':
    running_majority=0

    print('undersampling')
    for ind_graph in train_dataset:
      
      if ind_graph.y==(pos_is_majority):
        if running_majority==minority_size:
          continue
        iter_size=(1/factor)+((1/factor))
        if random.rand()<iter_size:
          new_train.append(ind_graph)
          running_majority+=1      
      else:
        new_train.append(ind_graph)
  
  elif sampling=='mid':
    running_minority=0
    running_majority=0
    print('average sampling')
    average_size=np.round(np.mean([minority_size,majority_size]))
    factor=average_size/minority_size
    for ind_graph in train_dataset:
      
      if ind_graph.y==(pos_is_majority):
        if running_majority==average_size:
          continue
        iter_size=(1/factor)+((2/factor))
        if random.rand()<iter_size:
          new_train.append(ind_graph)
          running_majority+=1      
      else:
        if np.random.randint(10)>7:
          iter_size=np.ceil(factor)
        else:
          iter_size=np.ceil(factor)+1
        for iijj in range(iter_size.astype(int)):

          if running_minority==average_size:
            
            break
          new_train.append(ind_graph)
          running_minority+=1 
  train_dataset=new_train

  train_ratio_list=[x.y for x in train_dataset]
  print('Train Ratio:',np.sum(train_ratio_list)/len(train_ratio_list),'or',np.sum(train_ratio_list),'out of',len(train_ratio_list),)

  if add_clinical:
    model=ReadoutGAT_clinical(layers,num_features,hidden_dim=hidden_features,num_heads=heads,dropout=dropout)
  else:  
    model=ReadoutGAT(layers,num_features,hidden_dim=hidden_features,num_heads=heads,dropout=dropout)

#######################################################################
##Dataloaders for calculating metrics and performing eval
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader_hnpet=DataLoader(test_dataset_hnpet,batch_size=batch_size,shuffle=False)
  test_loader_mdacc=DataLoader(test_dataset_mdacc,batch_size=batch_size,shuffle=False)
  test_loader_hn1=DataLoader(test_dataset_hn1,batch_size=batch_size,shuffle=False)
  test_loader_radcure=DataLoader(test_dataset_radcure,batch_size=batch_size,shuffle=False)
  test_loader_all=DataLoader(test_dataset_all,batch_size=batch_size,shuffle=False)

  metric_loader_dict={
    'validation_radcure':val_loader,
    'headneckpetct':test_loader_hnpet,
    'head-neck-radiomics-1':test_loader_hn1,
    'mdacc':test_loader_mdacc,
    'radcure_test':test_loader_radcure,
    'all_test':test_loader_all
  }
#######################################################################
  model.double()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  criterion = torch.nn.CrossEntropyLoss()


  def train():
    model.train()
    optimizer.zero_grad()
    for data in train_loader:  
      if add_clinical:
        out = model(data.x, data.edge_index, data.batch,data.clin_data)
      else:
        out = model(data.x, data.edge_index, data.batch,)
      loss = criterion(out, torch.tensor(data.y))
      loss.backward()  
      optimizer.step() 
      optimizer.zero_grad()
      

  def test(loader):
    model.eval()
    probs=[]
    gts=[]
    bin_preds=[]
    correct = 0
    running_loss=0
    for data in loader:  
      if add_clinical:
        out = model(data.x, data.edge_index, data.batch,data.clin_data)
      else:
        out = model(data.x, data.edge_index, data.batch,)
      pred = out.argmax(dim=1)  
      loss=criterion(out, torch.tensor(data.y)).item()
      running_loss+=loss
      correct += int((pred == torch.tensor(data.y)).sum())  
      
      probs=np.concatenate((probs,out[:,1].detach().cpu().numpy()))
      
      gts=np.concatenate((gts,np.array(data.y).flatten()))
      bin_preds=np.concatenate((bin_preds,pred.detach().cpu().numpy()))
    

    

    auc_metric=roc_auc_score(gts,probs)
    conf=confusion_matrix(np.ravel(gts),bin_preds)
    specificity=(conf[0,0]/(conf[0,0]+conf[0,1]))
    sensitivity=(conf[1,1]/(conf[1,0]+conf[1,1]))
    tloss=running_loss/len(loader)
    return auc_metric,sensitivity,specificity,tloss  

  lowest_loss=90000
  best_epoch=1212
  running_losses=[] 
  for epoch in range(epochs):
    train()
    train_auc,train_sens,train_spec,train_loss = test(train_loader)
    val_auc,val_sens,val_spec,val_loss = test(val_loader)
    wandb.log({'train_sen':train_sens},step=epoch)
    wandb.log({'train_auc':train_auc},step=epoch)
    wandb.log({'train_spe':train_spec},step=epoch)
    wandb.log({'train_loss':train_loss},step=epoch)
    wandb.log({'val_sen':val_sens},step=epoch)
    wandb.log({'val_auc':val_auc},step=epoch)
    wandb.log({'val_spe':val_spec},step=epoch)
    wandb.log({'val_loss':val_loss},step=epoch)
    running_losses.append(val_loss)
    if epoch>10:
      if val_loss<=lowest_loss:
        lowest_loss=val_loss
        best_epoch=epoch
        ckpt_model=model.state_dict()
        best_val_auc=val_auc
        best_val_sens=val_sens
        best_val_spec=val_spec
    if epoch%5==0:
      print(f'Epoch: {epoch:03d}')
      print(f'Train AUC: {train_auc:.4f}, Train Sens: {train_sens:.4f}, Train Spec: {train_spec:.4f}') 
      print(f'Val AUC: {val_auc:.4f}, Val Sens: {val_sens:.4f}, Val Spec: {val_spec:.4f}')
    if early_stopping:
      if epoch>20:
        if are_items_increasing(running_losses,10):
          break

  model.load_state_dict(ckpt_model) 
  torch.save(ckpt_model,os.path.join(curr_out_dir,ExpName+'.pth'))

  save_pkl(config_dict,os.path.join(curr_out_dir,ExpName+'__parameters.pkl'))
  wandb.log({'lowest_val_loss':lowest_loss},step=epoch)
  wandb.log({'best_epoch':best_epoch},step=epoch)
  
  out_frame=pd.DataFrame()
  for curr_loader_name, curr_loader in metric_loader_dict.items():
    curr_row=pd.Series(dtype=object,name=curr_loader_name)
    test_auc,test_sens,test_spec,test_loss = test(curr_loader)
    curr_row['dataset']=curr_loader_name
    curr_row['auc']=test_auc
    curr_row['sensitivity']=test_sens
    curr_row['specificity']=test_spec

    out_frame=pd.concat((out_frame,curr_row.to_frame().T),axis='rows')
    wandb.log({curr_loader_name+'_test_sen':test_sens},step=epoch)
    wandb.log({curr_loader_name+'_test_auc':test_auc},step=epoch)
    wandb.log({curr_loader_name+'_test_spe':test_spec},step=epoch)
    wandb.log({curr_loader_name+'_test_loss':test_loss},step=epoch)
  table=wandb.Table(dataframe=out_frame)
  wandb.log({'metric_table':table})
  wandb.log({'out_path':curr_out_dir})
