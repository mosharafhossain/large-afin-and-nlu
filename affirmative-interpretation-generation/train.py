# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import random
import numpy as np
import torch
import tqdm
import argparse
import json
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    Adafactor,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from model import AFINGenerator
from data import AFINDataset
from config import Config
import time
import utils

torch.cuda.empty_cache()

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)  
      torch.cuda.manual_seed_all(seed)    
    
    
def get_optimizer(params, model):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": params.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if params.optimizer == "AdamW":
        optimizer = AdamW(grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon)
    elif params.optimizer == "Adafactor":
        optimizer = Adafactor(grouped_parameters, lr=params.learning_rate, scale_parameter=False,
                         relative_step=False)
    return optimizer



start_time = time.time()
argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)  
args        = argParser.parse_args()
config_path = args.config_path



# Step 1: Read parameters and set a seed ----------------------------------------------------
with open(config_path) as json_file_obj: 
	params = json.load(json_file_obj)

params = Config(params)
set_seed(params.seed)

# Step 2: Get appropriate device ------------------------------------------------------------
if torch.cuda.is_available()==True and params.use_gpu: 
    device = torch.device("cuda:"+str(params.device))
else: 
    device = torch.device("cpu") 



# Step 3: Get tokenizer and initialize the model------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained(params.T5_path[params.T5_type])
model = AFINGenerator(params, device, tokenizer) 
model.to(device)  
print("Model is successfully loaded!")
state = dict(model=model.state_dict())



# Step 4: Prepare the datasets for data loader -------------------------------------------------
target_attribute_name = "pi"
train_data = utils.read_data(params.data_path["train"])
train_dataset_gold = AFINDataset(tokenizer, train_data, params, target_attribute_name)
dev_data   = utils.read_data(params.data_path["dev"])
dev_dataset   = AFINDataset(tokenizer, dev_data, params, target_attribute_name)
train_size = len(train_dataset_gold)
dev_size   = len(dev_dataset)
print("\n train size: {}".format(train_size))
print("\n dev size: {}".format(dev_size))


# using weak labeled data (large afin corpus)
if params.with_weak_labels: #wld: weak labeled data
    large_afin_data = utils.read_data(params.large_afin_path)
    afin_dataset = AFINDataset(tokenizer, large_afin_data, params)    
    afin_dataset = [afin_dataset[i] for i in range(len(afin_dataset))] 
    train_size = len(train_dataset_gold + afin_dataset)


# Step 5: Get optimizer and Scheduler ----------------------------------------------------------
optimizer = get_optimizer(params, model)
batch_num = int(np.ceil(train_size/params.batch_size))
print(f"Number of batches for training: {batch_num}")
print("Number of batches for dev. data: {}".format(int(np.ceil(dev_size/params.batch_size))))
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * params.warmup_epoch,
                                           num_training_steps=batch_num * params.max_epoch)



# Step 6: Train the model and save the best model--------------------------------------------------------
min_loss = float('inf')
best_epoch = 0
# Training the model
for epoch in range(params.max_epoch):
    # training set    
    optimizer.zero_grad()
        
    
    # If blended training is activated    
    if params.with_weak_labels and params.blended_training:        
        if epoch + 1 >= params.blended_start_epoch and epoch + 1 <= params.blended_end_epoch:                
            size_afin = len(afin_dataset)
            indices = np.random.choice(size_afin, round( (1-params.alpha)*size_afin), replace=False)
            afin_dataset = [afin_dataset[i] for i in indices]            
        elif epoch + 1 > params.blended_end_epoch:
            afin_dataset = []
    
    if params.with_weak_labels:        
        train_dataset = train_dataset_gold + afin_dataset
        indices = np.random.choice(len(train_dataset), len(train_dataset), replace=False)
        train_dataset = [train_dataset[i] for i in indices]
    else:        
        train_dataset = train_dataset_gold
    
    train_size = len(train_dataset)
    batch_num = int(np.ceil(train_size/params.batch_size))
    progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train size: {}, Train Epoch {}/{}'.format(train_size, epoch+1, params.max_epoch ))
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=params.train_shuffle) 
    
    for batch_idx, batch in enumerate(train_loader):              
        loss = model(batch)  
        loss = loss * (1 / params.accumulate_step)
        if params.use_multi_gpus:
            loss.mean().backward()
        else:
            loss.backward()
        
        
        if (batch_idx + 1) % params.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        print(f"batch_idx: {batch_idx}, loss: {loss}")
    
    
    if params.is_model_tuning and (epoch + 1) > params.epochs_no_eval:                  
        dev_output = []
        dev_gold   = []
        
        dev_loader = DataLoader(dev_dataset, batch_size=params.batch_size, shuffle=False) 
        dev_loss = 0
        for batch_idx, batch in enumerate(dev_loader):                     
            #model.eval()
            
            temp_loss = model(batch)
            
            dev_loss += temp_loss.item()
            #print(f"batch_idx: {batch_idx}, dev batch: {len(batch)}, dev_loss: {dev_loss/(batch_idx+1)}")   
            
            if params.use_multi_gpus:
                dev_batch_output = model.module.predict(batch)
            else:
                dev_batch_output = model.predict(batch)
            dev_output.extend(dev_batch_output)  
            
        
        current_loss = dev_loss/(batch_idx+1)
        print(f"\nBest epoch: {best_epoch}, Current loss: {current_loss}, min loss: {min_loss}, dev_output: {len(dev_output)}")        
        if current_loss < min_loss:
            min_loss = current_loss 
            best_epoch = epoch
            print('Saving the best model at {}'.format(params.best_model_path))
            torch.save(state, params.best_model_path)
            patience = 0
        else:
            patience += 1
            if patience > params.patience:
                break
        
            
    progress.close()  

if not params.is_model_tuning: 
    print('Saving the current model at {}'.format(params.best_model_path))
    torch.save(state, params.best_model_path)
    
end_time = time.time()
elapsed_time = end_time - start_time
print("Completed! Training duration: {} hours".format(elapsed_time/3600.0))
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  
    
