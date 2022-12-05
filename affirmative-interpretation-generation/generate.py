# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import random
import numpy as np
import torch
import argparse
import json
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer
)

from model import AFINGenerator
from data import AFINDataset, NewDataset
from config import Config
import utils


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
 

#Example commands:
# With default file: python generate.py -c ./config/config.json
# With user's file: python generate.py -c ./config/config.json -inp INPUT_FILE_PATH -out OUTPUT_FILE_PATH

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", 
                       "--config_path", 
                       help="Path to the configuration file", 
                       required=True)        
argParser.add_argument("-inp", 
                       "--path_input_file", 
                       help="Path to the user's specified input text file. Each line in the file should contain a single sentence/text", 
                       default=None,
                       required=False) 

argParser.add_argument("-out", 
                       "--path_output_file", 
                       help="Path to the output text file", 
                       default=None,
                       required=False) 


args        = argParser.parse_args()
config_path = args.config_path
path_input_file = args.path_input_file
path_output_file = args.path_output_file


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


# Step 3: Get tokenizer --------------------------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained(params.T5_path[params.T5_type])



# Step 4: Initialize model and load with the trained weights--------------------------------
map_location = 'cuda:{}'.format(params.device) if params.use_gpu else 'cpu'
state = torch.load(params.best_model_path, map_location=map_location)

model = AFINGenerator(params, device, tokenizer) 
model.load_state_dict(state['model'])
model.to(device)  
model.eval() 
print(f"Model is loaded from {params.best_model_path}")



# Step 4: Prepare the datasets for the data loader -------------------------------------------------
if path_input_file and path_output_file:
    test_data = utils.read_text_data(path_input_file)
    test_dataset = NewDataset(tokenizer, test_data, params)
elif params.data_path["test"]:
    test_data = utils.read_data(params.data_path["test"])
    target_attribute_name = "pi"
    test_dataset = AFINDataset(tokenizer, test_data, params, target_attribute_name)
else:
    print("No input/output data file is found!")
    exit(0)


test_size = len(test_dataset)
batch_num = int(np.ceil(test_size/params.batch_size))
print(f"Number of batches to complete: {batch_num}")



# Step 5: Generate batches-------------------------------------------------
test_output = []
test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)


for batch_idx, batch in enumerate(test_loader):                     

    if params.use_multi_gpus:
        test_batch_output = model.module.predict(batch)
    else:
        test_batch_output = model.predict(batch)
    test_output.extend(test_batch_output)  


if path_input_file and path_output_file:
    utils.save_text_output(test_output, params.num_return_sequences, path_output_file)
elif params.output_path["test"]:
    utils.save_output(test_output, params.num_return_sequences, params.output_path["test"])

print(f"\nCompleted!")