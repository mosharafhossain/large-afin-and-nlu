# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import json
import sys
import os

def read_text_data(file_path):
    """
    Read a file that contains one sentence in each line.
    :param file_path: path to the input file
    """
    
    
    try:
        file_obj = open(file_path, 'r')
    except FileNotFoundError:
        file_name = os.path.basename(file_path)
        print(f"File {file_name} not found at {file_path}!")
        sys.exit(1)
    except OSError:
        print(f"OS error occurred when trying to open {file_path}")
        sys.exit(1)
    except Exception as e:
        print("Unexpected error occured! Detail error: {}".format(repr(e)))
        sys.exit(1) 
    else:
      with file_obj:
          data_list = []
          for line in file_obj:
              data_list.append(line)
          return data_list
      
def read_data(file_path):
    """
    Read a file that contains a json object in each line.
    :param file_path: path to the input file
    """
    
    
    try:
        file_obj = open(file_path, 'r')
    except FileNotFoundError:
        file_name = os.path.basename(file_path)
        print(f"File {file_name} not found at {file_path}!")
        sys.exit(1)
    except OSError:
        print(f"OS error occurred when trying to open {file_path}")
        sys.exit(1)
    except Exception as e:
        print("Unexpected error occured! Detail error: {}".format(repr(e)))
        sys.exit(1) 
    else:
      with file_obj:
          data_list = []
          for line in file_obj:
              data_list.append(json.loads(line))
          return data_list
    


def write_data(data_list, file_path):
    """
    Write json data into a file. Each line of the output file contains a json object.
    :param data_list: list of data containing dictionary objects
    :param file_path: path to the output file
    """
          
    try:
        with open(file_path, "w", encoding="utf-8") as file_obj:
            for json_obj in data_list:
                file_obj.write(json.dumps(json_obj, ensure_ascii=False))
                file_obj.write("\n")
    
    except Exception as e:
        print("Unexpected error occured! Detail error: {}".format(repr(e)))
        sys.exit(1) 
                
          
            
def save_text_output(outputs, num_ret_seq, file_path):
    """
    Save generated affirmative interpretation.    
    :param outputs: list of generated outputs. Each list element is a dictionary containg n (i.e., num_ret_seq) number of generated affirmative interpretations for each input
    :param num_ret_seq: number of generated affirmative interpretations for each input sentence
    :param file_path: path to the output file
    """
    
    try:
        with open(file_path, "w", encoding="utf-8") as file_obj:
            for i, text in enumerate(outputs):            
                if i% num_ret_seq == 0:  
                    afin = text.replace("<pad>", "").replace("</s>", "").replace("pi:", "").replace("affirmative_interpretation:", "").strip()
                    file_obj.write(afin)
                    file_obj.write("\n")
                    
    except Exception as e:
        print("Unexpected error occured! Detail error: {}".format(repr(e)))
        sys.exit(1) 
        

def save_output(outputs, num_ret_seq, file_path):
    """
    Save generated affirmative interpretation.    
    :param outputs: list of generated outputs. Each list element is a dictionary containg n (i.e., num_ret_seq) number of generated affirmative interpretations for each input
    :param num_ret_seq: number of generated affirmative interpretations for each input sentence
    :param file_path: path to the output file
    """
    
    try:
        with open(file_path, "w", encoding="utf-8") as file_obj:
            gen_texts = {}
            for i, text in enumerate(outputs):            
                if i% num_ret_seq == 0: 
                    gen_texts["affirmative_interpretation"] = text.replace("<pad>", "").replace("</s>", "").replace("pi:", "").replace("affirmative_interpretation:", "").strip()
                    file_obj.write(json.dumps(gen_texts))
                    file_obj.write("\n")
                    gen_texts = {}
                    
    except Exception as e:
        print("Unexpected error occured! Detail error: {}".format(repr(e)))
        sys.exit(1) 
                
    
