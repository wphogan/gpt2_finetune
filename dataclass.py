# 
# PyTorch Dataset module for acronym dataset
#
import os
import csv
import json
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

# Local imports
from utils import  *

class ArcronymDataset(Dataset):
    def __init__(self, tokenizer, ROOT_DIR='', dataset_path = 'data/'):
        super().__init__()
        self.acronym_list = []
        
        acronym_data_path = os.path.join((ROOT_DIR + dataset_path), 'acronym_data.json')
        data = read_json(acronym_data_path)
        
        for entry in data:
            entry_id        = entry          
            inputs          = str(data[entry]['inputs']) + tokenizer.eos_token
            targets         = tokenizer.bos_token + str(data[entry]['targets']) + tokenizer.eos_token
            entryId_input_target = [entry_id, inputs, targets]
            self.acronym_list.append(entryId_input_target)
            
    def __len__(self):
        return len(self.acronym_list)

    def __getitem__(self, item):
        return self.acronym_list[item]