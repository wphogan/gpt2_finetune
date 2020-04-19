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
    def __init__(self, dataset_path = 'data/'):
        super().__init__()
        self.acronym_list = []
        self.end_of_text_token = "<|endoftext|>"
        
        acronym_data_path = os.path.join(dataset_path, 'acronym_data.json')
        data = read_json(acronym_data_path)
        
        for entry in data:
            entry_id        = entry          
            inputs          = str(data[entry]['inputs']) + str(self.end_of_text_token)
            targets         = str(data[entry]['targets']) + str(self.end_of_text_token)
            entryId_input_target = [entry_id, inputs, targets]
            self.acronym_list.append(entryId_input_target)
            
    def __len__(self):
        return len(self.acronym_list)

    def __getitem__(self, item):
        return self.acronym_list[item]