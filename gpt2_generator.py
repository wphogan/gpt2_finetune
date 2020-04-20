
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW

from utils import * 

"""
Generating Sequences
"""
# Use GPU if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
__ROOT_DIR = ''
MODEL_EPOCH = 4
models_folder = __ROOT_DIR + "trained_models"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

model_path = os.path.join(models_folder, f"gpt2_medium_acronym_{MODEL_EPOCH}.pt")
model.load_state_dict(torch.load(model_path))

acronyms_output_file_path = __ROOT_DIR + f'out/generated_{MODEL_EPOCH}.sequences'

model.eval()
if os.path.exists(acronyms_output_file_path):
    os.remove(acronyms_output_file_path)
    
acronym_num = 0
with torch.no_grad():
   
    for acronym_idx in range(1000):
    
        acronym_finished = False

        #### FIX THIS
        cur_ids = torch.tensor(tokenizer.encode("JOKE:")).unsqueeze(0).to(device)

        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                acronym_finished = True
                break

        
        if acronym_finished:
            
            acronym_num = acronym_num + 1
            
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)

            with open(acronyms_output_file_path, 'a') as f:
                f.write(f"{output_text} \n\n")
                
    
# 3rd epoch model seemed to perform the best.