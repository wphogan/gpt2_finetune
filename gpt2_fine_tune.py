# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: 'Python 3.7.4 64-bit (''anaconda3'': virtualenv)'
#     language: python
#     name: python37464bitanaconda3virtualenv6cea5baf3a944c8087a64ca66fd11a70
# ---

# # Fine-tuning GPT-2 on a jokes dataset in PyTorch
#
# This notebook was created as a part of a blog post - [Fine-tuning large Transformer models on a single GPU in PyTorch - Teaching GPT-2 a sense of humor](https://mf1024.github.io/2019/11/12/Fun-With-GPT-2/). Here I demonstrate how to fine-tune a pre-trained GPT-2 model on a jokes dataset. 
#
# #### If you haven't yet, check out the notebook in this [gist](https://gist.github.com/mf1024/430d7fd6ff527350d3e4b5bda0d8614e) where use the same pretrained model to generate text.

# +
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
import numpy as np

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

# Local imports 
from dataclass import *
from utils import *
# -

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# Get dataset
# dataset = JokesDataset()
# joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)

print('Processing data...', end=' ')
dataset = ArcronymDataset()
acronym_loader = DataLoader(dataset, batch_size=1, shuffle=True)
print('processed.')


# Init model
print('Loading model and tokenizer...', end=' ')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)
print('loaded.')

# ### Hyperparameters
# For a parameter value starting point for fine-tuning, I inspired from [this](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py) and [this](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py) huggingface fine-tuning code.

# +
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400
# -

# ### Model training
#
# I will train the model and save the model weights after each epoch and then I will try to generate jokes with each version of the weight to see which performs the best.

# +
model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

input_tmp_acronym_tens = None
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

print('Starting model...')
for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,acronym in enumerate(acronym_loader):
        input = acronym[1][0]
        target = acronym[2][0]
        
        
        
        ##### QUESTION!! PADDING ADDED TO START OR END OF TENSOR?
        ##### DOES THAT CHANGE FOR INPUT VS TARGET TENSORS?
        input_acronym_tens, target_acronym_tens  = padded_tensors(tokenizer, input, target, device)

        outputs = model(input_acronym_tens, labels=target_acronym_tens)
        loss, logits = outputs[:2]                        
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
                       
        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0
    
    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_acronym_{epoch}.pt"))
            
# -
# ### Generating the sequences
# +
MODEL_EPOCH = 4

models_folder = "trained_models"

model_path = os.path.join(models_folder, f"gpt2_medium_acronym_{MODEL_EPOCH}.pt")
model.load_state_dict(torch.load(model_path))

acronyms_output_file_path = f'generated_{MODEL_EPOCH}.sequences'

model.eval()
if os.path.exists(acronyms_output_file_path):
    os.remove(acronyms_output_file_path)
    
acronym_num = 0
with torch.no_grad():
   
        for acronym_idx in range(1000):
        
            acronym_finished = False

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

                with open(acronym_output_file_path, 'a') as f:
                    f.write(f"{output_text} \n\n")
                    
      
# 3rd epoch model seemed to perform the best.