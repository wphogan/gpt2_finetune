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

# Fine-tuning GPT-2 in PyTorch
import torch
import shutil
import random
import logging
import warnings
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW

# Local imports 
from utils import *
from dataclass import *

# Experiment tracking
from sacred.run import Run
from sacred import Experiment
from sacred.observers import MongoObserver
from zipfile import ZipFile, ZIP_DEFLATED

# Experiment title and save location
ex = Experiment('Acronym Resolution: GPT2 Fine Tuning')
ex.observers.append(MongoObserver(url=os.getenv('CMI_MONGO_URI'), db_name='BACNORM_LPSN'))


# System variables
__ROOT_DIR = '' # Set this if on UCSD cluster
random.seed(13)
np.random.seed(13)

# Use GPU if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

@ex.config
def config():
    # Hyperparameters
    epochs        = 5
    batch_size    = 16
    max_seq_len   = 400
    warmup_steps  = 5000
    learning_rate = 3e-5

@ex.post_run_hook
def post_run():
    with ZipFile('artifacts.zip', 'w', ZIP_DEFLATED) as zfile:
        for root, dirs, files in os.walk('out/'):
            for file in files:
                zfile.write(os.path.join(root, file))
    ex.add_artifact('artifacts.zip')
    os.remove('artifacts.zip')    
   
@ex.pre_run_hook
def pre_run():
    if os.path.exists('out/'):
        shutil.rmtree('out/')
    os.makedirs('out/')
     
# Get dataset
def get_dataset():
    print('Processing data...', end=' ')
    dataset = ArcronymDataset(__ROOT_DIR)
    acronym_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print('processed.')
    return acronym_loader

# Init model
def get_model():
    print('Loading model and tokenizer...', end=' ')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model     = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    print('loaded.')
    return tokenizer, model

@ex.automain
def run_experiment(_run: Run, epochs: int, batch_size: int, warmup_steps: int, max_seq_len: int,
                   learning_rate: float):
    
    # Load tokenizer and model
    tokenizer, model = get_model()
    
    # Load dataset
    acronym_loader = get_dataset()

    # Set optimizer, scheduler, and vars
    model = model.to(device)
    model.train()
    optimizer      = AdamW(model.parameters(), lr=learning_rate)
    scheduler      = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps = -1)
    proc_seq_count = 0
    sum_loss       = 0.0
    batch_count    = 0

    models_folder = __ROOT_DIR + "trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    print('Starting model...')
    
    for epoch in range(epochs):     
        print(f"EPOCH {epoch} started" + '=' * 30)
        
        for idx ,acronym in enumerate(acronym_loader):
            if idx == 0 or idx == 1 or idx % 200 == 0:
                print("Iteration: ", idx)

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
            if proc_seq_count == batch_size:
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
        
    print('Training complete.')