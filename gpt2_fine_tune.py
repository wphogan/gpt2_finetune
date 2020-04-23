# Fine-tuning GPT-2 in PyTorch
import torch
import shutil
import random
import logging
import warnings
import numpy as np
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW

# Local imports 
from utils import *
from dataclass import *

# Experiment tracking
import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.run import Run
from zipfile import ZipFile, ZIP_DEFLATED

# Experiment title and save location
ex = Experiment('Acronym Resolution: GPT2 Fine Tuning')
ex.observers.append(MongoObserver(url=os.getenv('CMI_MONGO_URI'), db_name='BACNORM_LPSN'))

# Set random seeds and timestamp
__TIMESTAMP = datetime.now().strftime('%m%d_%H%M%S')
random.seed(13)
np.random.seed(13)


# Directory settings
__ROOT_DIR       = ''
__OUT_DIR        = __ROOT_DIR +'out/' + __TIMESTAMP + '/' # Where outputs are saved
__MODEL_DIR      = __OUT_DIR  + 'trained_models/' # Where model is saved for reuse

make_directory(__OUT_DIR)
make_directory(__MODEL_DIR)

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
        for root, dirs, files in os.walk(__OUT_DIR):
            for file in files:
                zfile.write(os.path.join(root, file))
    ex.add_artifact('artifacts.zip')
    os.remove('artifacts.zip')    
     
# Get dataset
def get_dataset():
    dataset = ArcronymDataset(__ROOT_DIR)
    acronym_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return acronym_loader

# Init model
def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model     = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    return tokenizer, model

@ex.capture
def log_message(_log, log_type='info', message='Logging test.'):
    """
    Function to handle all Sacred logging.
    Inputs:
    Log type takes in debug, info, warning, error, critical.
    Message is the message that is logged.
    """
    if   log_type == 'info' : _log.info(message)
    elif log_type == 'debug' : _log.debug(message)
    elif log_type == 'warning' : _log.warning(message)
    elif log_type == 'error' : _log.error(message)
    elif log_type == 'critical' : _log.critical(message)
    
@ex.automain
def run_experiment(_log, _run: Run, epochs: int, batch_size: int, warmup_steps: int, max_seq_len: int,
                   learning_rate: float):
    
    # Load tokenizer and model
    log_message(_log, message="Loading model and tokenizer...")
    tokenizer, model = get_model()
    log_message(_log, message="Loaded.")
    
    # Load dataset
    log_message(_log, message="Pre-processing data.")
    acronym_loader = get_dataset()
    log_message(_log, message="Pre-process complete.")

    # Set optimizer, scheduler, and vars
    log_message(_log, message="Starting model...")
    model = model.to(device)
    model.train()
    optimizer      = AdamW(model.parameters(), lr=learning_rate)
    scheduler      = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps = -1)
    proc_seq_count = 0
    sum_loss       = 0.0
    batch_count    = 0
    
    
    for epoch in range(epochs):    
        epoch_log = '=' * 10 + f" EPOCH {epoch} started " + '=' * 10
        log_message(_log, log_type='info', message=epoch_log) 
        
        for idx ,acronym in enumerate(acronym_loader):
            if idx == 0 or idx % 100 == 0:
                log_message(_log, log_type='info', message=("Iteration: "+ str(idx) ))
            
            entry_id    = acronym[0][0]
            input_text  = acronym[1][0]
            target_text = acronym[2][0]
            
            ##### QUESTION!! PADDING ADDED TO START OR END OF TENSOR?
            ##### DOES THAT CHANGE FOR INPUT VS TARGET TENSORS?
            input_tokens, target_tokens  = padded_tensors(tokenizer, input_text, target_text, device)

            outputs = model(input_tokens, labels=target_tokens)
            loss, logits = outputs[:2]                        
            loss.backward()
            sum_loss = sum_loss + loss.detach().item()
                        
            proc_seq_count = proc_seq_count + 1
            
            if proc_seq_count == batch_size:
                proc_seq_count = 0    
                batch_count += 1
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                log_sum_loss = f"sum loss {sum_loss}"
                log_message(_log, log_type='info', message=log_sum_loss)
                
                batch_count = 0
                sum_loss = 0.0
            
            # Produce model-generated output text and output tokens
            output_text, output_tokens = logits_to_text(logits, tokenizer)
            
            # Calculate accuracy
            accuracy = calc_accuracy(target_tokens, output_tokens)
            
            # Record current progress
            if idx % 200 == 0:
                # Record loss and accuracy
                _run.log_scalar("training.loss", loss.item())
                _run.log_scalar("training.accuracy", accuracy) 
                
                # Log loss and accuracy for debugging purposes
                msg = "Loss: {}, Accuracy: {}".format(loss.item(), accuracy)
                log_message(_log, log_type='info', message=msg)
                
                # Save model generated vs target text
                save_sample_outputs(__OUT_DIR, target_text, output_text, epoch, loss.item(), accuracy)

            # =========
            # END EPOCH
            # =========

        # Save the model after each epoch
        log_message(_log, log_type='info', message="Saving model...")
        torch.save(model.state_dict(), os.path.join(__MODEL_DIR, f"gpt2_medium_acronym_{epoch}.pt"))
        log_message(_log, log_type='info', message="Saved.")
        
    print('Training complete.')