import os
import json
import torch 
import timeit
import logging
import datetime as date_time
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger()

class Timer:
    """Measure time used."""
    def __init__(self, round_ndigits: int = 0):
        self._round_ndigits = round_ndigits
        self._start_time = timeit.default_timer()

    def __call__(self):
        return timeit.default_timer() - self._start_time

    def __str__(self):
        return str(date_time.timedelta(seconds=round(self(), self._round_ndigits)))
    
def make_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception:
        logger.warn("Error -- could not create directory.")

def read_json(filename) :
    """
    Reads json input into dictionary
    """
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
        return data
    except Exception:
        raise ValueError('Could not read json file: {}'.format(filename))

def padded_tensors(tokenizer, input, target, device='cpu'):
    """
    Function that takes input and target strings and creates tokenized 
    tensors, with added padding to match sizes
    """
    input  = torch.tensor(tokenizer.encode(input, pad_to_max_length=True, padding_side='right', return_attention_masks=True )).to(device)
    target = torch.tensor(tokenizer.encode(target, pad_to_max_length=True, padding_side='left' )).to(device)
    assert input.shape == target.shape
    return input.unsqueeze(0).to(device), target.unsqueeze(0).to(device)

def logits_to_text(logits, tokenizer):
    """
    Converts model's output logits to text and tokens
    """
    softmax_logits = torch.softmax(logits.squeeze(0).detach(), dim=1) #Take the first(from only one in this case) batch and the last predicted embedding
    token_ids      = torch.argmax(softmax_logits, dim=1)
    output_list    = list(token_ids.squeeze().to('cpu').numpy())
    output_text    = tokenizer.decode(output_list)
    return output_text, token_ids
    
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
    
def set_device(_log):
    if torch.cuda.is_available():
        log_message(_log, message="Nice! Using GPU.")
        return 'cuda'
    else: 
        log_message(_log, log_type='warning', message="Watch out! Using CPU.")
        return 'cpu'
    
def save_model(_log, __MODEL_DIR, epoch, model):
    log_message(_log, log_type='info', message="Saving model...")
    torch.save(model.state_dict(), os.path.join(__MODEL_DIR, f"gpt2_medium_acronym_{epoch}.pt"))
    log_message(_log, log_type='info', message="Saved.")
    
def save_sample_outputs(file_path, entry_id, target_text, output_text, epoch, loss, accuracy):
    """
    Saves a sample output text paired with the target text
    """
    file_path = file_path + f'/generated_{epoch}.sequences'
    output_text = output_text.split('<|endoftext|>')[0]
    with open(file_path, 'a') as f:
        f.write("EPOCH: {}, LOSS: {}, ACCURCY: {}\n".format(epoch, loss, accuracy))
        f.write("ID: " + f"{entry_id}\n")
        f.write("TARGET TEXT: " + f"{target_text}\n")
        f.write("OUTPUT TEXT: " + f"{output_text}\n\n")

def calc_accuracy(target_tokens, output_tokens):
    """
    Calculates accuracy of generted sequences versus target sequences
    """
    correct  = (target_tokens.squeeze(0) == output_tokens).float().sum()
    total    = target_tokens.shape[1]
    accuracy = float(correct.item() / total)
    return accuracy