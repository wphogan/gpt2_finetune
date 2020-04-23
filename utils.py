import os
import json
import torch 
import logging
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger()

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
    input  = torch.tensor(tokenizer.encode(input)).to(device)
    target = torch.tensor(tokenizer.encode(target)).to(device)
    
    input_shape  = input.shape[0]
    target_shape = target.shape[0]
    
    if input_shape > target_shape:
        diff   = input_shape - target_shape
        target = F.pad(target,(0,diff), "constant", 0)
    
    elif input_shape < target_shape:
        diff  = target_shape - input_shape
        input = F.pad(input,(0,diff), "constant", 0)

    assert input.shape == target.shape
    
    return input.unsqueeze(0), target.unsqueeze(0)

def logits_to_text(logits, tokenizer):
    """
    Converts model's output logits to text and tokens
    """
    softmax_logits = torch.softmax(logits.squeeze(0).detach(), dim=1) #Take the first(from only one in this case) batch and the last predicted embedding
    token_ids      = torch.argmax(softmax_logits, dim=1)
    output_list    = list(token_ids.squeeze().to('cpu').numpy())
    output_text    = tokenizer.decode(output_list)
    return output_text, token_ids
    
        
def save_sample_outputs(file_path, target_text, output_text, epoch, loss, accuracy):
    """
    Saves a sample output text paired with the target text
    """
    file_path = file_path + f'/generated_{epoch}.sequences'
    with open(file_path, 'a') as f:
        f.write("EPOCH: {}, LOSS: {}, ACCURCY: {}\n".format(epoch, loss, accuracy))
        f.write("ID: " + f"{target_text}\n")
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