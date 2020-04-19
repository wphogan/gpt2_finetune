import json
import torch 
import torch.nn.functional as F
import numpy as np

def choose_from_top(probs, n=1):
    """
    Used to generate text
    """
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def read_json(filename) :
    """
    Reads json input into dictionary
    """
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
        return data
    except:
        return False

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
