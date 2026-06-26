import torch
from einops import einsum, rearrange, reduce

def calculate_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:


    # Find the vocab index corresponding to the target token
    # Is this one target token or many?
    # I believe the targets has to be teh index otherwise this becomes a really expensive 
    
    # Spell it out quicklu
    # logits is batch_size x seq_len x vocab_size - For each word in each sequence, it is a distribution over all vocabs
    # targets is batch_size x seq_len - It is the batch of sentences. Contains a sequence of words index as per vocab
    # To get the loss, we will have to first do it for a single batch and then sum it across all batches
    # For a single sentence, we look at each word one by one
    # For each word, get the index from targets[index]
    # For that word, get the distribution from logits[batch][index]
    # Now do teh calculation and repeat it for every index to complete loss calculation for a single sequence
    # Now repeat it for all batches and sum it up to get the final loss

    max_val = reduce(logits, 'b s v -> b s', 'max')
    max_val = rearrange(max_val, 'b s -> b s 1')
    # logits is b s v, max_val is b s 1
    logits = logits - max_val
    targets = rearrange(targets, 'b s -> b s 1')

    loss =  rearrange(torch.gather(input = logits, dim=2, index=targets), 'b s 1 -> b s') \
            - torch.log(torch.sum(torch.exp(logits), dim=2))#

    loss = -reduce(loss, 'b s -> 1', 'mean') # sum over sequence length
    return loss
