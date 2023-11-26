import torch
import torch.nn.functional as F


def consistency_loss(outputs, targets, temperature=0.4, mask_prob=0.8):
    """ Consistency loss for unsupervised data augmentation (UDA)
    Step1) Build mask that indicates the high confident samples, whose softmax prob 
           greater than the mask_prob, to build pseudo-labels. For example, if the ith
           sample's softmax prob is greater than the mask_prob, then the mask[i] should
           be 1.0. If not, the mask[i] should be 0.0
    Step2) Scale targets to minimize entropy using a temperature variable
    Step3) Compute KL divergence between the softmax probabilites of outputs and that of targets
    Step4) Mask out the loss on low confident samples by multiply mask
    Step5) Compute the average loss on mini-batch

    Args:
        outptus (B, K): model's prediction for noisy (augmented) unlabeled data
        targets (B, K): model's prediction for clean unlabled data
    
    Returns: 
        loss (1, ): consistency loss
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Build mask
    # -------------------------------------------------------------------------
    # Fill this
    mask = mask.ge(mask_prob).float().detach()

    # -------------------------------------------------------------------------
    # Step 2: Minimize target's entropy
    # -------------------------------------------------------------------------
    # Fill this

    # -------------------------------------------------------------------------
    # Step 3: Compute KL divergence 
    # -------------------------------------------------------------------------
    outputs = torch.log_softmax(outputs, dim=-1)
    targets = torch.log_softmax(targets / temperature, dim=-1).detach()
    loss = F.kl_div(outputs, targets, log_target=True, reduction='none')

    # -------------------------------------------------------------------------
    # Step 4: Mask out the low confident sample 
    # -------------------------------------------------------------------------
    # Fill this

    # -------------------------------------------------------------------------
    # Step 5: Compute the average loss  
    # -------------------------------------------------------------------------
    loss = loss.sum(dim=1)
    loss = loss.mean()

    return loss