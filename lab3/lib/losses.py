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
    """
    배치 크기 안에서 각 샘플에 대한 모든 클래스에 대한 확률이 softmax로 나오고, 각 샘플에서 클래스에 해당하는 확률이 0.8보다 크거나 같으면 1,작으면 0 으로 변환
    ex) [0.8,0.2,0.3]-->[1,0,0] : 높은 확신 라벨로
    """
    mask = F.softmax(outputs,dim=-1)
    mask = mask.ge(mask_prob).float().detach() # 이 mask는

    # -------------------------------------------------------------------------
    # Step 2: Minimize target's entropy
    # -------------------------------------------------------------------------
    scaled_targets = targets / temperature

    # -------------------------------------------------------------------------
    # Step 3: Compute KL divergence 
    # -------------------------------------------------------------------------
    """
    outputs,targets가 [batch,class]로 나오기 때문에 loss도 [batch,class]형태
    outputs는 변환하는 값(변형된 이미지에 대한 모델이 출력한 라벨)
    targets 변하지 않는 값(원본이미지에 대한 모델이 출력한 라벨) -->고정해야하므로 학습업데이트 x: detach()사용
    """
    outputs = torch.log_softmax(outputs, dim=-1)
    targets = torch.log_softmax(scaled_targets, dim=-1).detach()
    loss = F.kl_div(outputs, targets, log_target=True, reduction='none') # 하나의 샘플안에 각 클래스별 손실 값이 출력

    # -------------------------------------------------------------------------
    # Step 4: Mask out the low confident sample 
    # -------------------------------------------------------------------------
    """
    하나의 샘플에서 확실한 클래스 확률을 곱해주면서 필요한 클래스의 손실만 가져오고 나머지는 0과 곱해져서 사라짐.
    """
    loss = loss * mask 

    # -------------------------------------------------------------------------
    # Step 5: Compute the average loss  
    # -------------------------------------------------------------------------
    loss = loss.sum(dim=1)
    loss = loss.mean()

    return loss
