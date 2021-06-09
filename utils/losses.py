"""Loss functions
"""

import torch

from torch.nn import functional as F


def getValid(true, pred, nonvalid=-1):
    """ On CDNEt dataset, the frames are not fully labeled. Only some predefined region of them are labeled.
    This function extracts the labeled part from ground truth and the corresponding part from prediction as  1-D tensors
    Args:
        true (tensor): Ground truth tensor of shape Bx1xHxW
        preds (tensor): Prediction tensor of shape Bx1xHxW
        nonvalid (int): Value used to indicate nonvalid parts of ground truth

    Returns:
        (tensor): 1-D tensor containing the valid pixels of ground truth
        (tensor): 1-D tensor of prediction corresponding the valid ground truth pixels
    """
    # Turn predictions and labels into 1D arrays
    true_valid = true.reshape(-1)
    pred_valid = pred.reshape(-1)

    # Mask of the known parts of the ground truth
    mask = torch.where(true_valid == nonvalid, torch.tensor(0).cuda(), torch.tensor(1).cuda()).type(torch.bool)

    # Discard the unknown parts from the predictions and labels
    return torch.masked_select(true_valid, mask), torch.masked_select(pred_valid, mask)

def jaccard_loss(true, pred, smooth=100):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
        eps (int): Smoothing factor
    Returns:
        jacc_loss: the Jaccard loss.
    """
    intersection = torch.sum(true*pred)
    jac = (intersection + smooth) / (torch.sum(true) + torch.sum(pred) - intersection + smooth)
    return (1 - jac) * smooth

def weighted_crossentropy(true, pred, weight_pos=15, weight_neg=1):
    """Weighted cross entropy between ground truth and predictions
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
    Returns:
        (tensor): Weighted CE.
    """
    bce = (true*pred.log()) + ((1-true)*(1-pred).log())  # Binary cross-entropy

    # Weighting for class imbalance
    weight_vector = true * weight_pos + (1. - true) * weight_neg
    weighted_bce = weight_vector * bce
    return -torch.mean(weighted_bce)

def acc(true, pred):
    """Accuracy between ground truth and predictions
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
    Returns:
        acc: Accuracy.
    """
    return torch.mean((true == pred.round()).float())

def f_score(true, pred):
    """False Negative Rate between ground truth and predictions
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
    Returns:
        (tensor): precision
        (tensor): recall
        (tensor): f-score
    """
    fn = torch.sum(true * (1 - pred))
    fp = torch.sum((1 - true) * pred)
    tp = torch.sum(true * pred)
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)

    if tp+fn == 0:
        f_score = torch.tensor(1)
    elif tp == 0:
        f_score = torch.tensor(0)
    else:
        f_score = 2 * (prec * recall) / (prec + recall)

    return f_score
