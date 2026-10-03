import numpy as np

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    raise NotImplementedError

def dice_loss(probs, targets, eps=1.0):
    raise NotImplementedError

def pinball_loss(y, pred, tau):
    raise NotImplementedError

def label_smoothing_ce(logits, y, eps=0.1):
    raise NotImplementedError
