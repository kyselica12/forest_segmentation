import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction:
    
    def forward(self, logits, targets, weight=None):
        raise NotImplementedError
    
    def apply_weight(self, x, weight):
        return x if weight is None else x * weight

class IoULoss(LossFunction):
    
    def __init__(self, n_classes, bakground_class=True, ignore_index=None) -> None:
        super(IoULoss).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.bkg_class = bakground_class

    def forward(self, logits, targets, weight=None):
        # preds[targets == self.ignore_index] = self.ignore_index

        start_idx = 1 if self.bkg_class else 0 
        iou = 0

        N = logits.size(0)
        preds = torch.softmax(logits, dim=1)

        for i in range(start_idx, self.n_classes):
            s1 = preds[:,i]
            s2 = (targets == i)
            intersection = self.apply_weight(s1 * s2, weight).sum()
            union = self.apply_weight(s1 + s2, weight).sum() - intersection
            iou += (1 - (intersection / union)) / N
            # iou += (- torch.log(intersection / union)) / N
        
        iou /= (self.n_classes - start_idx)

        return iou

class FocalLoss(LossFunction):
    
    def __init__(self, gamma=0, ingnore_index=255) -> None:
        super(FocalLoss).__init__()
        self.gamma = gamma
        self.ignore_index = ingnore_index
    
    def forward(self, logits, target, weight=None):

        logpt = -F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        pt = torch.exp(logpt) # probability of target class
        loss = -((1-pt)**self.gamma) * logpt
        loss = self.apply_weight(loss, weight)

        return loss.mean()

class CrossEntropyLoss(LossFunction):
    
    def __init__(self, ignore_index=255) -> None:
        super(CrossEntropyLoss).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, logits, target, weight=None):
        loss = F.cross_entropy(logits, target,reduction='none', ignore_index=self.ignore_index)
        loss = self.apply_weight(loss, weight)

        return loss.mean()
        