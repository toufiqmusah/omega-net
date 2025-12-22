import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ==================== Loss Functions ====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, jaccard_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.jaccard = JaccardLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.jaccard_weight = jaccard_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        jaccard_loss = self.jaccard(pred, target)
        return (self.bce_weight * bce_loss + 
                self.dice_weight * dice_loss + 
                self.jaccard_weight * jaccard_loss)


# ==================== Dataset Class ====================
class NucleiDataset(Dataset):
    def __init__(self, images_norm, images_hema, masks):
        self.images_norm = torch.FloatTensor(images_norm)
        self.images_hema = torch.FloatTensor(images_hema)
        self.masks = torch.FloatTensor(masks)
        
    def __len__(self):
        return len(self.images_norm)
    
    def __getitem__(self, idx):
        return (self.images_norm[idx].permute(2, 0, 1),  # HWC -> CHW
                self.images_hema[idx].permute(2, 0, 1),
                self.masks[idx].permute(2, 0, 1))