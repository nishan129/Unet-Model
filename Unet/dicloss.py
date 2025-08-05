import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        total_dice = 0
        for i in range(predictions.shape[1]):
            pred_channel = predictions[:, i:i+1, :, :].contiguous().view(-1)
            target_channel = targets[:, i:i+1, :, :].contiguous().view(-1)
            
            intersection = (pred_channel * target_channel).sum()
            dice = (2.0 * intersection + self.smooth) / (pred_channel.sum() + target_channel.sum() + self.smooth)
            total_dice += dice
        
        return 1 - (total_dice / predictions.shape[1])
    
    
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss