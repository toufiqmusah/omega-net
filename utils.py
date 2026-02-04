import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


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


class NucleiDatasetFromPath(Dataset):
    """
    Dataset class for loading nuclei segmentation images from directory structure.
    
    Expected directory structure:
        root_dir/
        ├── Hematoxylin_MCT/
        │   └── *.bmp (or .png, .jpg)
        ├── Normalized_MCT/
        │   └── *.bmp (or .png, .jpg)
        └── Mask_MCT/
            └── *.bmp (or .png, .jpg)
    
    Args:
        root_dir (str): Root directory containing the three subdirectories
        transform (callable, optional): Optional transform to be applied on images
        extensions (tuple): Tuple of valid image extensions (default: ('.bmp', '.png', '.jpg', '.jpeg'))
    """
    
    def __init__(self, root_dir, transform=None, extensions=('.bmp', '.png', '.jpg', '.jpeg')):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions
        
        # Define subdirectories
        self.hema_dir = os.path.join(root_dir, 'Hematoxylin_MCT')
        self.norm_dir = os.path.join(root_dir, 'Normalized_MCT')
        self.mask_dir = os.path.join(root_dir, 'Mask_MCT')
        
        # Validate directories exist
        for dir_path, dir_name in [(self.hema_dir, 'Hematoxylin_MCT'),
                                     (self.norm_dir, 'Normalized_MCT'),
                                     (self.mask_dir, 'Mask_MCT')]:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory not found: {dir_path}")
        
        # Get list of image files from Hematoxylin directory
        self.image_files = []
        for file in sorted(os.listdir(self.hema_dir)):
            if file.lower().endswith(self.extensions):
                self.image_files.append(file)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.hema_dir}")
        
        # Verify that matching files exist in all three directories
        self._validate_files()
    
    def _validate_files(self):
        """Validate that all files exist in all three directories"""
        missing_files = []
        for filename in self.image_files:
            hema_path = os.path.join(self.hema_dir, filename)
            norm_path = os.path.join(self.norm_dir, filename)
            mask_path = os.path.join(self.mask_dir, filename)
            
            if not os.path.exists(hema_path):
                missing_files.append(f"Hematoxylin_MCT/{filename}")
            if not os.path.exists(norm_path):
                missing_files.append(f"Normalized_MCT/{filename}")
            if not os.path.exists(mask_path):
                missing_files.append(f"Mask_MCT/{filename}")
        
        if missing_files:
            raise ValueError(f"Missing files: {', '.join(missing_files)}")
    
    def _load_image(self, path):
        """Load an image and convert to numpy array normalized to [0, 1]"""
        try:
            img = Image.open(path)
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array
        except Exception as e:
            raise IOError(f"Error loading image {path}: {str(e)}")
    
    def _load_mask(self, path):
        """Load a mask and convert to single channel numpy array normalized to [0, 1]"""
        try:
            mask = Image.open(path)
            # Convert to grayscale if not already
            if mask.mode != 'L':
                mask = mask.convert('L')
            # Convert to numpy array and normalize
            mask_array = np.array(mask, dtype=np.float32) / 255.0
            # Add channel dimension (H, W) -> (H, W, 1)
            mask_array = np.expand_dims(mask_array, axis=-1)
            return mask_array
        except Exception as e:
            raise IOError(f"Error loading mask {path}: {str(e)}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # Load images
        hema_path = os.path.join(self.hema_dir, filename)
        norm_path = os.path.join(self.norm_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        hema_img = self._load_image(hema_path)
        norm_img = self._load_image(norm_path)
        mask_img = self._load_mask(mask_path)
        
        # Apply transforms if provided
        if self.transform:
            hema_img = self.transform(hema_img)
            norm_img = self.transform(norm_img)
            mask_img = self.transform(mask_img)
        
        # Convert to tensors and permute to CHW format
        norm_tensor = torch.from_numpy(norm_img).permute(2, 0, 1)  # HWC -> CHW
        hema_tensor = torch.from_numpy(hema_img).permute(2, 0, 1)  # HWC -> CHW
        mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1)  # HWC -> CHW
        
        return norm_tensor, hema_tensor, mask_tensor