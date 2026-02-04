## <p align="center">[OMEGA-Net: Dual-Channel Progressive Depth Attention Network for Nuclei Segmentation in Diverse Histopathological Datasets](link)</p>

## Overview
OMEGA-Net is a dual-input U-Net architecture with Kolmogorov-Arnold Network (KAN) convolutions and attention mechanisms for nuclei segmentation in histopathological images. The model accepts two types of input images (Hematoxylin stained and Normalized) and produces segmentation masks.

## Dataset Structure

### Directory-based Dataset (Recommended)
Organize your images in the following directory structure:

```
Nuclei-Segmentation-Data/Nuclei_Segmentation_256/Train/
├── Hematoxylin_MCT/
│   ├── 01_1_p1.bmp
│   ├── 01_1_p2.bmp
│   └── ...
├── Normalized_MCT/
│   ├── 01_1_p1.bmp
│   ├── 01_1_p2.bmp
│   └── ...
└── Mask_MCT/
    ├── 01_1_p1.bmp
    ├── 01_1_p2.bmp
    └── ...
```

**Important Notes:**
- All three subdirectories must contain matching filenames
- Supported image formats: `.bmp`, `.png`, `.jpg`, `.jpeg`
- Images are automatically normalized to [0, 1] range
- Expected image size: 256x256 pixels
- Masks should be grayscale images (single channel)

## Installation

```bash
pip install torch torchvision pillow numpy
```

## Usage

### Option 1: Loading from Directory (Recommended)

```python
from models import OMEGANet
from utils import NucleiDatasetFromPath, CombinedLoss
from torch.utils.data import DataLoader, random_split
import torch

# Load dataset from directory
data_dir = "Nuclei-Segmentation-Data/Nuclei_Segmentation_256/Train"
full_dataset = NucleiDatasetFromPath(root_dir=data_dir)

# Create train/validation split (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Initialize model
model = OMEGANet(in_channels=3, base_filters=64, num_heads=8, dropout=0.1)

# Setup training
criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, jaccard_weight=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from train import train_model
history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                     num_epochs=100, device=device)
```

### Option 2: Loading from Pre-loaded Arrays

```python
from models import OMEGANet
from utils import NucleiDataset, CombinedLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming you have pre-loaded numpy arrays:
# images_norm: normalized images (N, H, W, C)
# images_hema: hematoxylin images (N, H, W, C)
# masks: ground truth masks (N, H, W, C)

# Split data
indices = np.arange(len(images_norm))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

# Create datasets
train_dataset = NucleiDataset(
    images_norm[train_idx], 
    images_hema[train_idx], 
    masks[train_idx]
)
val_dataset = NucleiDataset(
    images_norm[val_idx], 
    images_hema[val_idx], 
    masks[val_idx]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Train as shown above
```

## Model Architecture

The OMEGA-Net architecture features:
- **Dual Input Streams**: Processes Hematoxylin and Normalized images separately
- **KAN Convolutions**: Uses learnable B-spline basis functions
- **Multi-Scale Attention**: Combines spatial and channel attention mechanisms
- **Progressive Depth**: Multi-pooling strategy with transformer encoders
- **Attention Gates**: For effective skip connections in the decoder

## Training

To train the model, run:

```bash
python train.py
```

The training script will:
1. Load the dataset from the specified directory
2. Split into training and validation sets
3. Train for the specified number of epochs
4. Save the best model based on validation loss

## Data Preparation

1. **Organize your images** into the required directory structure
2. **Ensure filenames match** across all three subdirectories
3. **Verify image format**: BMP, PNG, or JPG (256x256 recommended)
4. **Mask format**: Grayscale images where white (255) indicates nuclei regions

Example data preparation script:

```python
import os
from PIL import Image

# Create directory structure
os.makedirs("data/Train/Hematoxylin_MCT", exist_ok=True)
os.makedirs("data/Train/Normalized_MCT", exist_ok=True)
os.makedirs("data/Train/Mask_MCT", exist_ok=True)

# Copy or move your images to the appropriate directories
# Make sure filenames match across all three directories
```

## Model Parameters

- `in_channels`: Number of input channels (default: 3 for RGB)
- `base_filters`: Base number of filters (default: 64)
- `num_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)

## Loss Functions

The model supports multiple loss functions:
- **DiceLoss**: Measures overlap between prediction and ground truth
- **JaccardLoss**: IoU-based loss function
- **CombinedLoss**: Weighted combination of BCE, Dice, and Jaccard losses

## Citation

If you use this code in your research, please cite:

```bibtex
@article{omega-net,
  title={OMEGA-Net: Dual-Channel Progressive Depth Attention Network for Nuclei Segmentation},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Add your license information here]
