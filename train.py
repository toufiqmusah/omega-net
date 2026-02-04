import torch
import torch.nn as nn
from models import OMEGANet
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utils import NucleiDataset, NucleiDatasetFromPath, DiceLoss, JaccardLoss


# ==================== Training Function ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_jaccard': []}
    
    for epoch in range(num_epochs):

        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (img1, img2, mask) in enumerate(train_loader):
            img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
            
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_jaccard = 0.0
        
        with torch.no_grad():
            for img1, img2, mask in val_loader:
                img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
                output = model(img1, img2)
                
                loss = criterion(output, mask)
                val_loss += loss.item()
                
                # Calculate metrics
                dice_loss = DiceLoss()
                jaccard_loss = JaccardLoss()
                val_dice += (1 - dice_loss(output, mask).item())
                val_jaccard += (1 - jaccard_loss(output, mask).item())
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_jaccard /= len(val_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_jaccard'].append(val_jaccard)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Jaccard: {val_jaccard:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'omega_net_best.pth')
            print('  Saved best model!')
    
    return history


# ==================== Usage ====================

if __name__ == '__main__':

    model = OMEGANet(in_channels=3, base_filters=64, num_heads=8, dropout=0.1)
    
    # model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # ==================== Example 1: Using NucleiDatasetFromPath (Directory-based) ====================
    # This is the recommended approach when you have images organized in directories
    """
    # Directory structure should be:
    # Nuclei-Segmentation-Data/Nuclei_Segmentation_256/Train/
    # ├── Hematoxylin_MCT/
    # │   └── *.bmp
    # ├── Normalized_MCT/
    # │   └── *.bmp
    # └── Mask_MCT/
    #     └── *.bmp
    
    # Load dataset from directory
    data_dir = "Nuclei-Segmentation-Data/Nuclei_Segmentation_256/Train"
    full_dataset = NucleiDatasetFromPath(root_dir=data_dir)
    
    # Create train/validation split (e.g., 80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    # Note: num_workers can be adjusted based on available CPU cores
    # Use 0 for single-core systems or debugging
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Setup training
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, jaccard_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs=100, device=device)
    
    print("Training complete!")
    print(f"Best validation dice score: {max(history['val_dice']):.4f}")
    """
    
    # ==================== Example 2: Using NucleiDataset (Array-based) ====================
    # This approach is for when you have pre-loaded numpy arrays
    """
    # Assuming you have already loaded your data as numpy arrays:
    # images_norm: normalized images (N, H, W, C)
    # images_hema: hematoxylin images (N, H, W, C)
    # masks: ground truth masks (N, H, W, C)
    
    from sklearn.model_selection import train_test_split
    
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
    
    # Setup training
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, jaccard_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs=100, device=device)
    
    print("Training complete!")
    """
    
    # output = model(x1, x2)
    # print(f'Input shapes: {x1.shape}, {x2.shape}')
    # print(f'Output shape: {output.shape}')