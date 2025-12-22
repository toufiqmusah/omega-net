import torch
import torch.nn as nn
from models import OMEGANet
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import NucleiDataset, DiceLoss, JaccardLoss


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
    
    # output = model(x1, x2)
    # print(f'Input shapes: {x1.shape}, {x2.shape}')
    # print(f'Output shape: {output.shape}')