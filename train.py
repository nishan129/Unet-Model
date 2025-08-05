import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import time
from tqdm import tqdm
from Unet.color_net import ColorConditionalUNet
from Unet.datasets import ColorConditionalDataset
from Unet.dicloss import CombinedLoss
import wandb




COLOR_MAPPING = {
    'red': 0, 'green': 1, 'blue': 2, 'yellow': 3, 
    'orange': 4, 'purple': 5, 'cyan': 6, 'magenta': 7
}

REVERSE_COLOR_MAPPING = {v: k for k, v in COLOR_MAPPING.items()}

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """Calculate Intersection over Union (IoU) metric"""
    pred_binary = (pred_mask > threshold).float()
    true_binary = true_mask.float()
    
    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def calculate_accuracy(pred_mask, true_mask, threshold=0.5):
    """Calculate pixel-wise accuracy"""
    pred_binary = (pred_mask > threshold).float()
    true_binary = true_mask.float()
    
    correct = (pred_binary == true_binary).float().sum()
    total = true_binary.numel()
    
    return (correct / total).item()

def train_color_conditional_unet(dataset_path, config):
    """Train the color-conditional U-Net model with wandb logging"""
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ColorConditionalDataset(
        dataset_path, 
        split='training', 
        image_size=config.image_size,
        augment=config.augment
    )
    
    val_dataset = ColorConditionalDataset(
        dataset_path, 
        split='validation', 
        image_size=config.image_size,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Log dataset info to wandb
    wandb.log({
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "num_train_batches": len(train_loader),
        "num_val_batches": len(val_loader)
    })
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ColorConditionalUNet(
        n_channels=config.n_channels, 
        n_classes=config.n_classes, 
        n_colors=len(COLOR_MAPPING),
        bilinear=config.bilinear
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Log model info to wandb
    wandb.log({"model_parameters": num_params})
    wandb.watch(model, log="all", log_freq=100)
    
    # Loss function and optimizer
    criterion = CombinedLoss(alpha=config.loss_alpha)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=config.patience, 
        factor=0.5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    print(f"Starting training for {config.num_epochs} epochs...")
    
    for epoch in range(config.num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_acc = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]')
        
        for batch_idx, (images, color_indices, masks, color_names) in enumerate(train_pbar):
            images = images.to(device)
            color_indices = color_indices.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, color_indices)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(outputs)
                batch_iou = calculate_iou(pred_sigmoid, masks)
                batch_acc = calculate_accuracy(pred_sigmoid, masks)
                
                train_loss += loss.item()
                train_iou += batch_iou
                train_acc += batch_acc
                num_train_batches += 1
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{batch_iou:.4f}',
                'Acc': f'{batch_acc:.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_acc = 0.0
        num_val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Val]')
        
        with torch.no_grad():
            for images, color_indices, masks, color_names in val_pbar:
                images = images.to(device)
                color_indices = color_indices.to(device)
                masks = masks.to(device)
                
                outputs = model(images, color_indices)
                loss = criterion(outputs, masks)
                
                # Calculate metrics
                pred_sigmoid = torch.sigmoid(outputs)
                batch_iou = calculate_iou(pred_sigmoid, masks)
                batch_acc = calculate_accuracy(pred_sigmoid, masks)
                
                val_loss += loss.item()
                val_iou += batch_iou
                val_acc += batch_acc
                num_val_batches += 1
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{batch_iou:.4f}',
                    'Acc': f'{batch_acc:.4f}'
                })
        
        # Calculate average metrics
        avg_train_loss = train_loss / num_train_batches
        avg_train_iou = train_iou / num_train_batches
        avg_train_acc = train_acc / num_train_batches
        
        avg_val_loss = val_loss / num_val_batches
        avg_val_iou = val_iou / num_val_batches
        avg_val_acc = val_acc / num_val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to wandb
        epoch_time = time.time() - start_time
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_iou": avg_train_iou,
            "val_iou": avg_val_iou,
            "train_accuracy": avg_train_acc,
            "val_accuracy": avg_val_acc,
            "learning_rate": current_lr,
            "epoch_time": epoch_time
        })
        
        # Save best model based on validation IoU
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_val_loss = avg_val_loss
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_iou': avg_train_iou,
                'val_iou': avg_val_iou,
                'color_mapping': COLOR_MAPPING,
                'config': dict(config)
            }
            
            torch.save(checkpoint, config.save_path)
            
            # Save model artifact to wandb
            model_artifact = wandb.Artifact(
                name=f"color_conditional_unet_epoch_{epoch+1}",
                type="model",
                description=f"Best model at epoch {epoch+1} with val_iou={avg_val_iou:.4f}"
            )
            model_artifact.add_file(config.save_path)
            wandb.log_artifact(model_artifact)
            
            print(f"✓ New best model saved (Val IoU: {avg_val_iou:.4f}, Val Loss: {avg_val_loss:.4f})")
        
        # Log learning rate changes
        if new_lr != current_lr:
            print(f"Learning rate reduced from {current_lr:.2e} to {new_lr:.2e}")
        
        print(f'Epoch [{epoch+1}/{config.num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}, '
              f'Time: {epoch_time:.2f}s')
    
    # Log final training summary
    wandb.log({
        "best_val_loss": best_val_loss,
        "best_val_iou": best_val_iou,
        "total_epochs_trained": config.num_epochs
    })
    
    return model, train_losses, val_losses

def predict_with_color(model_path, input_image_path, desired_color, output_path, config):
    """
    Predict colored shape with specified color
    
    Args:
        model_path: Path to trained model
        input_image_path: Path to input shape image
        desired_color: Color name (e.g. 'red', 'blue', 'green')
        output_path: Path to save prediction
        config: Model configuration
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ColorConditionalUNet(
        n_channels=config.n_channels, 
        n_classes=config.n_classes,
        n_colors=len(COLOR_MAPPING)
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(input_image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get color index
    if desired_color not in COLOR_MAPPING:
        raise ValueError(f"Color '{desired_color}' not supported. Available colors: {list(COLOR_MAPPING.keys())}")
    
    color_idx = torch.tensor([COLOR_MAPPING[desired_color]]).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor, color_idx)
        prediction = torch.sigmoid(output).squeeze().cpu()
    
    # Convert to RGB image format
    prediction_rgb = prediction.permute(1, 2, 0)
    prediction_rgb = torch.clamp(prediction_rgb, 0, 1)
    prediction_rgb = (prediction_rgb * 255).numpy().astype(np.uint8)
    
    # Save prediction
    pred_image = Image.fromarray(prediction_rgb, mode='RGB')
    pred_image.save(output_path)
    
    print(f"Prediction with color '{desired_color}' saved to: {output_path}")
    
    return prediction_rgb

def test_color_conditional_model(model_path, dataset_path, config, num_samples=5):
    """Test the model with different color combinations and log to wandb"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ColorConditionalUNet(
        n_channels=config.n_channels, 
        n_classes=config.n_classes,
        n_colors=len(COLOR_MAPPING)
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load validation dataset
    val_dataset = ColorConditionalDataset(dataset_path, split='validation', image_size=config.image_size)
    
    # Select random samples
    indices = torch.randperm(len(val_dataset))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    test_images = []
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, color_idx, mask, color_name = val_dataset[idx]
            
            # Denormalize image for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_vis = image * std + mean
            image_vis = torch.clamp(image_vis, 0, 1)
            
            # Make predictions with original color and a different color
            image_batch = image.unsqueeze(0).to(device)
            color_batch = torch.tensor([color_idx]).to(device)
            
            # Original prediction
            output = model(image_batch, color_batch)
            prediction = torch.sigmoid(output).squeeze().cpu()
            prediction_rgb = prediction.permute(1, 2, 0)
            prediction_rgb = torch.clamp(prediction_rgb, 0, 1)
            
            # Different color prediction (e.g., red)
            different_color_idx = COLOR_MAPPING['blue'] if color_name != 'blue' else COLOR_MAPPING['red']
            different_color_batch = torch.tensor([different_color_idx]).to(device)
            output_diff = model(image_batch, different_color_batch)
            prediction_diff = torch.sigmoid(output_diff).squeeze().cpu()
            prediction_diff_rgb = prediction_diff.permute(1, 2, 0)
            prediction_diff_rgb = torch.clamp(prediction_diff_rgb, 0, 1)
            
            # Plot
            axes[i, 0].imshow(image_vis.permute(1, 2, 0))
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.permute(1, 2, 0))
            axes[i, 1].set_title(f'Ground Truth ({color_name})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(prediction_rgb)
            axes[i, 2].set_title(f'Prediction ({color_name})')
            axes[i, 2].axis('off')
            
            diff_color_name = REVERSE_COLOR_MAPPING[different_color_idx]
            axes[i, 3].imshow(prediction_diff_rgb)
            axes[i, 3].set_title(f'Different Color ({diff_color_name})')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Log the test results to wandb
    wandb.log({"test_predictions": wandb.Image(fig)})
    plt.show()

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot and log training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb
    wandb.log({"training_curves": wandb.Image(plt)})
    plt.show()

def get_default_config():
    """Get default configuration dictionary"""
    return {
        'image_size': (256, 256),
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'n_channels': 3,
        'n_classes': 3,
        'bilinear': True,
        'augment': True,
        'loss_alpha': 0.5,
        'patience': 15,
        'num_workers': 4,
        'save_path': 'color_conditional_unet.pth'
    }

# Main script
if __name__ == "__main__":
    # Configuration
    config_dict = get_default_config()
    
    # Initialize wandb with proper configuration
    run = wandb.init(
        project="color-conditional-unet",
        name=f"unet_lr{config_dict['learning_rate']}_bs{config_dict['batch_size']}_ep{config_dict['num_epochs']}",
        config=config_dict,
        save_code=True,
        tags=["unet", "color-conditional", "segmentation"]
    )
    
    # Get config from wandb (allows for hyperparameter sweeps)
    config = wandb.config
    
    dataset_path = '/content/data/dataset'  # Update this path
    
    if os.path.exists(dataset_path):
        print("🚀 Starting color-conditional training...")
        
        try:
            # Train the model
            model, train_losses, val_losses = train_color_conditional_unet(dataset_path, config)
            
            # Plot training curves
            plot_training_curves(train_losses, val_losses, 'color_conditional_training_curves.png')
            
            # Test the model
            print("\n🧪 Testing the color-conditional model...")
            test_color_conditional_model(config.save_path, dataset_path, config, num_samples=5)
            
            print("\n🎉 Training completed!")
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            wandb.log({"error": str(e)})
            raise
        
        finally:
            wandb.finish()
        
    else:
        print(f"❌ Dataset path '{dataset_path}' does not exist!")
        wandb.finish()