import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import random

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # Allows for faster training
    torch.backends.cudnn.benchmark = True  # Can speed up training significantly

set_seed(42)

# Simple configuration
CONFIG = {
    'data_root': 'path/to/data',  # Will be overridden by command line
    'output_dir': './football_results',
    'num_frames': 16,  # Reduced number of frames for speed
    'frame_size': 112,  # Smaller frame size for speed
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 3e-4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'model_type': 'r3d',  # Simplest model for faster training
    'test_size': 0.2,  # Simple train/test split instead of cross-validation
    'num_workers': 4,  # Parallel data loading
}

# Simplified video dataset
class FootballVideoDataset(Dataset):
    def __init__(self, video_paths, labels, config, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.config = config
        self.is_training = is_training
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess video frames
        frames = self._load_video(video_path)
        
        return frames, label
    
    def _load_video(self, video_path):
        """Optimized video loading"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Simple uniform sampling - faster than complex temporal augmentation
        if frame_count <= self.config['num_frames']:
            indices = np.arange(frame_count)
            indices = np.concatenate([indices] * (self.config['num_frames'] // frame_count + 1))
            indices = indices[:self.config['num_frames']]
        else:
            indices = np.linspace(0, frame_count - 1, self.config['num_frames'], dtype=int)
        
        # Extract frames
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize - faster than complex augmentation
                frame = cv2.resize(frame, (self.config['frame_size'], self.config['frame_size']))
                
                # Basic augmentation only for training
                if self.is_training and random.random() < 0.5:
                    # Horizontal flip (simple but effective augmentation)
                    frame = cv2.flip(frame, 1)
                
                # Normalize [0-255] to [0-1] and standardize
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - np.array([0.43216, 0.394666, 0.37645])) / np.array([0.22803, 0.22145, 0.216989])
                
                frames.append(frame)
            else:
                # Handle error: duplicate last frame if reading fails
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create an empty frame
                    empty_frame = np.zeros((self.config['frame_size'], self.config['frame_size'], 3), dtype=np.float32)
                    frames.append(empty_frame)
        
        cap.release()
        
        # Convert to tensor [C, T, H, W]
        frames = np.array(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))  # [T, H, W, C] -> [C, T, H, W]
        
        return torch.from_numpy(frames).float()

# Simple R3D model - the fastest 3D CNN for this task
class R3DModel(nn.Module):
    def __init__(self, num_classes):
        super(R3DModel, self).__init__()
        self.r3d = video_models.r3d_18(pretrained=True)
        
        # Replace final layer
        in_features = self.r3d.fc.in_features
        self.r3d.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.r3d(x)

# Create model
def create_model(model_type, num_classes):
    # Focus only on R3D for simplicity and speed
    return R3DModel(num_classes)

# Load dataset
def load_dataset(config):
    video_paths = []
    labels = []
    label_to_idx = {}
    
    # Load video paths and labels
    data_root = Path(config['data_root'])
    print("Loading dataset...")
    
    for idx, class_folder in enumerate(sorted(os.listdir(data_root))):
        class_path = data_root / class_folder
        if class_path.is_dir():
            print(f"Found class {idx}: {class_folder}")
            label_to_idx[class_folder] = idx
            
            video_files = [f for f in os.listdir(class_path) 
                         if f.endswith(('.mp4', '.avi', '.mov'))]
            
            if not video_files:
                print(f"Warning: No video files found in {class_folder}")
                continue
                
            for video_file in video_files:
                video_paths.append(str(class_path / video_file))
                labels.append(idx)
    
    if not video_paths:
        raise ValueError(f"No videos found in {data_root}")
        
    print(f"Found {len(video_paths)} videos across {len(label_to_idx)} classes")
    
    # Split data into train and test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        video_paths, labels, test_size=config['test_size'], random_state=42, stratify=labels
    )
    
    return train_paths, test_paths, train_labels, test_labels, label_to_idx

# Train function
def train(config):
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load and split dataset
    train_paths, test_paths, train_labels, test_labels, label_to_idx = load_dataset(config)
    num_classes = len(label_to_idx)
    
    # Create datasets
    train_dataset = FootballVideoDataset(train_paths, train_labels, config, is_training=True)
    test_dataset = FootballVideoDataset(test_paths, test_labels, config, is_training=False)
    
    # Create data loaders with prefetching
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = create_model(config['model_type'], num_classes)
    model.to(config['device'])
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    
    # Initialize tracking
    best_acc = 0.0
    best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Mixed precision for speed
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        # Calculate epoch statistics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': val_correct / val_total
                })
        
        # Calculate epoch statistics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        # Report performance
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_acc > best_acc:
            print(f"Validation accuracy improved from {best_acc:.4f} to {epoch_val_acc:.4f}. Saving model...")
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_curves.png'))
    plt.close()
    
    # Final evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config['device'])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_to_idx.keys()),
                yticklabels=list(label_to_idx.keys()))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'confusion_matrix.png'))
    plt.close()
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Model saved to {best_model_path}")
    print(f"Results and visualizations saved to {config['output_dir']}")
    
    # Save label mapping
    import json
    with open(os.path.join(config['output_dir'], 'label_mapping.json'), 'w') as f:
        json.dump(label_to_idx, f, indent=4)
    
    return best_model_path, label_to_idx

# Main function
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Football Video Classification (Simplified)')
    parser.add_argument('--data_root', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./football_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to sample')
    parser.add_argument('--frame_size', type=int, default=112, help='Frame size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG.update({
        'data_root': args.data_root,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'num_frames': args.num_frames,
        'frame_size': args.frame_size,
        'learning_rate': args.learning_rate,
        'num_workers': args.num_workers
    })
    
    # Run training
    train(CONFIG)

if __name__ == "__main__":
    main()