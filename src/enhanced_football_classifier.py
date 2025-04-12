import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchvision.models.video as video_models
import numpy as np
import cv2
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm
import random
import albumentations as A
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Default configuration
DEFAULT_CONFIG = {
    'data_root': 'path/to/extracted/folder',  # Update this with your folder path
    'output_dir': './football_classifier_results',
    
    # Dataset parameters
    'num_frames': 32,  # Increased number of frames
    'frame_height': 224,  # Larger input size
    'frame_width': 224,
    'clip_duration': 10,  # Duration of each clip in seconds
    
    # Training parameters
    'batch_size': 8,
    'num_epochs': 10,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'optimizer': 'adamw',  # 'adamw', 'sgd'
    'scheduler': 'cosine',
    'lr_min': 1e-6,  # Minimum learning rate for cosine scheduler
    'early_stopping_patience': 10,
    'gradient_accumulation_steps': 4,  # Accumulate gradients over multiple batches
    
    # Model parameters
    'model_type': 'r3d', #'mc3', 'r2plus1d', 'i3d', 'x3d'
    'dropout': 0.5,
    'pretrained': True,
    
    # Augmentation parameters
    'use_augmentation': True,
    'spatial_aug_prob': 0.8,
    'temporal_aug_prob': 0.5,
    
    # Cross-validation
    'num_folds': 5,
    
    # Device
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'mixed_precision': True,  # Use mixed precision training
}

# Video dataset class with augmentations
class EnhancedFootballDataset(Dataset):
    def __init__(self, video_paths, labels, config, augment=False, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.config = config
        self.augment = augment
        self.transform = transform
        
        # Spatial augmentations
        self.spatial_transform = A.Compose([
            A.RandomResizedCrop(size=(config['frame_height'], config['frame_width']), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        
        # Non-augmented transform
        self.basic_transform = A.Compose([
            A.Resize(height=config['frame_height'], width=config['frame_width']),
            A.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess video frames
        frames = self._load_video(video_path)
        
        # Apply additional transforms if specified
        if self.transform:
            frames = self.transform(frames)
            
        return frames, label
    
    def _temporal_augment(self, frames, total_frames):
        """Apply temporal augmentations to the frames"""
        target_frames = self.config['num_frames']
        
        if random.random() < 0.5:
            # Speed up or slow down by sampling frames at different rates
            rate_multiplier = random.uniform(0.8, 1.2)
            effective_length = int(target_frames * rate_multiplier)
            effective_length = min(effective_length, total_frames)
            
            if effective_length <= target_frames:
                # Sample and possibly duplicate frames
                indices = np.linspace(0, total_frames - 1, effective_length, dtype=int)
                indices = np.concatenate([indices] * (target_frames // effective_length + 1))
                indices = indices[:target_frames]
            else:
                # Downsample frames
                indices = np.linspace(0, effective_length - 1, target_frames, dtype=int)
                indices = np.clip(indices, 0, total_frames - 1)
                
        else:
            # Standard uniform sampling
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            
        # Temporally jitter frames
        if random.random() < 0.3:
            jitter = np.random.randint(-2, 3, size=len(indices))
            indices = np.clip(indices + jitter, 0, total_frames - 1)
            
        return indices
    
    def _load_video(self, video_path):
        """Load video and extract frames with augmentations"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frames to sample
        if self.augment and random.random() < self.config['temporal_aug_prob']:
            # Apply temporal augmentation
            indices = self._temporal_augment(None, frame_count)
        else:
            # Standard uniform sampling
            if frame_count <= self.config['num_frames']:
                indices = np.arange(frame_count)
                indices = np.concatenate([indices] * (self.config['num_frames'] // frame_count + 1))
                indices = indices[:self.config['num_frames']]
            else:
                indices = np.linspace(0, frame_count - 1, self.config['num_frames'], dtype=int)
        
        # Extract the selected frames
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply spatial augmentations
                if self.augment and random.random() < self.config['spatial_aug_prob']:
                    frame = self.spatial_transform(image=frame)['image']
                else:
                    frame = self.basic_transform(image=frame)['image']
                
                frames.append(frame)
            else:
                # Handle error: duplicate last frame if reading fails
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create an empty frame
                    empty_frame = np.zeros((self.config['frame_height'], self.config['frame_width'], 3), dtype=np.uint8)
                    if self.augment:
                        empty_frame = self.spatial_transform(image=empty_frame)['image']
                    else:
                        empty_frame = self.basic_transform(image=empty_frame)['image']
                    frames.append(empty_frame)
        
        cap.release()
        
        # Convert to tensor of shape [C, T, H, W]
        frames = np.array(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))  # [T, H, W, C] -> [C, T, H, W]
        
        return torch.from_numpy(frames).float()


# Enhanced base model with dropout
class EnhancedR3DModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5, pretrained=True):
        super(EnhancedR3DModel, self).__init__()
        self.r3d = video_models.r3d_18(pretrained=pretrained)
        
        # Replace the final fully connected layer with a dropout layer followed by a linear layer
        in_features = self.r3d.fc.in_features
        self.r3d.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.r3d(x)

class MC3Model(nn.Module):
    def __init__(self, num_classes, dropout=0.5, pretrained=True):
        super(MC3Model, self).__init__()
        self.mc3 = video_models.mc3_18(pretrained=pretrained)
        
        # Replace the final fully connected layer with dropout
        in_features = self.mc3.fc.in_features
        self.mc3.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.mc3(x)

# Create model based on config
def create_model(config, num_classes):
    model_type = config['model_type'].lower()
    pretrained = config['pretrained']
    dropout = config['dropout']
    
    if model_type == 'r3d':
        return EnhancedR3DModel(num_classes, dropout, pretrained)
    elif model_type == 'mc3':
        return MC3Model(num_classes, dropout, pretrained)
    else:
        return EnhancedR3DModel(num_classes, dropout, pretrained)

# Create optimizer based on config
def create_optimizer(config, model):
    
    if config['optimizer'].lower() == 'adamw':
        return optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'].lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

# Create scheduler based on config
def create_scheduler(config, optimizer):
    if config['scheduler'].lower() == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config['lr_min']
        )
    elif config['scheduler'].lower() == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {config['scheduler']}")

# Load and prepare dataset
def load_dataset(config):
    video_paths = []
    labels = []
    label_to_idx = {}
    
    # Load all video paths and labels
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
    print(f"Class mapping: {label_to_idx}")
    
    return video_paths, labels, label_to_idx

# Cross-validation training function
def train_with_cross_validation(config):
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump({k: str(v) if isinstance(v, torch.device) else v 
                  for k, v in config.items()}, f, indent=4)
    

    
    # Load dataset
    video_paths, labels, label_to_idx = load_dataset(config)
    num_classes = len(label_to_idx)
    
    # Initialize cross-validation
    kfold = StratifiedKFold(n_splits=config['num_folds'], shuffle=True, random_state=42)
    splits = kfold.split(video_paths, labels)
    
    
    # Store results for each fold
    fold_results = []
    best_fold_models = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{config['num_folds']}")
        print(f"{'='*50}")
        
        # Create datasets for this fold
        train_dataset = EnhancedFootballDataset(
            [video_paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            config, 
            augment=config['use_augmentation']
        )
        
        val_dataset = EnhancedFootballDataset(
            [video_paths[i] for i in val_idx],
            [labels[i] for i in val_idx],
            config,
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model, optimizer, and scheduler
        model = create_model(config, num_classes)
        model.to(config['device'])
        
        optimizer = create_optimizer(config, model)
        scheduler = create_scheduler(config, optimizer)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Using label smoothing
        
        # Initialize mixed precision scaler if enabled
        scaler = torch.cuda.amp.GradScaler('cuda') if config['mixed_precision'] else None
        
        # Training loop
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_model_path = os.path.join(config['output_dir'], f'best_model_fold{fold+1}.pth')
        patience_counter = 0
        
        # Initialize tracking variables
        fold_tracking = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'best_epoch': 0,
            'best_acc': 0.0,
            'learning_rates': []
        }
        
        for epoch in range(config['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_corrects = 0
            train_samples = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if config['mixed_precision']:
                    with torch.cuda.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                    # Scale the loss and perform backprop
                    scaler.scale(loss).backward()
                    
                    # Only update every gradient_accumulation_steps
                    if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # Only update every gradient_accumulation_steps
                    if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                        optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_corrects += torch.sum(preds == labels).item()
                train_samples += inputs.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': train_corrects / train_samples
                })
            
            # Calculate epoch statistics
            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_acc = train_corrects / train_samples
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
                for inputs, labels in progress_bar:
                    inputs = inputs.to(config['device'])
                    labels = labels.to(config['device'])
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels).item()
                    
                    # Save predictions for analysis
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': loss.item(), 
                        'acc': val_corrects / len(val_loader.dataset)
                    })
            
            # Calculate epoch statistics
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = val_corrects / len(val_loader.dataset)
            
            # Update scheduler if appropriate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(epoch_val_acc)
                else:
                    scheduler.step()
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{config['num_epochs']}")
            print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.8f}")
            
            # Update tracking
            fold_tracking['train_losses'].append(epoch_train_loss)
            fold_tracking['val_losses'].append(epoch_val_loss)
            fold_tracking['train_accs'].append(epoch_train_acc)
            fold_tracking['val_accs'].append(epoch_val_acc)
            fold_tracking['learning_rates'].append(current_lr)

            
            # Save best model
            if epoch_val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {epoch_val_acc:.4f}. Saving model...")
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                fold_tracking['best_epoch'] = epoch
                fold_tracking['best_acc'] = epoch_val_acc
                
                # Save model
                torch.save(model.state_dict(), best_model_path)
                
                # Reset patience counter
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve. Patience: {patience_counter}/{config['early_stopping_patience']}")
                
                # Early stopping
                if patience_counter >= config['early_stopping_patience']:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        # Final evaluation with best model
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(config['device'])
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_true, all_preds)
        cm = confusion_matrix(all_true, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='weighted')
        class_report = classification_report(all_true, all_preds, target_names=list(label_to_idx.keys()), output_dict=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(label_to_idx.keys()),
                    yticklabels=list(label_to_idx.keys()))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Fold {fold+1} - Confusion Matrix (Accuracy: {accuracy:.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], f'confusion_matrix_fold{fold+1}.png'))
        plt.close()
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(fold_tracking['train_losses'], label='Training Loss')
        plt.plot(fold_tracking['val_losses'], label='Validation Loss')
        plt.axvline(x=fold_tracking['best_epoch'], color='r', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold+1} - Loss Curves')
        
        plt.subplot(1, 3, 2)
        plt.plot(fold_tracking['train_accs'], label='Training Accuracy')
        plt.plot(fold_tracking['val_accs'], label='Validation Accuracy')
        plt.axvline(x=fold_tracking['best_epoch'], color='r', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Fold {fold+1} - Accuracy Curves')
        
        plt.subplot(1, 3, 3)
        plt.plot(fold_tracking['learning_rates'])
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title(f'Fold {fold+1} - Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], f'training_curves_fold{fold+1}.png'))
        plt.close()
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_epoch': fold_tracking['best_epoch'],
            'class_report': class_report,
            'confusion_matrix': cm.tolist(),
            'val_indices': val_idx.tolist(),  # Save validation indices for later analysis
            'best_model_path': best_model_path
        }
        
        fold_results.append(fold_result)
        best_fold_models.append((fold, best_val_acc, best_model_path))
    
    # Save all fold results
    with open(os.path.join(config['output_dir'], 'fold_results.json'), 'w') as f:
        json.dump(fold_results, f, indent=4)
    
    # Calculate average metrics across folds
    avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
    avg_precision = np.mean([res['precision'] for res in fold_results])
    avg_recall = np.mean([res['recall'] for res in fold_results])
    avg_f1 = np.mean([res['f1'] for res in fold_results])
    
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # Find best model across all folds
    best_fold, best_acc, best_model_path = max(best_fold_models, key=lambda x: x[1])
    print(f"Best model from Fold {best_fold+1} with accuracy {best_acc:.4f}")
    
    # Save best model as final model
    shutil.copy(best_model_path, os.path.join(config['output_dir'], 'final_model.pth'))
    
    # Save label mapping
    with open(os.path.join(config['output_dir'], 'label_mapping.json'), 'w') as f:
        json.dump(label_to_idx, f, indent=4)
    
    return fold_results, best_model_path, label_to_idx

# Perform error analysis
def error_analysis(config, model_path, label_to_idx, fold_results):
    """Analyze misclassified examples to understand model weaknesses"""
    print("\nPerforming detailed error analysis...")
    
    # Load best model
    model = create_model(config, len(label_to_idx))
    model.load_state_dict(torch.load(model_path))
    model.to(config['device'])
    model.eval()
    
    # Find best fold for analysis
    best_fold_result = max(fold_results, key=lambda x: x['accuracy'])
    best_fold = best_fold_result['fold'] - 1
    val_indices = best_fold_result['val_indices']
    
    # Load validation data from the best fold
    video_paths, labels, _ = load_dataset(config)
    
    val_videos = [video_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    val_dataset = EnhancedFootballDataset(
        val_videos, val_labels, config, augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Get predictions and confidences
    all_preds = []
    all_true = []
    all_probs = []
    all_videos = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_loader, desc="Evaluating examples")):
            batch_videos = val_videos[i*config['batch_size']:min((i+1)*config['batch_size'], len(val_videos))]
            
            inputs = inputs.to(config['device'])
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_videos.extend(batch_videos)
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame({
        'video_path': all_videos,
        'true_label': all_true,
        'predicted_label': all_preds,
        'correct': np.array(all_true) == np.array(all_preds)
    })
    
    # Add class names
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    results_df['true_class'] = results_df['true_label'].map(idx_to_label)
    results_df['predicted_class'] = results_df['predicted_label'].map(idx_to_label)
    
    # Add probabilities for each class
    for idx, class_name in idx_to_label.items():
        results_df[f'prob_{class_name}'] = [probs[idx] for probs in all_probs]
    
    # Add confidence (probability of predicted class)
    results_df['confidence'] = [probs[pred] for probs, pred in zip(all_probs, all_preds)]
    
    # Save full results
    results_df.to_csv(os.path.join(config['output_dir'], 'prediction_results.csv'), index=False)
    
    # Analyze errors
    error_df = results_df[~results_df['correct']]
    print(f"Total errors: {len(error_df)} out of {len(results_df)} ({len(error_df)/len(results_df)*100:.2f}%)")
    
    # Most common misclassifications
    misclass_counts = error_df.groupby(['true_class', 'predicted_class']).size().reset_index()
    misclass_counts.columns = ['True Class', 'Predicted Class', 'Count']
    misclass_counts = misclass_counts.sort_values('Count', ascending=False)
    
    print("\nMost common misclassifications:")
    print(misclass_counts.head(10))
    
    # Error rate by class
    class_error_rates = results_df.groupby('true_class')['correct'].agg(['count', lambda x: (~x).sum()])
    class_error_rates.columns = ['Total', 'Errors']
    class_error_rates['Error Rate'] = class_error_rates['Errors'] / class_error_rates['Total']
    class_error_rates = class_error_rates.sort_values('Error Rate', ascending=False)
    
    print("\nError rates by class:")
    print(class_error_rates)
    
    # Find extreme errors (high confidence wrong predictions)
    extreme_errors = error_df.sort_values('confidence', ascending=False)
    
    print("\nHighest confidence errors:")
    print(extreme_errors[['video_path', 'true_class', 'predicted_class', 'confidence']].head(10))
    
    # Visualize error distribution
    plt.figure(figsize=(14, 10))
    
    # Error rates by class plot
    plt.subplot(2, 2, 1)
    sns.barplot(x=class_error_rates.index, y=class_error_rates['Error Rate'])
    plt.title('Error Rate by Class')
    plt.xlabel('Class')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=45)
    
    # Confusion matrix for errors
    plt.subplot(2, 2, 2)
    error_cm = confusion_matrix(error_df['true_label'], error_df['predicted_label'])
    sns.heatmap(error_cm, annot=True, fmt='d', cmap='Reds',
               xticklabels=list(label_to_idx.keys()),
               yticklabels=list(label_to_idx.keys()))
    plt.title('Confusion Matrix of Errors')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Confidence distribution of errors vs correct predictions
    plt.subplot(2, 2, 3)
    sns.histplot(data=results_df, x='confidence', hue='correct', bins=20, 
                 kde=True, common_norm=False, stat='density')
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    
    # Most confused class pairs
    plt.subplot(2, 2, 4)
    top_misclass = misclass_counts.head(5)
    sns.barplot(x=top_misclass['True Class'] + ' → ' + top_misclass['Predicted Class'], 
                y=top_misclass['Count'])
    plt.title('Top 5 Most Common Misclassifications')
    plt.xlabel('Class Pair')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'error_analysis.png'))
    plt.close()
    
    # Write error analysis report
    with open(os.path.join(config['output_dir'], 'error_analysis_report.md'), 'w') as f:
        f.write('# Error Analysis Report\n\n')
        
        f.write('## Summary Statistics\n')
        f.write(f'- Total examples: {len(results_df)}\n')
        f.write(f'- Total errors: {len(error_df)}\n')
        f.write(f'- Overall error rate: {len(error_df)/len(results_df)*100:.2f}%\n\n')
        
        f.write('## Error Rates by Class\n')
        f.write(class_error_rates.to_markdown() + '\n\n')
        
        f.write('## Most Common Misclassifications\n')
        f.write(misclass_counts.head(10).to_markdown() + '\n\n')
        
        f.write('## Extreme Errors (High Confidence Mistakes)\n')
        f.write(extreme_errors[['video_path', 'true_class', 'predicted_class', 'confidence']].head(10).to_markdown() + '\n\n')
        
        f.write('## Insights and Recommendations\n')
        
        # Add specific insights based on the analysis
        most_confused_class = class_error_rates.index[0]
        most_common_error = misclass_counts.iloc[0]
        
        f.write(f'1. The class with the highest error rate is "{most_confused_class}" at {class_error_rates["Error Rate"].iloc[0]*100:.2f}%.\n')
        f.write(f'2. The most common misclassification is "{most_common_error["True Class"]}" being predicted as "{most_common_error["Predicted Class"]}" (occurred {most_common_error["Count"]} times).\n')
        
        # Look for confidence patterns
        avg_correct_conf = results_df[results_df['correct']]['confidence'].mean()
        avg_error_conf = results_df[~results_df['correct']]['confidence'].mean()
        
        f.write(f'3. Average confidence for correct predictions: {avg_correct_conf:.4f}\n')
        f.write(f'4. Average confidence for incorrect predictions: {avg_error_conf:.4f}\n')
        
        if avg_error_conf > 0.7:
            f.write('5. The model shows high confidence in its mistakes, suggesting overconfidence issues.\n')
        
        # Recommendations
        f.write('\n### Recommendations\n')
        f.write('1. Collect more training data for the most confused classes.\n')
        f.write('2. Review the videos that were misclassified with high confidence to identify patterns.\n')
        f.write('3. Consider using temporal attention mechanisms to focus on the most discriminative parts of the videos.\n')
        f.write('4. Explore ensemble methods combining different model architectures.\n')
        f.write('5. Implement test-time augmentation to improve prediction reliability.\n')
    
    return results_df, error_df

# Ablation study function
def perform_ablation_study(base_config):
    """
    Perform ablation study by testing different configurations
    """
    print("\nStarting ablation study...")
    
    # Define configurations to test
    ablation_configs = [
        # Different model architectures
        {'name': 'r3d_18', 'model_type': 'r3d', 'batch_size': 8},
        {'name': 'mc3_18', 'model_type': 'mc3', 'batch_size': 8},
        
        # Different learning rates
        {'name': 'lr_0.001', 'learning_rate': 0.001, 'model_type': 'r3d'},
        {'name': 'lr_0.0001', 'learning_rate': 0.0001, 'model_type': 'r3d'},
        
        # Different optimizers
        {'name': 'adamw', 'optimizer': 'adamw', 'model_type': 'r3d'},
        {'name': 'sgd', 'optimizer': 'sgd', 'model_type': 'r3d'},
        
        # Different frame counts
        {'name': 'frames_16', 'num_frames': 16, 'model_type': 'r3d'},
        {'name': 'frames_32', 'num_frames': 32, 'model_type': 'r3d'},
        
        # Different frame sizes
        {'name': 'size_224', 'frame_height': 224, 'frame_width': 224, 'model_type': 'r3d'},
        
        # With and without augmentation
        {'name': 'no_aug', 'use_augmentation': False, 'model_type': 'r3d'},
        {'name': 'with_aug', 'use_augmentation': True, 'model_type': 'r3d'},
        
        # Dropout values
        {'name': 'dropout_0.3', 'dropout': 0.3, 'model_type': 'r3d'},
        {'name': 'dropout_0.5', 'dropout': 0.5, 'model_type': 'r3d'},
        {'name': 'dropout_0.7', 'dropout': 0.7, 'model_type': 'r3d'},
    ]
    
    # Ensure at least 10 different configurations
    if len(ablation_configs) < 10:
        print("Warning: Less than 10 configurations for ablation study")
    
    # Run a smaller version of cross-validation for each config
    ablation_results = []
    
    for config_mods in ablation_configs:
        # Create a copy of the base config
        config = base_config.copy()
        
        # Apply modifications
        for key, value in config_mods.items():
            if key != 'name':
                config[key] = value
        
        # Shorter training for ablation
        config['num_epochs'] = 5
        config['num_folds'] = 2  # Use fewer folds for speed
        config['early_stopping_patience'] = 3
        
        # Set output directory
        config['output_dir'] = os.path.join(base_config['output_dir'], f"ablation_{config_mods['name']}")
        
        print(f"\n{'='*50}")
        print(f"Ablation Study: {config_mods['name']}")
        print(f"{'='*50}")
        
        try:
            # Run simplified training
            fold_results, _, _ = train_with_cross_validation(config)
            
            # Extract results
            avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
            avg_precision = np.mean([res['precision'] for res in fold_results])
            avg_recall = np.mean([res['recall'] for res in fold_results])
            avg_f1 = np.mean([res['f1'] for res in fold_results])
            
            result = {
                'config_name': config_mods['name'],
                'accuracy': float(avg_accuracy),
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1': float(avg_f1),
                'config': {k: (str(v) if isinstance(v, torch.device) else v) 
                         for k, v in config_mods.items() if k != 'name'}
            }
            
            ablation_results.append(result)
            
        except Exception as e:
            print(f"Error in configuration {config_mods['name']}: {str(e)}")
            continue
    
    # Save ablation results
    with open(os.path.join(base_config['output_dir'], 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=4)
    
    # Create summary visualizations
    if ablation_results:
        # Convert to DataFrame for easier visualization
        results_df = pd.DataFrame(ablation_results)
        
        # Sort by accuracy
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        # Plot accuracies
        plt.figure(figsize=(12, 8))
        sns.barplot(x='config_name', y='accuracy', data=results_df)
        plt.title('Ablation Study - Accuracy Comparison')
        plt.xlabel('Configuration')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(base_config['output_dir'], 'ablation_accuracy.png'))
        plt.close()
        
        # Plot all metrics
        plt.figure(figsize=(14, 10))
        
        # Reshape for seaborn
        plot_df = pd.melt(results_df, 
                          id_vars=['config_name'], 
                          value_vars=['accuracy', 'precision', 'recall', 'f1'],
                          var_name='Metric', value_name='Value')
        
        sns.barplot(x='config_name', y='Value', hue='Metric', data=plot_df)
        plt.title('Ablation Study - All Metrics')
        plt.xlabel('Configuration')
        plt.ylabel('Value')
        plt.legend(title='Metric')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(base_config['output_dir'], 'ablation_all_metrics.png'))
        plt.close()
        
        # Write ablation study report
        with open(os.path.join(base_config['output_dir'], 'ablation_study_report.md'), 'w') as f:
            f.write('# Ablation Study Report\n\n')
            
            f.write('## Performance Summary\n')
            f.write(results_df[['config_name', 'accuracy', 'precision', 'recall', 'f1']].to_markdown(index=False) + '\n\n')
            
            f.write('## Best Configuration\n')
            best_config = results_df.iloc[0]
            f.write(f"The best configuration was **{best_config['config_name']}** with an accuracy of {best_config['accuracy']:.4f}\n\n")
            
            f.write('Configuration details:\n```\n')
            f.write(json.dumps(best_config['config'], indent=2) + '\n```\n\n')
            
            # Group by different parameters to see their effect
            f.write('## Parameter Impact Analysis\n\n')
            
            # Analyze model types
            model_df = results_df[results_df['config_name'].str.contains('r3d|mc3|r2plus1d|i3d|slowfast|x3d')]
            if len(model_df) > 1:
                f.write('### Model Architecture\n')
                f.write(model_df[['config_name', 'accuracy']].to_markdown(index=False) + '\n\n')
                
                best_model = model_df.iloc[model_df['accuracy'].idxmax()]
                f.write(f"The best model architecture was **{best_model['config_name']}** with accuracy {best_model['accuracy']:.4f}\n\n")
            
            # Analyze frame counts
            frames_df = results_df[results_df['config_name'].str.contains('frames_')]
            if len(frames_df) > 1:
                f.write('### Number of Frames\n')
                f.write(frames_df[['config_name', 'accuracy']].to_markdown(index=False) + '\n\n')
                
                best_frames = frames_df.iloc[frames_df['accuracy'].idxmax()]
                f.write(f"The best frame count was **{best_frames['config_name']}** with accuracy {best_frames['accuracy']:.4f}\n\n")
            
            # Analyze learning rates
            lr_df = results_df[results_df['config_name'].str.contains('lr_')]
            if len(lr_df) > 1:
                f.write('### Learning Rate\n')
                f.write(lr_df[['config_name', 'accuracy']].to_markdown(index=False) + '\n\n')
                
                best_lr = lr_df.iloc[lr_df['accuracy'].idxmax()]
                f.write(f"The best learning rate was **{best_lr['config_name']}** with accuracy {best_lr['accuracy']:.4f}\n\n")
            
            # Analyze augmentation
            aug_df = results_df[results_df['config_name'].str.contains('aug')]
            if len(aug_df) > 1:
                f.write('### Data Augmentation\n')
                f.write(aug_df[['config_name', 'accuracy']].to_markdown(index=False) + '\n\n')
                
                best_aug = aug_df.iloc[aug_df['accuracy'].idxmax()]
                f.write(f"The best augmentation setting was **{best_aug['config_name']}** with accuracy {best_aug['accuracy']:.4f}\n\n")
            
            # Conclusions
            f.write('## Conclusions and Recommendations\n\n')
            f.write('Based on the ablation study results, we recommend:\n\n')
            
            # Add specific recommendations based on results
            recommended_config = {}
            
            # Find best model type
            if len(model_df) > 1:
                best_model_name = model_df.iloc[model_df['accuracy'].idxmax()]['config_name']
                f.write(f"1. Use the **{best_model_name}** model architecture which performed best.\n")
                recommended_config['model_type'] = model_df.iloc[model_df['accuracy'].idxmax()]['config']['model_type']
            
            # Find best frame count
            if len(frames_df) > 1:
                best_frames_name = frames_df.iloc[frames_df['accuracy'].idxmax()]['config_name']
                f.write(f"2. Sample **{best_frames_name.split('_')[1]}** frames from each video.\n")
                recommended_config['num_frames'] = frames_df.iloc[frames_df['accuracy'].idxmax()]['config']['num_frames']
            
            # Find best learning rate
            if len(lr_df) > 1:
                best_lr_name = lr_df.iloc[lr_df['accuracy'].idxmax()]['config_name']
                f.write(f"3. Use a learning rate of **{best_lr_name.split('_')[1]}**.\n")
                recommended_config['learning_rate'] = lr_df.iloc[lr_df['accuracy'].idxmax()]['config']['learning_rate']
            
            # Write out the recommended configuration
            f.write('\n### Recommended Configuration\n\n')
            f.write('```\n')
            f.write(json.dumps(recommended_config, indent=2) + '\n')
            f.write('```\n')
    
    return ablation_results

def main():
    import argparse
    import shutil
    
    # parser = argparse.ArgumentParser(description='Football Key Moment Classification')
    # parser.add_argument('--data_root', type=str, required=True, help='Path to the data folder')
    # parser.add_argument('--output_dir', type=str, default='./football_results', help='Output directory')
    # parser.add_argument('--model_type', type=str, default='slow_fast', help='Model type: r3d, mc3, r2plus1d, i3d, slow_fast, x3d')
    # parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    # parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    # parser.add_argument('--num_frames', type=int, default=32, help='Number of frames to sample')
    # parser.add_argument('--frame_size', type=int, default=224, help='Frame size (height and width)')
    # parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    # parser.add_argument('--num_folds', type=int, default=5, help='Number of cross-validation folds')
    # parser.add_argument('--ablation', action='store_true', help='Perform ablation study')

    
    # args = parser.parse_args()
    
    # Create configuration from arguments
    config = DEFAULT_CONFIG.copy()
    # config.update({
    #     'data_root': args.data_root,
    #     'output_dir': args.output_dir,
    #     'model_type': args.model_type,
    #     'batch_size': args.batch_size,
    #     'num_epochs': args.num_epochs,
    #     'num_frames': args.num_frames,
    #     'frame_height': args.frame_size,
    #     'frame_width': args.frame_size,
    #     'learning_rate': args.learning_rate,
    #     'num_folds': args.num_folds,
    # })
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    
    # Train model with cross-validation
    fold_results, best_model_path, label_to_idx = train_with_cross_validation(config)
    
    # Perform error analysis
    error_df, extreme_errors = error_analysis(config, best_model_path, label_to_idx, fold_results)
    
    # Perform ablation study if requested
    # if args.ablation:
    #     ablation_results = perform_ablation_study(config)
    
    print(f"\nTraining complete! Results saved to {config['output_dir']}")


if __name__ == "__main__":
    print("Starting execution...")
    main()
