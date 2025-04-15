import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----- Device Setup -----
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# ----- Dataset -----
class FootballClipDataset2D(Dataset):
    def __init__(self, clips_dir, frames_per_clip=16, transform=None, frame_size=(112, 112), cache_frames=True):
        self.clips_dir = clips_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.frame_size = frame_size
        self.cache_frames = cache_frames
        self.frames_cache = {}
        self.classes = [d for d in os.listdir(clips_dir) if os.path.isdir(os.path.join(clips_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(clips_dir, class_name)
            for clip_name in os.listdir(class_dir):
                if clip_name.endswith('.mp4'):
                    self.samples.append((os.path.join(class_dir, clip_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def _extract_frames(self, clip_path):
        if self.cache_frames and clip_path in self.frames_cache:
            return self.frames_cache[clip_path]
        cap = cv2.VideoCapture(clip_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= self.frames_per_clip:
            frame_indices = list(range(total_frames)) + [total_frames-1] * (self.frames_per_clip - total_frames)
        else:
            frame_indices = np.linspace(0, total_frames-1, self.frames_per_clip, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(cv2.resize(frame, self.frame_size), cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        if self.cache_frames:
            self.frames_cache[clip_path] = frames
        return frames

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        frames = self._extract_frames(clip_path)
        clip = torch.stack(frames, dim=0)  # shape: (frames, 3, H, W)
        return clip, label

# ----- Model -----
class FrameWiseResNetClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(FrameWiseResNetClassifier, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Remove last FC
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)           # Merge batch and time
        features = self.base(x)              # (B*T, 512, 1, 1)
        features = features.view(B, T, -1)   # (B, T, 512)
        features = features.mean(dim=1)      # Temporal average
        return self.fc(features)             # (B, num_classes)

# ----- Training -----
def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs=10, device='cpu'):
    model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loaders[phase], desc=phase.capitalize()):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'best_resnet2d_model.pth')

    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    return model

# ----- Evaluation -----
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", conf_matrix)

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# ----- Main -----
def main():
    device = get_device()
    print(f"Using device: {device}")

    clips_dir = "../data/extracted"  # ⬅️ Update this path
    num_classes = 16
    batch_size = 4
    frames_per_clip = 16
    num_epochs = 10
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = FootballClipDataset2D(clips_dir, frames_per_clip, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, temp_ds = torch.utils.data.random_split(dataset, [train_size, val_size + test_size])
    val_ds, test_ds = torch.utils.data.random_split(temp_ds, [val_size, test_size])

    data_loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    base_resnet = models.resnet18(pretrained=True)
    model = FrameWiseResNetClassifier(base_resnet, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    trained_model = train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs, device)
    evaluate_model(trained_model, data_loaders['test'], device)

if __name__ == "__main__":
    main()
