import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import time

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

class HoyaGuardDataset(Dataset):
    def __init__(self, hoya_paths, neg_paths, transform=None):
        self.samples = []
        for p in hoya_paths:
            self.samples.append((p, 1))
        for p in neg_paths:
            self.samples.append((p, 0))
        random.shuffle(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
        return img, label

class MiniCNNGuard(nn.Module):
    """Standalone Lightweight Mini-CNN Guard Architecture."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Binary output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)

def get_all_hoya_paths(root_dir):
    paths = []
    if os.path.exists(root_dir):
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(root, file))
    return paths

def get_all_neg_paths(dir_pool, dir_leaves):
    paths = []
    for d in [dir_pool, dir_leaves]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        paths.append(os.path.join(root, file))
    return paths

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"[*] Training Standalone Mini-CNN Guard")
    print(f"    Device: {device}")
    print("=" * 65)

    # 1. Load Paths
    hoya_dir = r"D:\FILE AND TASK\MBKM\Dataset MBKM\data_real_KP"
    neg_pool_dir = r"D:\FILE AND TASK\MBKM\Dataset MBKM\Dataset Non-Hoya\negative_pool"
    neg_leaves_dir = r"D:\FILE AND TASK\MBKM\Dataset MBKM\Dataset Non-Hoya\negative_leaves"

    hoya_paths = get_all_hoya_paths(hoya_dir)
    neg_paths = get_all_neg_paths(neg_pool_dir, neg_leaves_dir)

    print(f"[DATASET SUMMARY]")
    print(f"   [+] Positive (Hoya Leaves)   : {len(hoya_paths)} gambar")
    print(f"   [-] Negative Pool (Random)  : {len(get_all_hoya_paths(neg_pool_dir))} gambar")
    print(f"   [-] Negative Leaves (Lookalikes): {len(get_all_hoya_paths(neg_leaves_dir))} gambar")
    print(f"   [-] TOTAL NEGATIVE COMBINED : {len(neg_paths)} gambar")

    # 2. Train / Val Split
    random.seed(SEED)
    random.shuffle(hoya_paths)
    random.shuffle(neg_paths)

    n_hoya_val = int(len(hoya_paths) * 0.2)
    n_neg_val  = int(len(neg_paths) * 0.2)

    val_hoya  = hoya_paths[:n_hoya_val]
    train_hoya = hoya_paths[n_hoya_val:]

    val_neg  = neg_paths[:n_neg_val]
    train_neg = neg_paths[n_neg_val:]

    print(f"\n[SPLIT SUMMARY]")
    print(f"   Train: {len(train_hoya)} Hoya + {len(train_neg)} Negative = {len(train_hoya)+len(train_neg)}")
    print(f"   Val  : {len(val_hoya)} Hoya + {len(val_neg)} Negative = {len(val_hoya)+len(val_neg)}")

    # 3. Transforms
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    train_dataset = HoyaGuardDataset(train_hoya, train_neg, transform=train_transform)
    val_dataset   = HoyaGuardDataset(val_hoya, val_neg, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 4. Model & Optimizer
    model = MiniCNNGuard(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # 5. Training Loop
    best_acc = 0.0
    models_dir = r"D:\FILE AND TASK\MBKM\Hoya-Leaf-Disease-Detection-main\models"
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "MobileNetV3 Small - Guard Final Model.pth")

    print(f"\n[*] Starting Training (5 Epochs)...")
    for epoch in range(5):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.float().to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d}/05 | Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.2f}% | Time: {elapsed:.1f}s", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] New best model saved to {save_path} (Acc: {val_acc*100:.2f}%)", flush=True)

    print("\n" + "=" * 65)
    print(f"[DONE] Training finished! Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"       Model path: {save_path}")
    print("=" * 65)

if __name__ == '__main__':
    main()
