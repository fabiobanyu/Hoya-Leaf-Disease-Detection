import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys

# Import functions and classes from Mobile V3.py
import importlib.util
script_path = r"D:\FILE AND TASK\MBKM\Hoya-Leaf-Disease-Detection-main\Kode_Model\Model Perbandingan\Mobile V3.py"
spec = importlib.util.spec_from_file_location("mobile_v3", script_path)
mobile_v3 = importlib.util.module_from_spec(spec)
sys.modules["mobile_v3"] = mobile_v3
spec.loader.exec_module(mobile_v3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Re-create the exact same validation dataset setup
hoya_dir = r"D:\FILE AND TASK\MBKM\Dataset MBKM\data_real_KP"
neg_pool_dir = r"D:\FILE AND TASK\MBKM\Dataset MBKM\Dataset Non-Hoya\negative_pool"
neg_leaves_dir = r"D:\FILE AND TASK\MBKM\Dataset MBKM\Dataset Non-Hoya\negative_leaves"

hoya_paths = mobile_v3.get_all_hoya_paths(hoya_dir)
neg_paths = mobile_v3.get_all_neg_paths(neg_pool_dir, neg_leaves_dir)

SEED = 42
random.seed(SEED)
random.shuffle(hoya_paths)
random.shuffle(neg_paths)

n_hoya_val = int(len(hoya_paths) * 0.2)
n_neg_val  = int(len(neg_paths) * 0.2)

val_hoya  = hoya_paths[:n_hoya_val]
val_neg  = neg_paths[:n_neg_val]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Label 1 for Hoya, 0 for Negative as defined in HoyaGuardDataset
val_dataset = mobile_v3.HoyaGuardDataset(val_hoya, val_neg, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

print("Loading model...")
model = mobile_v3.MiniCNNGuard(pretrained=False).to(device)
model_path = r"D:\FILE AND TASK\MBKM\Hoya-Leaf-Disease-Detection-main\models\MobileNetV3 Small - Guard Final Model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Evaluating...")
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\n=== CONFUSION MATRIX ===")
print("True Negative (Non-Hoya):", cm[0][0])
print("False Positive (Non-Hoya as Hoya):", cm[0][1])
print("False Negative (Hoya as Non-Hoya):", cm[1][0])
print("True Positive (Hoya):", cm[1][1])

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=["Non-Hoya", "Hoya"]))

# Plot and save
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Non-Hoya", "Hoya"], 
            yticklabels=["Non-Hoya", "Hoya"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - MobileNetV3 Small (Guardian)')
plt.tight_layout()
plt.savefig(r'Figures BAB IV\Confusion Matrix Guardian.png')
print("\n[SUCCESS] Evaluation complete. Image saved.")
