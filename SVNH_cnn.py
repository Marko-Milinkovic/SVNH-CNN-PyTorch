# Simple CNN for Street View House Numbers (SVHN) classification (10 digits)

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
PATH = "svhn_cnn_best.pth"
BATCH_SIZE = 128          # reduce to 64 if RAM is tight
VAL_RATIO = 0.1           # 10% of train for validation
MAX_EPOCHS = 10           # small cap; early stopping will stop earlier
PATIENCE = 2              # early stop if no val acc improvement for N epochs
SEED = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD  = (0.1980, 0.2010, 0.1970)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(SVHN_MEAN, SVHN_STD),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(SVHN_MEAN, SVHN_STD),
])

# ---------------------------
# Datasets
# ---------------------------
# SVHN labels are in {0..9}
train_full = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=train_tf)
test_set   = torchvision.datasets.SVHN(root="./data", split="test",  download=True, transform=test_tf)

# Train/val split
val_size = int(len(train_full) * VAL_RATIO)
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Model
# ---------------------------
class SVHNCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: 3 -> 32 -> 32, pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        # Block 2: 32 -> 64 -> 64, pool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        # Block 3: 64 -> 128, pool
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(2, 2)
        self.drop2d= nn.Dropout2d(0.20)

        # After three pools on 32x32 -> 4x4 with 128 channels
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.bnfc1 = nn.BatchNorm1d(256)
        self.drop  = nn.Dropout(0.3)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        # 32x32
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)          # -> 16x16
        x = self.drop2d(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)          # -> 8x8
        x = self.drop2d(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)          # -> 4x4
        x = self.drop2d(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bnfc1(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)
        return x

net = SVHNCNN().to(device)

# ---------------------------
# Optimizer / Scheduler / Train helpers
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
# Use val_loss for ReduceLROnPlateau (compatible with older torch without verbose)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def run_epoch(model, loader, train: bool):
    model.train(train)
    total_loss, total_correct, total_batches, total_samples = 0.0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_batches += 1
        total_samples += labels.size(0)

    return total_loss / total_batches, total_correct / total_samples

# ---------------------------
# Training loop with early stopping
# ---------------------------
best_val_acc = 0.0
no_improve = 0
hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(net, train_loader, train=True)
    val_loss, val_acc = run_epoch(net, val_loader,   train=False)

    scheduler.step(val_loss)

    hist["train_loss"].append(tr_loss)
    hist["val_loss"].append(val_loss)
    hist["train_acc"].append(tr_acc)
    hist["val_acc"].append(val_acc)

    print(f"[Epoch {epoch:03d}] "
          f"loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% | "
          f"val_loss {val_loss:.4f} val_acc {val_acc*100:5.2f}% | ")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(net.state_dict(), PATH)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

print(f"Best val acc: {best_val_acc*100:.2f}%")

# ---------------------------
# Test
# ---------------------------

net.load_state_dict(torch.load(PATH, map_location=device))
net.eval()
correct, total = 0, 0

correct_images, wrong_images = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        preds = logits.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        correct_images.append(images[preds == labels].cpu())
        wrong_images.append(images[preds != labels].cpu())

print(f"Test accuracy on {len(test_set)} images: {100.0*correct/total:.2f}%")

# ---------------------------
# Misslabeled images examples
# ---------------------------

correct_images = torch.cat(correct_images)
wrong_images = torch.cat(wrong_images)

def show_examples(imgs, title, n=16):
    imgs = imgs[:n] * torch.tensor(SVHN_STD).view(1,3,1,1) + torch.tensor(SVHN_MEAN).view(1,3,1,1)
    imgs = imgs.permute(0,2,3,1).numpy().clip(0,1)
    cols = 8
    plt.figure(figsize=(cols, n//cols))
    for i in range(n):
        plt.subplot(n//cols, cols, i+1)
        plt.imshow(imgs[i])
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

show_examples(correct_images, "Correctly classified examples")
show_examples(wrong_images, "Misclassified examples")

# ---------------------------
# Plots
# ---------------------------
plt.figure()
plt.plot(hist["train_loss"], label="train")
plt.plot(hist["val_loss"], label="val")
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(hist["train_acc"], label="train")
plt.plot(hist["val_acc"], label="val")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.show()