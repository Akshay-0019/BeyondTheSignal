import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.models.teacher import get_teacher_model
from src.utils import get_device, compute_metrics


def train_teacher(epochs=15, batch_size=32, lr=3e-4, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = get_device()
    print(f"[Teacher] Training ResNet-50 Teacher on device: {device}")

    # Clean data loader for teacher (to establish strong clean feature representations)
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_dir='./data/eurosat_2000',
        batch_size=batch_size,
        use_noise_augmentation=False
    )

    model = get_teacher_model(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_path = os.path.join(checkpoint_dir, 'teacher_best.pth')

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.detach().cpu().numpy())

        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(train_targets, train_preds)

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_metrics = compute_metrics(val_targets, val_preds)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new best teacher model (Val F1: {best_val_f1:.4f})")

    elapsed = time.time() - start_time
    print(f"[Teacher] Training completed in {elapsed:.1f}s. Best Val F1: {best_val_f1:.4f}")
    return best_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResNet-50 Teacher Model")
    parser.add_argument('--epochs', type=int, default=12, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train_teacher(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
