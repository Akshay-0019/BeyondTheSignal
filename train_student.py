import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_dataloaders
from src.models.student import get_student_model
from src.models.teacher import get_teacher_model
from src.distillation import DistillationLoss
from src.utils import get_device, compute_metrics


def train_student(
    mode='distilled_noise',
    epochs=20,
    batch_size=32,
    lr=1e-3,
    teacher_checkpoint='checkpoints/teacher_best.pth',
    checkpoint_dir='checkpoints',
    temperature=4.0,
    alpha=0.5
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = get_device()
    use_noise = (mode == 'distilled_noise')
    use_distillation = ('distilled' in mode)

    print(f"\n==================================================")
    print(f"[Student Training] Mode: {mode.upper()} | Device: {device}")
    print(f"Dynamic Noise Augmentation: {use_noise} | Teacher Distillation: {use_distillation}")
    print(f"==================================================")

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_dir='./data/eurosat_2000',
        batch_size=batch_size,
        use_noise_augmentation=use_noise,
        return_clean=use_distillation
    )

    student = get_student_model(num_classes=10).to(device)

    # Load frozen teacher if distillation is enabled
    teacher = None
    if use_distillation:
        if not os.path.exists(teacher_checkpoint):
            raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint}. Train teacher first!")
        print(f"[Student] Loading trained Teacher weights from {teacher_checkpoint}...")
        teacher = get_teacher_model(num_classes=10, pretrained=False).to(device)
        teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    distill_criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for noise robustness
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    save_path = os.path.join(checkpoint_dir, f'student_{mode}.pth')

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for batch_data in train_loader:
            if use_distillation and len(batch_data) == 3:
                noisy_images, clean_images, labels = batch_data
                noisy_images = noisy_images.to(device)
                clean_images = clean_images.to(device)
            else:
                noisy_images, labels = batch_data
                noisy_images = noisy_images.to(device)
                clean_images = noisy_images

            labels = labels.to(device)

            optimizer.zero_grad()
            student_logits = student(noisy_images)

            if use_distillation and teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(clean_images)
                loss, _, _ = distill_criterion(student_logits, teacher_logits, labels)
            else:
                loss = ce_criterion(student_logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * noisy_images.size(0)
            preds = torch.argmax(student_logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.detach().cpu().numpy())

        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(train_targets, train_preds)

        # Validation
        student.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = student(images)
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
            torch.save(student.state_dict(), save_path)
            print(f"  --> Saved new best {mode} student (Val F1: {best_val_f1:.4f})")

    elapsed = time.time() - start_time
    print(f"[Student: {mode}] Finished in {elapsed:.1f}s. Best Val F1: {best_val_f1:.4f} saved to {save_path}")
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MicroSatNet Student Model")
    parser.add_argument('--mode', type=str, default='distilled_noise',
                        choices=['scratch', 'distilled_clean', 'distilled_noise'],
                        help="Training mode: scratch, distilled_clean, or distilled_noise")
    parser.add_argument('--epochs', type=int, default=18, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--temperature', type=float, default=4.0, help="Distillation temperature")
    parser.add_argument('--alpha', type=float, default=0.3, help="Loss balance factor (alpha * CE + (1-alpha) * KL)")
    args = parser.parse_args()

    train_student(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha
    )
