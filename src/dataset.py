import os
import shutil
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 10 EuroSAT classes
EUROSAT_CLASSES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]


def create_mini_split(
    source_dir='./data/eurosat/2750',
    target_dir='./data/eurosat_2000',
    num_per_class=200,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
):
    """
    Creates a balanced 2,000-patch mini-split from the EuroSAT dataset (64x64 patches).
    200 patches per class * 10 classes = 2,000 patches.
    Split: 1,400 train, 300 val, 300 test.
    """
    random.seed(seed)
    np.random.seed(seed)

    if os.path.exists(target_dir):
        # Check if already populated
        train_count = sum(len(files) for _, _, files in os.walk(os.path.join(target_dir, 'train')))
        test_count = sum(len(files) for _, _, files in os.walk(os.path.join(target_dir, 'test')))
        val_count = sum(len(files) for _, _, files in os.walk(os.path.join(target_dir, 'val')))
        if train_count + val_count + test_count == 2000:
            print(f"[Dataset] Mini-split already prepared at {target_dir} (Train: {train_count}, Val: {val_count}, Test: {test_count})")
            return target_dir

    print(f"[Dataset] Creating balanced 2,000-patch mini-split at {target_dir}...")
    for split in ['train', 'val', 'test']:
        for cls in EUROSAT_CLASSES:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

    n_train = int(num_per_class * train_ratio)  # 140
    n_val = int(num_per_class * val_ratio)      # 30
    n_test = num_per_class - n_train - n_val    # 30

    for cls in EUROSAT_CLASSES:
        cls_src = os.path.join(source_dir, cls)
        if not os.path.exists(cls_src):
            raise FileNotFoundError(f"Class directory not found: {cls_src}")

        all_files = sorted([f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        random.shuffle(all_files)
        selected = all_files[:num_per_class]

        train_files = selected[:n_train]
        val_files = selected[n_train:n_train + n_val]
        test_files = selected[n_train + n_val:]

        for fname in train_files:
            shutil.copy2(os.path.join(cls_src, fname), os.path.join(target_dir, 'train', cls, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(cls_src, fname), os.path.join(target_dir, 'val', cls, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(cls_src, fname), os.path.join(target_dir, 'test', cls, fname))

    print(f"[Dataset] Mini-split created: 1,400 Train, 300 Val, 300 Test (Total: 2,000 patches).")
    return target_dir


class GaussianNoiseTransform:
    """Injects Gaussian sensor noise: x_noisy = x + N(0, sigma^2)."""
    def __init__(self, std_range=(0.05, 0.25), p=0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class SaltAndPepperNoiseTransform:
    """Injects impulse sensor defects (salt-and-pepper noise)."""
    def __init__(self, amount_range=(0.02, 0.15), salt_vs_pepper=0.5, p=0.5):
        self.amount_range = amount_range
        self.salt_vs_pepper = salt_vs_pepper
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        amount = random.uniform(self.amount_range[0], self.amount_range[1])
        noisy = tensor.clone()
        c, h, w = tensor.shape
        num_pixels = int(amount * h * w)

        # Salt (defective white pixels)
        num_salt = int(num_pixels * self.salt_vs_pepper)
        if num_salt > 0:
            coords_y = torch.randint(0, h, (num_salt,))
            coords_x = torch.randint(0, w, (num_salt,))
            noisy[:, coords_y, coords_x] = 1.0

        # Pepper (defective dead/black pixels)
        num_pepper = num_pixels - num_salt
        if num_pepper > 0:
            coords_y = torch.randint(0, h, (num_pepper,))
            coords_x = torch.randint(0, w, (num_pepper,))
            noisy[:, coords_y, coords_x] = 0.0

        return noisy


class BandDropoutTransform:
    """Randomly zeros out 1 or 2 spectral bands (e.g., Red, Green, or Blue channel loss)."""
    def __init__(self, max_bands=2, p=0.5):
        self.max_bands = max_bands
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        noisy = tensor.clone()
        num_bands_to_drop = random.randint(1, min(self.max_bands, tensor.shape[0] - 1))
        drop_indices = random.sample(range(tensor.shape[0]), num_bands_to_drop)
        for idx in drop_indices:
            noisy[idx, :, :] = 0.0
        return noisy


class CloudOcclusionTransform:
    """Simulates semi-transparent or dense cloud patch occlusion."""
    def __init__(self, coverage_range=(0.2, 0.6), alpha_range=(0.4, 0.85), p=0.3):
        self.coverage_range = coverage_range
        self.alpha_range = alpha_range
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        c, h, w = tensor.shape
        coverage = random.uniform(self.coverage_range[0], self.coverage_range[1])
        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])

        # Generate Gaussian cloud blob
        center_x = random.uniform(0.2, 0.8) * w
        center_y = random.uniform(0.2, 0.8) * h
        sigma = (coverage * min(h, w)) / 1.5

        y_coords = torch.arange(h).float().unsqueeze(1).repeat(1, w)
        x_coords = torch.arange(w).float().unsqueeze(0).repeat(h, 1)
        dist_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        cloud_mask = torch.exp(-dist_sq / (2 * (sigma ** 2 + 1e-5))).unsqueeze(0).repeat(c, 1, 1)

        cloud_color = 0.95 + 0.05 * torch.rand_like(tensor)
        blended = (1.0 - alpha * cloud_mask) * tensor + (alpha * cloud_mask) * cloud_color
        return torch.clamp(blended, 0.0, 1.0)


class DynamicNoiseAugmentation:
    """
    On-the-fly composite dynamic sensor noise augmentation for training.
    Injects aggressive Gaussian noise, Salt-and-Pepper, Band-dropout, and Cloud occlusion.
    Probabilities and severity ranges are tuned for maximum noise robustness
    to harden the model against realistic CubeSat sensor degradation.
    """
    def __init__(
        self,
        gaussian_p=0.55,
        sp_p=0.45,
        band_p=0.35,
        cloud_p=0.30,
        gaussian_std=(0.05, 0.25),
        sp_amount=(0.02, 0.15)
    ):
        self.gaussian = GaussianNoiseTransform(std_range=gaussian_std, p=gaussian_p)
        self.sp = SaltAndPepperNoiseTransform(amount_range=sp_amount, p=sp_p)
        self.band = BandDropoutTransform(max_bands=2, p=band_p)
        self.cloud = CloudOcclusionTransform(coverage_range=(0.20, 0.55), alpha_range=(0.3, 0.75), p=cloud_p)

    def __call__(self, tensor):
        # Stochastic application order: each transform fires independently
        # with its own probability, creating a combinatorial noise space
        tensor = self.gaussian(tensor)
        tensor = self.sp(tensor)
        tensor = self.band(tensor)
        tensor = self.cloud(tensor)
        return tensor


class SatelliteDataset(Dataset):
    """
    Custom Dataset for 64x64 satellite patches with support for
    on-the-fly dynamic noise augmentation and optional paired clean/noisy return.
    """
    def __init__(self, split_dir, dynamic_noise=None, normalize=True, return_clean=False):
        self.split_dir = split_dir
        self.dynamic_noise = dynamic_noise
        self.normalize = normalize
        self.return_clean = return_clean
        self.classes = EUROSAT_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

        self.to_tensor = transforms.ToTensor()
        # EuroSAT RGB standard normalization
        self.norm_transform = transforms.Normalize(
            mean=[0.3444, 0.3803, 0.4078],
            std=[0.2024, 0.1368, 0.1157]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        raw_tensor = self.to_tensor(image)  # [3, 64, 64] in [0, 1]

        if self.dynamic_noise is not None:
            noisy_tensor = self.dynamic_noise(raw_tensor.clone())
        else:
            noisy_tensor = raw_tensor

        if self.normalize:
            noisy_norm = self.norm_transform(noisy_tensor)
            if self.return_clean:
                clean_norm = self.norm_transform(raw_tensor)
                return noisy_norm, clean_norm, label
            return noisy_norm, label

        if self.return_clean:
            return noisy_tensor, raw_tensor, label
        return noisy_tensor, label


def get_dataloaders(
    dataset_dir='./data/eurosat_2000',
    batch_size=32,
    use_noise_augmentation=True,
    return_clean=False,
    num_workers=0
):
    """
    Returns train, val, and test DataLoaders.
    """
    train_noise = DynamicNoiseAugmentation() if use_noise_augmentation else None

    train_ds = SatelliteDataset(
        os.path.join(dataset_dir, 'train'),
        dynamic_noise=train_noise,
        return_clean=return_clean
    )
    val_ds = SatelliteDataset(os.path.join(dataset_dir, 'val'), dynamic_noise=None, return_clean=False)
    test_ds = SatelliteDataset(os.path.join(dataset_dir, 'test'), dynamic_noise=None, return_clean=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def generate_sample_visualizations(
    dataset_dir='./data/eurosat_2000',
    output_path='deliverables/sample_augmentations.png'
):
    """
    Strictly terminal-only visualization saving sample patches and noise types.
    Does NOT invoke plt.show().
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_ds = SatelliteDataset(os.path.join(dataset_dir, 'train'), dynamic_noise=None, normalize=False)

    # Pick 4 representative samples
    sample_indices = [0, 140, 280, 420]
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.patch.set_facecolor('#0d1117')

    noise_gaussian = GaussianNoiseTransform(std_range=(0.20, 0.20), p=1.0)
    noise_sp = SaltAndPepperNoiseTransform(amount_range=(0.10, 0.10), p=1.0)
    noise_band = BandDropoutTransform(max_bands=1, p=1.0)
    noise_cloud = CloudOcclusionTransform(coverage_range=(0.5, 0.5), alpha_range=(0.7, 0.7), p=1.0)

    col_titles = ['Original (Clean)', 'Gaussian Noise (σ=0.20)', 'Salt & Pepper (10%)', 'Band Dropout (Red/Green)', 'Cloud Occlusion']

    for row_idx, s_idx in enumerate(sample_indices):
        clean_tensor, label = clean_ds[s_idx]
        cls_name = EUROSAT_CLASSES[label]

        variants = [
            clean_tensor,
            noise_gaussian(clean_tensor),
            noise_sp(clean_tensor),
            noise_band(clean_tensor),
            noise_cloud(clean_tensor)
        ]

        for col_idx, var_tensor in enumerate(variants):
            ax = axes[row_idx, col_idx]
            img_np = var_tensor.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            ax.imshow(img_np)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], color='white', fontsize=11, fontweight='bold', pad=10)
            if col_idx == 0:
                ax.text(-12, 32, cls_name, color='#58a6ff', fontsize=11, fontweight='bold',
                        va='center', ha='right', rotation=0)

    fig.suptitle('Phase 1: Dynamic Sensor Noise Augmentation Pipeline (64×64 Patches)',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.12, 0.02, 0.98, 0.95])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Dataset] Sample augmentations successfully written to {output_path}")


if __name__ == '__main__':
    create_mini_split()
    generate_sample_visualizations()
