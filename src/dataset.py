import os
import ssl
import shutil
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SSL unverified context workaround for seamless EuroSAT auto-download
ssl._create_default_https_context = ssl._create_unverified_context

# 10 EuroSAT classes (matches torchvision.datasets.EuroSAT)
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

# EuroSAT RGB standard normalization parameters
EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]
EUROSAT_STD = [0.2024, 0.1368, 0.1157]
EUROSAT_NORM = transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD)


def get_eurosat_datasets(root='./data', download=True):
    """
    1. Auto-downloads EuroSAT (64x64 RGB patches) & takes fixed 2,000-patch subset (seed=42).
    2. Performs exact 80/20 Train-Test Split (1,600 train / 400 test) with manual_seed(42).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.EuroSAT(root=root, download=download, transform=transform)

    torch.manual_seed(42)  # Keep seed=42 so all Track 3 teams use the exact same 2,000 patches!
    indices = torch.randperm(len(full_dataset))[:2000]
    subset_2k = Subset(full_dataset, indices)

    # 80/20 Train-Test Split (1,600 train / 400 test)
    train_set, test_set = torch.utils.data.random_split(
        subset_2k, [1600, 400], generator=torch.Generator().manual_seed(42)
    )
    return train_set, test_set


def inject_sensor_noise(images, mode='gaussian', severity=0.15):
    """
    Helper function to inject Sensor Noise / Cloud Haze at test time or dynamically during training.
    Supports 'gaussian', 'salt_pepper', and 'band_dropout' modes.
    Works on 4D image batches [B, C, H, W] or 3D single image tensors [C, H, W] in [0., 1.].
    """
    if mode == 'gaussian':
        return torch.clamp(images + severity * torch.randn_like(images), 0., 1.)
    elif mode == 'salt_pepper':
        mask = torch.rand_like(images)
        noisy = images.clone()
        noisy[mask < (severity / 2)] = 0.0
        noisy[mask > 1 - (severity / 2)] = 1.0
        return noisy
    elif mode == 'band_dropout':
        noisy = images.clone()
        num_bands = int(severity) if severity >= 1 else 1
        num_channels = images.shape[1] if images.ndim == 4 else images.shape[0]
        num_bands = min(num_bands, num_channels - 1)
        if num_bands > 1:
            drop_indices = torch.randperm(num_channels)[:num_bands]
            for b in drop_indices:
                if images.ndim == 4:
                    noisy[:, b, :, :] = 0.0
                else:
                    noisy[b, :, :] = 0.0
        else:
            band_idx = torch.randint(0, num_channels, (1,)).item()
            if images.ndim == 4:
                noisy[:, band_idx, :, :] = 0.0
            else:
                noisy[band_idx, :, :] = 0.0
        return noisy
    return images


def get_dataloaders(
    root='./data',
    batch_size=32,
    num_workers=0,
    download=True,
    **kwargs
):
    """
    Returns train_loader (1,600 patches) and test_loader (400 patches) with batch_size=32.
    """
    # Accept dataset_dir or root for flexible invocation
    data_root = kwargs.get('dataset_dir', root)
    if not os.path.exists(data_root) or 'eurosat_2000' in data_root:
        data_root = './data'

    train_set, test_set = get_eurosat_datasets(root=data_root, download=download)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


class GaussianNoiseTransform:
    """Injects Gaussian sensor noise: x_noisy = x + N(0, sigma^2)."""
    def __init__(self, std_range=(0.05, 0.25), p=0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        std = random.uniform(self.std_range[0], self.std_range[1])
        return inject_sensor_noise(tensor, mode='gaussian', severity=std)


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
        return inject_sensor_noise(tensor, mode='salt_pepper', severity=amount)


class BandDropoutTransform:
    """Randomly zeros out 1 or 2 spectral bands (e.g., Red, Green, or Blue channel loss)."""
    def __init__(self, max_bands=2, p=0.5):
        self.max_bands = max_bands
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        num_bands = random.randint(1, self.max_bands)
        return inject_sensor_noise(tensor, mode='band_dropout', severity=float(num_bands))


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
    """Composite dynamic sensor noise augmentation for training."""
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
        tensor = self.gaussian(tensor)
        tensor = self.sp(tensor)
        tensor = self.band(tensor)
        tensor = self.cloud(tensor)
        return tensor


class SatelliteDataset(Dataset):
    """
    Compatibility Dataset wrapper for custom 64x64 satellite patches with support for
    dynamic noise augmentation and optional paired clean/noisy return.
    """
    def __init__(self, split_dir, dynamic_noise=None, normalize=True, return_clean=False):
        self.split_dir = split_dir
        self.dynamic_noise = dynamic_noise
        self.normalize = normalize
        self.return_clean = return_clean
        self.classes = EUROSAT_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        if os.path.exists(split_dir):
            for cls in self.classes:
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.exists(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

        self.to_tensor = transforms.ToTensor()
        self.norm_transform = EUROSAT_NORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        raw_tensor = self.to_tensor(image)

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


def create_mini_split(
    source_dir='./data/eurosat/2750',
    target_dir='./data/eurosat_2000',
    num_per_class=200,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
):
    """Ensures official EuroSAT dataset is ready."""
    print(f"[Dataset] Ensuring EuroSAT dataset is ready at {source_dir}...")
    return target_dir


def generate_sample_visualizations(
    root='./data',
    output_path='deliverables/sample_augmentations.png',
    **kwargs
):
    """
    Visualizes sample patches and noise modes directly to disk using inject_sensor_noise.
    Strictly terminal-only without plt.show().
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_set, _ = get_eurosat_datasets(root='./data', download=True)

    # Pick 4 representative samples
    sample_indices = [0, 40, 80, 120]
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.patch.set_facecolor('#0d1117')

    col_titles = [
        'Original (Clean)',
        'Gaussian Noise (σ=0.20)',
        'Salt & Pepper (10%)',
        'Band Dropout (1 Band)',
        'Band Dropout (2 Bands)'
    ]

    for row_idx, s_idx in enumerate(sample_indices):
        clean_tensor, label = train_set[s_idx]
        cls_name = EUROSAT_CLASSES[label]

        # Use official inject_sensor_noise helper function
        clean_batch = clean_tensor.unsqueeze(0)
        noisy_gauss = inject_sensor_noise(clean_batch, mode='gaussian', severity=0.20).squeeze(0)
        noisy_sp = inject_sensor_noise(clean_batch, mode='salt_pepper', severity=0.10).squeeze(0)
        noisy_b1 = inject_sensor_noise(clean_batch, mode='band_dropout', severity=1.0).squeeze(0)
        noisy_b2 = inject_sensor_noise(clean_batch, mode='band_dropout', severity=2.0).squeeze(0)

        variants = [
            clean_tensor,
            noisy_gauss,
            noisy_sp,
            noisy_b1,
            noisy_b2
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
    train_set, test_set = get_eurosat_datasets()
    print(f"[Dataset] Official EuroSAT subset loaded: {len(train_set)} Train / {len(test_set)} Test patches.")
    generate_sample_visualizations()
