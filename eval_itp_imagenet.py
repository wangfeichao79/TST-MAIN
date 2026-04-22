#!/usr/bin/env python3
"""
ITP (Instance-aware Test Pruning) OOD Detection Evaluation
Paper: ITP: Instance-Aware Test Pruning for OOD Detection (AAAI-25)
ID: ImageNet-1K, OOD: iNaturalist, SUN, Places, Textures(dtd)
Model: MobileNetV2 (pretrained)
"""

import os, sys, csv, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = '/home/lenovo/wfc/LAPS-main'

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])

# =============================================================================
# ITP Hyperparameters (from paper, Table 4: p=30, lambda=1.5 for ImageNet)
# =============================================================================
CRP_PERCENTILE = 30
FTP_THRESHOLD = 1.5
NUM_CLASSES = 1000


# =============================================================================
# Model
# =============================================================================
class MobileNetITP(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        # Store as registered buffers so they move with .to(device)
        self.register_buffer('fc_weight', base.classifier[1].weight.clone())
        self.register_buffer('fc_bias', base.classifier[1].bias.clone())
        self.last_channel = base.last_channel

    def forward_features(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

    def forward_head(self, features):
        return F.linear(features, self.fc_weight, self.fc_bias)

    def forward(self, x):
        features = self.forward_features(x)
        return self.forward_head(features)


# =============================================================================
# Datasets
# =============================================================================
class ImageNetValDataset(Dataset):
    """ImageNet val from imglist + flat val directory."""
    def __init__(self, val_dir, imglist_path, transform):
        self.transform = transform
        self.samples = []
        # imglist format: "imagenet_1k/val/ILSVRC2012_val_00001.JPEG 0"
        prefix = 'imagenet_1k/val/'
        base = val_dir
        count = 0
        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel = parts[0]
                    if rel.startswith(prefix):
                        rel = rel[len(prefix):]
                    path = os.path.join(base, rel)
                    if os.path.exists(path):
                        self.samples.append((path, int(parts[1])))
                        count += 1
        print(f"  ImageNetVal: {count} samples loaded from imglist")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


class ImageNetFlatValDataset(Dataset):
    """ImageNet val from flat directory (50000 images)."""
    def __init__(self, val_dir, transform):
        self.transform = transform
        self.samples = []
        files = sorted([f for f in os.listdir(val_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for f in files:
            self.samples.append(os.path.join(val_dir, f))
        print(f"  ImageNetFlatVal: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        return self.transform(img), 0


class OODDataset(Dataset):
    def __init__(self, root_dir, transform, max_samples=10000):
        self.transform = transform
        self.samples = []
        for root, dirs, files in os.walk(root_dir):
            for n in files:
                if n.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(root, n))
                    if len(self.samples) >= max_samples:
                        break
            if len(self.samples) >= max_samples:
                break
        print(f"  OOD {os.path.basename(root_dir)}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        return self.transform(img), 0


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(id_scores, ood_scores):
    from sklearn.metrics import roc_curve, auc
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    fpr95 = float(fpr[idx_95] * 100)
    auroc = float(auc(fpr, tpr) * 100)
    precision, recall, _ = roc_curve(labels, scores, pos_label=1)
    aupr = float(auc(recall, precision) * 100)
    return fpr95, auroc, aupr


# =============================================================================
# ITP
# =============================================================================
class ITPPruner:
    """
    ITP = CRP + FTP
    Two-pass approach:
      Pass 1: Extract features + predicted labels for ALL val images (50k)
              At the same time accumulate contribution statistics
      Pass 2: Apply ITP scoring using the stored labels
    """
    def __init__(self, model, id_val_dir, imglist_path, device,
                 crp_p=30, ftp_lambda=1.5, num_classes=1000):
        self.device = device
        self.crp_p = crp_p
        self.ftp_lambda = ftp_lambda
        self.num_classes = num_classes
        self.model = model.to(device)

        print("=" * 60)
        print(f"ITP Pruner (CRP p={crp_p}, FTP lambda={ftp_lambda})")
        print("=" * 60)

        # Data
        id_dataset = ImageNetValDataset(id_val_dir, imglist_path, TRANSFORM)
        id_loader = DataLoader(id_dataset, batch_size=64, shuffle=False,
                               num_workers=4, pin_memory=True)
        print(f"  ID val samples: {len(id_dataset)}")

        # Allocate accumulators: contribution per class
        C, D = num_classes, 1280
        sum_contrib = torch.zeros(C, D, device=device)
        sum_sq_contrib = torch.zeros(C, D, device=device)
        class_counts = torch.zeros(C, device=device)
        self.all_features = []
        self.all_labels = []

        print("  [Pass 1] Extracting features & accumulating statistics...")
        t0 = time.time()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(id_loader):
                images = images.to(device)
                labels = labels.to(device)
                features = model.forward_features(images)
                preds = model.forward_head(features)

                # Store for scoring
                self.all_features.append(features.cpu())
                self.all_labels.append(labels.cpu())

                # Accumulate contribution per class: W[j] * h(x)
                W = model.fc_weight  # (C, D) - buffer, already on device
                for c in range(C):
                    mask = (labels == c)
                    if mask.sum() == 0:
                        continue
                    c_features = features[mask]          # (N_c, D)
                    contrib = c_features * W[c].unsqueeze(0)  # (N_c, D)
                    sum_contrib[c] += contrib.sum(dim=0)
                    sum_sq_contrib[c] += (contrib ** 2).sum(dim=0)
                    class_counts[c] += mask.sum().item()

                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"    batch {batch_idx+1}, {elapsed:.1f}s elapsed")

        self.all_features = torch.cat(self.all_features)  # (N_id, D)
        self.all_labels = torch.cat(self.all_labels)       # (N_id,)
        print(f"  [Pass 1] Done in {time.time()-t0:.1f}s")
        print(f"  Feature matrix: {self.all_features.shape}")

        # Compute mu and sigma
        self.mu = torch.zeros(C, D, device=device)
        self.sigma = torch.zeros(C, D, device=device)
        for c in range(C):
            n = int(class_counts[c].item())
            if n > 1:
                self.mu[c] = sum_contrib[c] / n
                var = (sum_sq_contrib[c] - n * self.mu[c]**2) / (n - 1)
                self.sigma[c] = torch.sqrt(torch.clamp(var, min=1e-8))

        # CRP mask
        print("  [CRP] Computing redundancy pruning mask...")
        avg_contrib = self.mu
        all_vals = avg_contrib.flatten()
        k_val = max(1, int(len(all_vals) * crp_p / 100))
        omega_p = torch.kthvalue(all_vals, k=k_val)[0].item()
        self.mcrp_mask = (avg_contrib > omega_p).float()
        pruned_pct = (1 - self.mcrp_mask.mean().item()) * 100
        print(f"  CRP threshold Omega_p={omega_p:.4f}, pruned {pruned_pct:.1f}%")

        W = model.fc_weight  # buffer, on device
        self.W_crp = self.mcrp_mask * W
        self.b_crp = model.fc_bias.clone()
        self.W_orig = W.clone()
        self.b_orig = model.fc_bias.clone()

        # Pre-compute CRP logits for all ID samples
        print("  [CRP] Pre-computing CRP logits for all ID samples...")
        t0 = time.time()
        id_features_gpu = self.all_features.to(device)
        self.crp_logits_id = F.linear(id_features_gpu, self.W_crp.detach(), self.b_crp.detach())
        self.energy_logits_id = F.linear(id_features_gpu, self.W_orig.detach(), self.b_orig.detach())
        self.preds_id = self.crp_logits_id.detach().argmax(dim=1)  # use CRP preds for ITP
        self.all_features_gpu = id_features_gpu.detach()
        print(f"  Done in {time.time()-t0:.1f}s")

        print("  ITP Pruner ready.")

    def energy_score(self, logits):
        return torch.logsumexp(logits, dim=1)

    def get_id_scores(self, mode='itp'):
        """Return ITP OOD scores for all ID samples (pre-computed)."""
        if mode == 'energy':
            return self.energy_score(self.energy_logits_id.detach()).cpu().numpy()
        elif mode == 'crp':
            return self.energy_score(self.crp_logits_id.detach()).cpu().numpy()
        elif mode == 'itp':
            return self.compute_itp_scores_batch(self.all_features_gpu, self.preds_id).cpu().numpy()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_itp_scores_batch(self, features, preds):
        """ITP: CRP + FTP. For efficiency, process batch by batch."""
        B = features.size(0)
        scores = []
        batch_size = 256

        # Detach all internal state to avoid grad graph issues
        W_crp = self.W_crp.detach()
        mu = self.mu.detach()
        sigma = self.sigma.detach()
        b_crp = self.b_crp.detach()
        ftp_lambda = self.ftp_lambda

        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            batch_feat = features[start:end].detach()
            batch_preds = preds[start:end].detach()

            W_pruned = W_crp.clone()
            for i in range(batch_preds.size(0)):
                c = batch_preds[i].item()
                h = batch_feat[i]
                contrib = W_pruned[c] * h
                z = (contrib - mu[c]) / (sigma[c] + 1e-8)
                ftp_mask = 1.0 - (z > ftp_lambda).float()
                W_pruned[c] = W_pruned[c] * ftp_mask

            logits = F.linear(batch_feat, W_pruned, b_crp)
            scores.append(self.energy_score(logits))

        return torch.cat(scores)

    def score_ood(self, ood_loader, mode='itp'):
        """Score OOD dataset."""
        all_scores = []
        all_preds = []

        with torch.no_grad():
            for images, _ in ood_loader:
                images = images.to(self.device)
                features = self.model.forward_features(images)
                preds = self.model.forward_head(features).argmax(dim=1)
                all_preds.append(preds.cpu())

                if mode == 'energy':
                    logits = self.model.forward_head(features)
                    scores = self.energy_score(logits)
                elif mode == 'crp':
                    logits = F.linear(features, self.W_crp, self.b_crp)
                    scores = self.energy_score(logits)
                elif mode == 'itp':
                    scores = self.compute_itp_scores_batch(features, preds)

                all_scores.append(scores.cpu())

        return torch.cat(all_scores).numpy()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("ITP OOD Detection Evaluation")
    print("ID: ImageNet-1K | OOD: iNaturalist, SUN, Places, Textures")
    print(f"Model: MobileNetV2 (pretrained) | Device: {DEVICE}")
    print("=" * 60)

    # Paths
    VAL_DIR = '/home/lenovo/data/images_largescale/imagenet_1k/val'
    IMGLIST = '/home/lenovo/data/benchmark_imglist/imagenet/val_imagenet.txt'
    OOD_DIRS = {
        'iNaturalist': '/home/lenovo/wfc/LAPS-main/datasets/ood_data/iNaturalist/images',
        'SUN': '/home/lenovo/wfc/LAPS-main/datasets/ood_data/sun50/images',
        'Places': '/home/lenovo/wfc/LAPS-main/datasets/ood_data/Places/images',
        'Textures': '/home/lenovo/wfc/LAPS-main/datasets/ood_data/dtd/images',
    }

    # Load model
    print("\n[1/4] Loading MobileNetV2...")
    model = MobileNetITP(pretrained=True).to(DEVICE).eval()
    print(f"  FC: {model.fc_weight.shape}")  # (1000, 1280)

    # Init ITP pruner (this does Pass 1 + CRP)
    print("\n[2/4] Initializing ITP pruner (Pass 1 + CRP)...")
    itp = ITPPruner(
        model=model,
        id_val_dir=VAL_DIR,
        imglist_path=IMGLIST,
        device=DEVICE,
        crp_p=CRP_PERCENTILE,
        ftp_lambda=FTP_THRESHOLD,
        num_classes=NUM_CLASSES,
    )

    # Get ID scores
    print("\n[3/4] Computing ID OOD scores...")
    id_scores_energy = itp.get_id_scores('energy')
    id_scores_crp = itp.get_id_scores('crp')
    id_scores_itp = itp.get_id_scores('itp')
    print(f"  ID scores computed: Energy={len(id_scores_energy)}, CRP={len(id_scores_crp)}, ITP={len(id_scores_itp)}")

    # Evaluate OOD
    print("\n[4/4] OOD detection evaluation...")
    results = []

    for ood_name, ood_dir in OOD_DIRS.items():
        if not os.path.exists(ood_dir):
            print(f"  WARNING: {ood_dir} not found")
            continue

        ood_dataset = OODDataset(ood_dir, TRANSFORM, max_samples=10000)
        if len(ood_dataset) == 0:
            continue

        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False,
                               num_workers=4, pin_memory=True)

        for mode in ['energy', 'crp', 'itp']:
            ood_scores = itp.score_ood(ood_loader, mode=mode)
            fpr95, auroc, aupr = compute_metrics(
                id_scores_energy if mode == 'energy' else
                id_scores_crp if mode == 'crp' else id_scores_itp,
                ood_scores
            )
            key = f"{mode}_results"
            if ood_name not in [r['ood_name'] for r in results]:
                results.append({'ood_name': ood_name})
            for r in results:
                if r['ood_name'] == ood_name:
                    r[f'{mode}_fpr95'] = fpr95
                    r[f'{mode}_auroc'] = auroc
                    r[f'{mode}_aupr'] = aupr

    # Print results
    print("\n" + "=" * 80)
    print(f"{'OOD Dataset':<15} {'Energy FPR95':<14} {'CRP FPR95':<12} {'ITP FPR95':<12} "
          f"{'Energy AUROC':<14} {'CRP AUROC':<12} {'ITP AUROC':<12}")
    print("=" * 80)

    for r in results:
        print(f"{r['ood_name']:<15} "
              f"{r['energy_fpr95']:>12.2f}%  {r['crp_fpr95']:>10.2f}%  {r['itp_fpr95']:>10.2f}%  "
              f"{r['energy_auroc']:>12.2f}%  {r['crp_auroc']:>10.2f}%  {r['itp_auroc']:>10.2f}%")

    # Average
    avg_e_fpr = np.mean([r['energy_fpr95'] for r in results])
    avg_c_fpr = np.mean([r['crp_fpr95'] for r in results])
    avg_i_fpr = np.mean([r['itp_fpr95'] for r in results])
    avg_e_auroc = np.mean([r['energy_auroc'] for r in results])
    avg_c_auroc = np.mean([r['crp_auroc'] for r in results])
    avg_i_auroc = np.mean([r['itp_auroc'] for r in results])

    print("-" * 80)
    print(f"{'Average':<15} "
          f"{avg_e_fpr:>12.2f}%  {avg_c_fpr:>10.2f}%  {avg_i_fpr:>10.2f}%  "
          f"{avg_e_auroc:>12.2f}%  {avg_c_auroc:>10.2f}%  {avg_i_auroc:>10.2f}%")
    print("=" * 80)

    # Save CSV
    output_dir = f'{PROJECT_DIR}/output/ood_scores/imagenet/itp_mobilenet'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'itp_results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'OOD Dataset', 'Method', 'FPR95 (%)', 'AUROC (%)', 'AUPR (%)',
            'ID Samples', 'CRP p (%)', 'FTP lambda'
        ])
        for r in results:
            for mode, fpr_key, auroc_key, aupr_key in [
                ('Energy', 'energy_fpr95', 'energy_auroc', 'energy_aupr'),
                ('CRP',    'crp_fpr95',    'crp_auroc',    'crp_aupr'),
                ('ITP',    'itp_fpr95',    'itp_auroc',    'itp_aupr'),
            ]:
                writer.writerow([
                    r['ood_name'], mode,
                    f'{r[fpr_key]:.2f}', f'{r[auroc_key]:.2f}', f'{r[aupr_key]:.2f}',
                    len(id_scores_energy), CRP_PERCENTILE, FTP_THRESHOLD
                ])
        for mode, avg_fpr, avg_auroc in [
            ('Energy', avg_e_fpr, avg_e_auroc),
            ('CRP',    avg_c_fpr, avg_c_auroc),
            ('ITP',    avg_i_fpr, avg_i_auroc),
        ]:
            writer.writerow(['Average', mode, f'{avg_fpr:.2f}', f'{avg_auroc:.2f}',
                            '', len(id_scores_energy), CRP_PERCENTILE, FTP_THRESHOLD])

    print(f"\nResults saved to: {csv_path}")
    print("Done!")


if __name__ == '__main__':
    main()
