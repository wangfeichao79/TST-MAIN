#!/usr/bin/env python
"""
TSRE Evaluation Script for OOD Detection
Based on paper: TSRE: Channel-Aware Typical Set Refinement for Out-of-Distribution Detection

ID: ImageNet-1K, OOD: iNaturalist, SUN, Places, Textures
Network: MobileNet (pretrained)
"""

from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torchvision
from torchvision import transforms
from util.metrics import compute_traditional_ood, compute_in
from util.args_loader import get_args
from util.data_loader import get_loader_out
from util.model_loader import get_model
from score import get_score

# TSRE Hyperparameters (from paper)
omega = 18   # skewness weight
theta = 1    # discriminability weight
p_percentile = 5  # activity percentile threshold
lam_base = 1.8   # base lambda
a_weight = 0.6   # discriminability balance between similarity and variance


transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImageNetValDataset(torch.utils.data.Dataset):
    """Custom dataset for ImageNet validation set (flat directory)"""

    def __init__(self, anno_file, transform=None):
        self.transform = transform
        self.samples = []
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    class_idx = int(parts[1])
                    full_path = os.path.join('/home/lenovo/data/images_largescale', img_path)
                    self.samples.append((full_path, class_idx))
        print(f"Loaded {len(self.samples)} samples from {anno_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = torchvision.datasets.folder.default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target


def compute_tsre_bounds_vectorized(feature_mean, feature_std, num_classes=1000):
    """
    Compute TSRE bounds using vectorized operations.
    """
    C = feature_mean.shape[0]

    # Generate prototypes: (num_classes, C)
    np.random.seed(42)
    prototypes = feature_mean + feature_std * np.random.randn(num_classes, C) * 0.1

    # Global statistics
    mu_bar = feature_mean.mean()
    sigma_bar = feature_std.mean()

    # === Compute S_k (inter-class similarity) ===
    # Normalize each channel's prototypes
    prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)  # (num_classes, C)
    # Mean of normalized values per channel = approximate cosine similarity
    S_k = prototypes_norm.mean(axis=0)  # (C,)

    # === Compute V_k (inter-class variance) ===
    h_bar = prototypes.mean(axis=0)  # (C,)
    V_k = ((prototypes - h_bar) ** 2).mean(axis=0)  # (C,)

    # === Normalize S_k and V_k ===
    S_k = (S_k - S_k.min()) / (S_k.max() - S_k.min() + 1e-8)
    V_k = (V_k - V_k.min()) / (V_k.max() - V_k.min() + 1e-8)

    # === Compute discriminability D_k ===
    D_k = a_weight * S_k - (1 - a_weight) * V_k  # (C,)

    # === Compute activity A_k ===
    A_raw = np.abs(prototypes).mean(axis=0)  # (C,)
    tau = np.percentile(A_raw, p_percentile)
    A_k = np.where(A_raw >= tau, A_raw, 0.0)  # (C,)

    # === Compute skewness ===
    skew_k = (((prototypes - feature_mean) / (feature_std + 1e-8)) ** 3).mean(axis=0)  # (C,)

    # === Compute adaptive lambda for each channel ===
    lambda_k = lam_base + omega * D_k * (mu_bar - feature_mean + sigma_bar - feature_std) + A_k  # (C,)

    # === Compute bounds ===
    l_k = feature_mean - lambda_k * feature_std - skew_k
    u_k = feature_mean + lambda_k * feature_std - skew_k

    return l_k, u_k, lambda_k, D_k, A_k, skew_k


class TSREWrapper:
    """Wrapper to compute TSRE bounds once and cache them"""

    def __init__(self, args):
        self.args = args
        self.bounds_computed = False

    def setup(self):
        if self.bounds_computed:
            return

        # Load pre-computed channel statistics (convert to float32)
        feature_std = torch.load("feat/mobilenet/imagenet_features_std.pt", map_location='cuda').float().cuda()
        feature_mean = torch.load("feat/mobilenet/imagenet_features_mean.pt", map_location='cuda').float().cuda()

        # Convert to numpy
        fm_np = feature_mean.detach().cpu().float().numpy()
        fs_np = feature_std.detach().cpu().float().numpy()

        # Compute TSRE bounds
        l_k_np, u_k_np, lambda_k_np, D_k_np, A_k_np, skew_k_np = compute_tsre_bounds_vectorized(
            fm_np, fs_np, num_classes=1000
        )

        # Convert bounds to tensors (float32 to match model)
        self.l_k = torch.from_numpy(l_k_np.astype(np.float32)).cuda()
        self.u_k = torch.from_numpy(u_k_np.astype(np.float32)).cuda()

        print(f"TSRE bounds computed. lambda range: [{lambda_k_np.min():.2f}, {lambda_k_np.max():.2f}]")
        print(f"Bounds range: l=[{l_k_np.min():.2f}, {l_k_np.max():.2f}], u=[{u_k_np.min():.2f}, {u_k_np.max():.2f}]")

        self.bounds_computed = True

    def __call__(self, inputs, model):
        self.setup()
        return self.forward_threshold(inputs, model)

    def forward_threshold(self, inputs, model):
        if self.args.model_arch.find('mobilenet') > -1:
            # Extract features
            features = model.forward_features(inputs)

            # Apply TSRE rectification
            features_tsre = torch.clamp(features, min=self.l_k.unsqueeze(0), max=self.u_k.unsqueeze(0))

            # Forward through classifier
            logits = model.forward_head(features_tsre)
        else:
            logits = model(inputs)

        return logits


def main():
    args = get_args()

    # Override arguments for TSRE evaluation
    args.in_dataset = "imagenet"
    args.out_datasets = ["inat", "sun50", "Places", "dtd"]
    args.model_arch = "mobilenet"
    args.name = "mobilenet"
    args.method = "energy"
    args.batch_size = 32
    args.gpu = "0"
    args.base_dir = "output/ood_scores"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # Create TSRE wrapper
    tsre_wrapper = TSREWrapper(args)

    # Evaluation
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    out_datasets = args.out_datasets
    method = args.method
    method_args = {}
    name = "tsre"

    in_save_dir = os.path.join(base_dir, in_dataset, method, name)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    # Create ImageNet validation DataLoader
    anno_file = '/home/lenovo/data/benchmark_imglist/imagenet/val_imagenet.txt'
    val_dataset = ImageNetValDataset(anno_file, transform=transform_test_largescale)
    testloaderIn = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    num_classes = 1000
    method_args['num_classes'] = num_classes

    model = get_model(args, num_classes, load_ckpt=True)

    # Pre-compute TSRE bounds
    print("Pre-computing TSRE bounds...")
    tsre_wrapper.setup()

    t0 = time.time()

    # In-distribution evaluation
    f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
    g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    print("Processing in-distribution images (ImageNet-1K)...")
    N = len(testloaderIn.dataset)
    count = 0
    for j, data in enumerate(testloaderIn):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        curr_batch_size = images.shape[0]

        inputs = images.float()

        with torch.no_grad():
            logits = tsre_wrapper(inputs, model)

            outputs = F.softmax(logits, dim=1)
            outputs = outputs.detach().cpu().numpy()
            preds = np.argmax(outputs, axis=1)
            confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

        scores = get_score(inputs, model, tsre_wrapper, method, method_args, logits=logits)
        for score in scores:
            f1.write("{}\n".format(score))

        count += curr_batch_size
        if (j + 1) % 50 == 0:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

    f1.close()
    g1.close()

    # OOD evaluation
    for out_dataset in out_datasets:
        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        print(f"\nProcessing OOD dataset: {out_dataset}")
        testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader

        t0 = time.time()
        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):
            images, labels = data
            images = images.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():
                logits = tsre_wrapper(inputs, model)

            scores = get_score(inputs, model, tsre_wrapper, method, method_args, logits=logits)
            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            if (j + 1) % 50 == 0:
                print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
                t0 = time.time()

        f2.close()

    # Compute metrics
    print("\nComputing OOD detection metrics...")
    compute_traditional_ood(args.base_dir, args.in_dataset, args.out_datasets, args.method, name)
    compute_in(args.base_dir, args.in_dataset, args.method, name)

    print("\nEvaluation complete! Results saved to:", in_save_dir)


if __name__ == '__main__':
    main()
