
import argparse
import os
import random
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import timm
from sklearn.model_selection import StratifiedKFold

ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unzip_if_needed(zip_path: Path, out_root: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到压缩包: {zip_path}")

    target_guess = out_root / zip_path.stem
    if target_guess.exists() and any(target_guess.iterdir()):
        print(f"{target_guess} already exists, skip unzip")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Unzip {zip_path} -> {out_root}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_root)


def find_image(root: Path, filename: str):
    p = root / filename
    if p.exists():
        return p
    hits = list(root.rglob(filename))
    return hits[0] if hits else None


def read_train_list(train_list: Path, train_root: Path):
    samples = []
    with open(train_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            rel = parts[0].replace("\\", "/")
            label = int(parts[1])
            name = os.path.basename(rel)

            img_path = find_image(train_root, name)
            if img_path is not None:
                samples.append((str(img_path), label, name))

    if not samples:
        raise RuntimeError("训练样本数为0，请检查 train_list.txt 和训练图片是否匹配。")

    return samples


def get_test_names_from_zip(test_zip: Path):
    names = []
    with zipfile.ZipFile(test_zip, "r") as z:
        for name in z.namelist():
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                names.append(os.path.basename(name))
    return names


class CatDataset(Dataset):
    def __init__(self, samples, transform=None, is_test=False):
        self.samples = samples
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path, name = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, name

        img_path, label, name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0), ratio=(0.78, 1.28)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
    ])

    valid_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, valid_tf


def create_model(model_name, num_classes=12, drop_rate=0.15):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = False
        if any(k in name.lower() for k in ["classifier", "head", "fc"]):
            p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=110)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += correct
        total += bs

        pbar.set_postfix(loss=total_loss / total, acc=total_correct / total)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Valid", ncols=110):
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        logits = model(images)
        loss = criterion(logits, labels)

        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += correct
        total += bs

    return total_loss / total, total_correct / total


def get_fold_indices(samples, n_splits, seed):
    labels = np.array([s[1] for s in samples])
    indices = np.arange(len(samples))

    if n_splits <= 1:
        rng = np.random.default_rng(seed)
        train_idx, val_idx = [], []
        by_label = defaultdict(list)
        for i, s in enumerate(samples):
            by_label[s[1]].append(i)
        for label, ids in by_label.items():
            ids = np.array(ids)
            rng.shuffle(ids)
            n_val = max(1, int(len(ids) * 0.12))
            val_idx.extend(ids[:n_val].tolist())
            train_idx.extend(ids[n_val:].tolist())
        return [(np.array(train_idx), np.array(val_idx))]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(indices, labels))


def train_fold(args, fold, train_samples, val_samples, device, train_tf, valid_tf, ckpt_dir):
    print(f"\n========== Fold {fold} ==========")
    print("Train:", len(train_samples), "Valid:", len(val_samples))

    train_ds = CatDataset(train_samples, train_tf)
    val_ds = CatDataset(val_samples, valid_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = create_model(args.model, 12, args.drop_rate).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = 0.0
    best_path = ckpt_dir / f"fold{fold}_{safe_name(args.model)}.pth"

    if args.freeze_epochs > 0:
        print("Stage 1: freeze backbone")
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        for epoch in range(1, args.freeze_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Fold {fold} Stage1 Epoch {epoch}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                print("Save best:", best_path, best_acc)

    print("Stage 2: full fine-tune")
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * args.finetune_lr_ratio,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Fold {fold} Stage2 Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print("Save best:", best_path, best_acc)

    print(f"Fold {fold} best acc: {best_acc:.4f}")
    return best_path, best_acc


def safe_name(s):
    return s.replace("/", "_").replace(":", "_").replace(".", "_")


@torch.no_grad()
def predict_ensemble(args, checkpoint_paths, test_dir, test_names, valid_tf, device, result_path):
    models = []
    for ckpt in checkpoint_paths:
        model = create_model(args.model, 12, args.drop_rate).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    test_samples = []
    for name in test_names:
        p = find_image(test_dir, name)
        if p is None:
            print("Warning: 找不到测试图片，默认填0:", name)
        else:
            test_samples.append((str(p), name))

    ds = CatDataset(test_samples, valid_tf, is_test=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pred_map = {}

    for images, names in tqdm(loader, desc="Predict", ncols=110):
        images = images.to(device, non_blocking=True)

        probs_sum = None
        for model in models:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            if args.tta:
                logits_flip = model(torch.flip(images, dims=[3]))
                probs = (probs + torch.softmax(logits_flip, dim=1)) / 2.0

            probs_sum = probs if probs_sum is None else probs_sum + probs

        probs_avg = probs_sum / len(models)
        preds = probs_avg.argmax(dim=1).cpu().numpy().tolist()

        for name, pred in zip(names, preds):
            pred_map[name] = int(pred)

    with open(result_path, "w", encoding="utf-8", newline="") as f:
        for name in test_names:
            pred = pred_map.get(name, 0)
            f.write(f"{name},{pred}\n")

    print("Saved:", result_path)
    print("Preview:")
    with open(result_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--work_dir", type=str, default="work")

    parser.add_argument(
        "--model",
        type=str,
        default="hf_hub:timm/tf_efficientnetv2_s.in21k_ft_in1k",
        help="推荐：hf_hub:timm/tf_efficientnetv2_s.in21k_ft_in1k 或 hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384"
    )
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--freeze_epochs", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--finetune_lr_ratio", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--drop_rate", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--skip_train", action="store_true", help="已有ckpt时只预测")
    args = parser.parse_args()

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_zip = data_dir / "cat_12_train.zip"
    test_zip = data_dir / "cat_12_test.zip"
    train_list = data_dir / "train_list.txt"

    unzip_if_needed(train_zip, work_dir)
    unzip_if_needed(test_zip, work_dir)

    train_dir = work_dir / "cat_12_train"
    test_dir = work_dir / "cat_12_test"

    samples = read_train_list(train_list, train_dir)
    print("Total train samples:", len(samples))

    train_tf, valid_tf = build_transforms(args.img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    ckpt_dir = work_dir / "checkpoints" / safe_name(args.model)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = []

    if args.skip_train:
        checkpoint_paths = sorted(ckpt_dir.glob("*.pth"))
        if not checkpoint_paths:
            raise RuntimeError("skip_train=True，但没有找到任何 checkpoint。")
    else:
        splits = get_fold_indices(samples, args.folds, args.seed)
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            ckpt, acc = train_fold(args, fold, train_samples, val_samples, device, train_tf, valid_tf, ckpt_dir)
            checkpoint_paths.append(ckpt)

    test_names = get_test_names_from_zip(test_zip)
    print("Test images:", len(test_names))

    result_path = work_dir / "result.csv"
    predict_ensemble(args, checkpoint_paths, test_dir, test_names, valid_tf, device, result_path)


if __name__ == "__main__":
    main()
