import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .dataset import Ham10000Dataset, load_metadata_with_paths, make_splits
from .utils import HAM_CLASSES


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, accs = [], []
    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(accuracy(outputs, labels))
    return float(np.mean(losses)), float(np.mean(accs))


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="valid", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            accs.append(accuracy(outputs, labels))
    return float(np.mean(losses)), float(np.mean(accs))


def save_artifacts(output_dir: str, model: nn.Module) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "mobilenetv3_ham10000.pt")
    torch.save(model.state_dict(), model_path)
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        import json
        json.dump({i: name for i, name in enumerate(HAM_CLASSES)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/semi/Vscode/Rawdata Comvis")
    parser.add_argument("--output_dir", type=str, default="/home/semi/Vscode/Comvis/models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Select device: auto/cpu/cuda")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("Requested CUDA but it's unavailable. Falling back to CPU.")
    else:
        device = torch.device("cpu")

    df = load_metadata_with_paths(args.data_root)
    train_df, val_df = make_splits(df, val_size=args.val_size)

    train_dataset = Ham10000Dataset(train_df, args.data_root, is_train=True, image_size=args.image_size)
    val_dataset = Ham10000Dataset(val_df, args.data_root, is_train=False, image_size=args.image_size)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    model = build_model(num_classes=len(HAM_CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"train loss={train_loss:.4f} acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_artifacts(args.output_dir, model)
            print(f"Saved new best model to {args.output_dir} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
