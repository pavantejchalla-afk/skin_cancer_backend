import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import CenterCrop, ColorJitter, Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor

from app.services.class_labels import CLASS_LABELS


class SkinDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(item["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, item["label_idx"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for skin lesion classification")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--csv_path", type=Path, required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--output", type=Path, default=Path("models/mobilenetv2_skin.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset_from_folders(data_dir):
    class_dirs = [path for path in data_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {data_dir}")

    class_names = sorted([path.name for path in class_dirs])
    if set(class_names) != set(CLASS_LABELS.keys()):
        raise ValueError(f"Expected classes: {sorted(CLASS_LABELS.keys())}, got: {class_names}")

    items = []
    for label in class_names:
        for image_path in (data_dir / label).glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            items.append({"path": image_path, "label": label})

    return items, class_names


def build_dataset_from_metadata(csv_path, data_dir):
    image_mapping = {}
    for folder in [data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]:
        if folder.exists():
            for image_path in folder.glob("*"):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    image_mapping[image_path.stem] = image_path

    if not image_mapping:
        raise FileNotFoundError("Could not locate HAM10000 image directories. Expected HAM10000_images_part_1 and HAM10000_images_part_2 under the data_dir.")

    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row.get("image_id")
            dx = row.get("dx")
            if not image_id or not dx or dx not in CLASS_LABELS:
                continue
            image_path = image_mapping.get(image_id)
            if image_path is None:
                continue
            rows.append({"path": image_path, "label": dx})

    if not rows:
        raise ValueError("No matching rows were found in the metadata CSV.")

    return rows, sorted(CLASS_LABELS.keys())


def split_items(items, val_split, seed):
    random.seed(seed)
    shuffled = items[:]
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_split)))
    return shuffled[:split_idx], shuffled[split_idx:]


def get_transforms(train: bool):
    if train:
        return Compose(
            [
                Resize((256, 256)),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {args.csv_path}")

    if args.data_dir.is_dir() and any((args.data_dir / label).is_dir() for label in CLASS_LABELS):
        items, class_names = build_dataset_from_folders(args.data_dir)
    else:
        items, class_names = build_dataset_from_metadata(args.csv_path, args.data_dir)

    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    train_items, val_items = split_items(items, args.val_split, args.seed)

    for item in train_items + val_items:
        item["label_idx"] = class_to_idx[item["label"]]

    train_dataset = SkinDataset(train_items, transform=get_transforms(train=True))
    val_dataset = SkinDataset(val_items, transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={running_loss / max(1, len(train_loader)):.4f} | "
            f"val_acc={correct / total:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": class_names,
            "epochs": args.epochs,
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
