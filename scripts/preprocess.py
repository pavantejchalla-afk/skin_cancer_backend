import argparse
import csv
import shutil
from pathlib import Path

from app.services.class_labels import CLASS_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare class-based training directories from HAM10000 metadata")
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        for label in CLASS_LABELS:
            (args.output_dir / split / label).mkdir(parents=True, exist_ok=True)

    image_map = {}
    for folder in [args.image_dir / "HAM10000_images_part_1", args.image_dir / "HAM10000_images_part_2"]:
        if folder.exists():
            for image_path in folder.glob("*"):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    image_map[image_path.stem] = image_path

    with args.csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            dx = row.get("dx")
            image_id = row.get("image_id")
            if dx not in CLASS_LABELS or image_id not in image_map:
                continue

            split = "train" if idx % 5 != 0 else "val"
            dest = args.output_dir / split / dx / f"{image_id}.jpg"
            if not dest.exists():
                shutil.copy2(image_map[image_id], dest)

    print(f"Prepared dataset in {args.output_dir}")


if __name__ == "__main__":
    main()
