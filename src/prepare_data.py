"""Split the raw TrashNet images into train/val/test folders (stratified by class)."""
import random
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "split"

SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# remaining ~0.15 goes to test

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    random.seed(SEED)
    classes = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    print(f"Classes found: {classes}")

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    summary = {}
    for cls in classes:
        files = [p for p in (RAW_DIR / cls).iterdir() if p.suffix.lower() in IMG_EXTS]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        summary[cls] = {k: len(v) for k, v in splits.items()}

        for split_name, split_files in splits.items():
            out_dir = SPLIT_DIR / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir / f.name)

    print("\nSplit summary (train/val/test):")
    total = {"train": 0, "val": 0, "test": 0}
    for cls, counts in summary.items():
        print(f"  {cls:10s} {counts}")
        for k in total:
            total[k] += counts[k]
    print(f"  {'TOTAL':10s} {total}")


if __name__ == "__main__":
    main()
