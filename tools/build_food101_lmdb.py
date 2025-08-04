#!/usr/bin/env python
"""
Convert the Food-101 image tree into fast, mmap-friendly LMDB files
(one for **train**, **val**, and **test**).  You can:
  • build the three splits in one go      (--split all)      ← default
  • build one split at a time (safer in Colab)  (--split train / val / test)

Typical usage ­– build *train* first, then *val* & *test*:
-----------------------------------------------------------------
python tools/build_food101_lmdb.py --root data \
       --out  /content/drive/MyDrive/food101_lmdb \
       --split train --split_per 0.75 --seed 1

# if Colab disconnects, rerun the same line; it appends -- no harm
python tools/build_food101_lmdb.py --root data \
       --out  /content/drive/MyDrive/food101_lmdb \
       --split val   --split_per 0.75 --seed 1

python tools/build_food101_lmdb.py --root data \
       --out  /content/drive/MyDrive/food101_lmdb \
       --split test
-----------------------------------------------------------------
LMDB structure
--------------
Every key  = b"{img_id:08d}"
Every value= pickle.dumps((PNG_bytes, label))
Extra keys = b'__keys__' (list of keys)  &  b'__len__' (int)
"""

# -----------------------------------------------------------------
#  Imports
# -----------------------------------------------------------------
import argparse, os, io, pickle, tqdm, lmdb
from torchvision.datasets import Food101
from PIL import Image          # pillow: converts tensor→PNG easily
import numpy as np

# -----------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------
P = argparse.ArgumentParser()
P.add_argument('--root',      required=True, help='where torchvision downloads JPEGs')
P.add_argument('--out',       required=True, help='directory that will hold *.lmdb')
P.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all',
               help='build only one split (default: all three)')
P.add_argument('--split_per', type=float, default=0.75,
               help='fraction of noisy-train kept for *train* (rest → val)')
P.add_argument('--seed',      type=int,   default=1,
               help='rng seed for the train/val shuffle split')
args = P.parse_args()

# -----------------------------------------------------------------
#  Helper: write a single LMDB split
# -----------------------------------------------------------------
def write_split(name: str, indices, base_ds):
    """
    Parameters
    ----------
    name     : 'train' | 'val' | 'test'
    indices  : iterable[int]  indices into base_ds
    base_ds  : torchvision.datasets.Food101 object (provides JPEGs)
    """
    # final LMDB file
    lmdb_path = f"{args.out}/food101_{name}.lmdb"
    os.makedirs(args.out, exist_ok=True)

    # 50 GB map_size is plenty (the whole dataset is ≈ 13 GB as PNG)
    env = lmdb.open(lmdb_path, map_size=50_000_000_000, subdir=False)

    keys = []                              # we record keys for __keys__
    with env.begin(write=True) as txn:     # one big write txn – fastest
        for i in tqdm.tqdm(indices, desc=f"→ {name}"):
            img, lbl = base_ds[i]          # img is a PIL.Image
            buf = io.BytesIO()
            img.save(buf, format='PNG')    # lossless & ~2× smaller than JPEG
            png_bytes = buf.getvalue()

            k = f"{i:08d}".encode()        # e.g. b'00001234'
            txn.put(k, pickle.dumps((png_bytes, lbl), protocol=pickle.HIGHEST_PROTOCOL))
            keys.append(k)

        # meta-information so the DataLoader can seek efficiently
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__',  pickle.dumps(len(keys)))

    env.sync(); env.close()
    print(f"[✓] {name:5s}  →  {lmdb_path}  ({len(keys):,} samples)")

# -----------------------------------------------------------------
#  Build requested splits
# -----------------------------------------------------------------
if args.split in ('train', 'all', 'val'):
    # one download covers both train & val
    train_full = Food101(args.root, split='train', download=True)

    # same random shuffle for train+val so they partition nicely
    rng  = np.random.default_rng(args.seed)
    idx  = np.arange(len(train_full))
    rng.shuffle(idx)

    cut  = int(len(idx) * args.split_per)
    train_idx, val_idx = idx[:cut], idx[cut:]

    if args.split in ('train', 'all'):
        write_split('train', train_idx, train_full)

    if args.split in ('val', 'all'):
        write_split('val', val_idx, train_full)

if args.split in ('test', 'all'):
    test_full = Food101(args.root, split='test', download=True)
    write_split('test', range(len(test_full)), test_full)

print("\nAll requested LMDB files written ✔")
