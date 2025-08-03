#!/usr/bin/env python
"""
Convert *once* the JPEG tree of Food-101 into three LMDB files:
    food101_train.lmdb, food101_val.lmdb, food101_test.lmdb
You only run this script once; training never touches JPEGs again.
-----------------------------------------------------------------
$ python tools/build_food101_lmdb.py --root data --out data/food101_lmdb \
        --split_per 0.75 --seed 1
"""
import argparse, os, io, pickle, tqdm, lmdb
from torchvision.datasets import Food101
from PIL import Image
import numpy as np

P = argparse.ArgumentParser()
P.add_argument('--root',      required=True, help='where torchvision downloads Food-101')
P.add_argument('--out',       required=True, help='output dir for *.lmdb')
P.add_argument('--split_per', type=float, default=0.75,
               help='fraction of noisy-train kept for *train* split (rest → val)')
P.add_argument('--seed',      type=int,   default=1)
args = P.parse_args()

def write_split(name, indices, base_ds):
    path = f"{args.out}/food101_{name}.lmdb"
    os.makedirs(args.out, exist_ok=True)
    env = lmdb.open(path, map_size=50_000_000_000, subdir=False)
    keys = []
    with env.begin(write=True) as txn:
        for i in tqdm.tqdm(indices, desc=f"→ {name}"):
            img, lbl = base_ds[i]
            buf = io.BytesIO(); img.save(buf, format='PNG'); buf = buf.getvalue()
            k = f"{i:08d}".encode()
            txn.put(k, pickle.dumps((buf, lbl), protocol=pickle.HIGHEST_PROTOCOL))
            keys.append(k)
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__',  pickle.dumps(len(keys)))
    env.sync(); env.close()

# ---------- prepare indices ----------
train_full = Food101(args.root, split='train', download=True)
N = len(train_full)
idx = np.arange(N); rng = np.random.default_rng(args.seed); rng.shuffle(idx)
cut = int(N * args.split_per)
train_idx, val_idx = idx[:cut], idx[cut:]

write_split('train', train_idx, train_full)
write_split('val',   val_idx,   train_full)

test_full = Food101(args.root, split='test',  download=True)
write_split('test', range(len(test_full)), test_full)

print("LMDBs written to", args.out)
