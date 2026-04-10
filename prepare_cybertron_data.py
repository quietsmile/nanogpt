"""
Prepare training/validation data for nanogpt by extracting it from cybertron's
blended megatron dataset cache, preserving the EXACT same data order as cybertron.

This script directly reads the pre-built cache files rather than rebuilding them.

Cache info:
  scaling_moe_00196 (PAI: dlc1q9arre48b0kx):
    BlendedDataset hash: 43adec39b46f5eb95d144361a0db6699
    Cache path: /prodcpfs/user/data/save/data/lossalign/data_cache
    Train samples: 479040

  scaling_moe_00198 (PAI: dlc1v93gneqwnrdz):
    BlendedDataset hash: b98975b5a0b37ed3a7f5437dc5951576
    Cache path: /prodcpfs/user/data/save/data/data_cache
    Train samples: 1388416

Output:
  data/cybertron_baseline/train.bin   -- uint16 token array, sequential samples
  data/cybertron_baseline/val.bin     -- validation tokens (Pile_test_5k)
  data/cybertron_baseline/meta.pkl    -- vocab_size=152064

Usage:
  # For 198 (default):
  python prepare_cybertron_data.py [--n_train_samples N] [--n_val_samples N]
  # For 196:
  python prepare_cybertron_data.py --exp 196
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np

# Paths
MEGATRON_PATH   = '/newcpfs/user/yuchen/llm/megatron_dots3.0_swa'
sys.path.insert(0, MEGATRON_PATH)

# Per-experiment configs
EXP_CONFIGS = {
    '196': {
        'data_cache_path': '/prodcpfs/user/data/save/data/lossalign/data_cache',
        'blended_hash':    '43adec39b46f5eb95d144361a0db6699',
        'n_train_samples': 479040,
    },
    '198': {
        'data_cache_path': '/prodcpfs/user/data/save/data/data_cache',
        'blended_hash':    'b98975b5a0b37ed3a7f5437dc5951576',
        'n_train_samples': 1388416,
    },
}

# Will be set in main() based on --exp argument
DATA_CACHE_PATH = None
BLENDED_HASH    = None
SEQ_LENGTH      = 8192
VOCAB_SIZE      = 152064

# Validation: Pile_test_5k (megatron indexed dataset)
VAL_DATA_PATH = '/cpfs/user/wangzerui/cybertron_workspace/datasets/prod_validation_datasets/OOD_Validation/megatron_bins_raw/Pile_test_5k_gpt'


def load_blended_index():
    """Load the BlendedDataset index arrays from cache."""
    cache = DATA_CACHE_PATH
    h = BLENDED_HASH

    dataset_index = np.load(
        f'{cache}/{h}-BlendedDataset-train-dataset_index.npy', mmap_mode='r'
    )
    sample_index = np.load(
        f'{cache}/{h}-BlendedDataset-train-dataset_sample_index.npy', mmap_mode='r'
    )
    return dataset_index, sample_index


def load_blended_description():
    """Load the BlendedDataset description to get per-dataset paths and hashes."""
    desc_path = f'{DATA_CACHE_PATH}/{BLENDED_HASH}-BlendedDataset-train-description.txt'
    with open(desc_path) as f:
        d = json.load(f)
    # The blended description wraps datasets in a top-level 'datasets' key
    return d


def build_dataset_hash(dataset_desc):
    """Compute the per-dataset hash from its description dict.
    This matches megatron's hashlib.md5(unique_description) logic.
    """
    import hashlib
    from collections import OrderedDict
    # Re-create the unique_description JSON exactly as megatron does
    unique_desc = json.dumps(dataset_desc, indent=4)
    return hashlib.md5(unique_desc.encode('utf-8'), usedforsecurity=False).hexdigest()


def load_per_dataset_indices(dataset_desc):
    """Load per-dataset document_index, shuffle_index, and sample_index from cache."""
    h = build_dataset_hash(dataset_desc)
    cache = DATA_CACHE_PATH
    split = dataset_desc['index_split']

    doc_idx = np.load(
        f'{cache}/{h}-MCoreGPTDataset-{split}-document_index.npy', mmap_mode='r'
    )
    shuffle_idx = np.load(
        f'{cache}/{h}-MCoreGPTDataset-{split}-shuffle_index.npy', mmap_mode='r'
    )
    sample_idx = np.load(
        f'{cache}/{h}-MCoreGPTDataset-{split}-sample_index.npy', mmap_mode='r'
    )
    return doc_idx, shuffle_idx, sample_idx


def open_indexed_dataset(path):
    """Open a megatron IndexedDataset with mmap for OS-cached access (no DirectIO)."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    return IndexedDataset(path, mmap=True)


def get_sample_tokens_cached(indexed_ds, doc_idx, sample_idx, actual_sample_id, seq_len):
    """Extract seq_len tokens for one sample.

    indexed_ds: mmap-backed IndexedDataset (fast page-cached access)
    doc_idx: document_index array (doc_index → real_doc_id)
    sample_idx: sample_index array (actual_id → (doc_index_beg, offset))
    """
    doc_index_beg, offset_start = sample_idx[actual_sample_id]
    doc_index_end, offset_end   = sample_idx[actual_sample_id + 1]

    if doc_index_beg == doc_index_end:
        real_doc_id = int(doc_idx[int(doc_index_beg)])
        doc = indexed_ds.get(real_doc_id)
        tokens = doc[int(offset_start):int(offset_start) + seq_len]
    else:
        real_doc_id = int(doc_idx[int(doc_index_beg)])
        tokens = list(indexed_ds.get(real_doc_id)[int(offset_start):])
        for di in range(int(doc_index_beg) + 1, int(doc_index_end) + 1):
            real_doc_id = int(doc_idx[di])
            doc = indexed_ds.get(real_doc_id)
            if di < int(doc_index_end):
                tokens.extend(doc)
            else:
                tokens.extend(doc[:int(offset_end)])
        tokens = tokens[:seq_len]

    # Pad or truncate to seq_len
    tokens = np.array(tokens, dtype=np.int32)
    if len(tokens) < seq_len:
        tokens = np.pad(tokens, (0, seq_len - len(tokens)))
    else:
        tokens = tokens[:seq_len]

    return tokens.astype(np.uint16)


def _extract_dataset_worker(args):
    """Top-level worker for multiprocessing: extracts all samples for one dataset.

    Must be at module level so it's picklable.
    Each worker independently loads its own indices and IndexedDataset handle.
    Writes directly to shared output memmap (non-overlapping regions → safe).
    """
    import sys
    sys.path.insert(0, MEGATRON_PATH)

    ds_id        = args['ds_id']
    blended_idxs = args['blended_idxs']
    ds_desc      = args['ds_desc']
    out_path     = args['out_path']
    n_total      = args['n_samples']

    # Load indices for this dataset
    try:
        doc_idx, shuffle_idx, sample_idx = load_per_dataset_indices(ds_desc)
        indexed_ds = open_indexed_dataset(ds_desc['dataset_path'])
    except Exception as e:
        return ds_id, 0, len(blended_idxs)

    # Load blended sample_index for this dataset's samples
    dataset_si = np.load(
        f'{DATA_CACHE_PATH}/{BLENDED_HASH}-BlendedDataset-train-dataset_sample_index.npy',
        mmap_mode='r'
    )

    # Compute (actual_id, blended_idx) pairs sorted by actual_id
    pairs = [(int(shuffle_idx[int(dataset_si[i])]), i) for i in blended_idxs]
    pairs.sort()

    # Open output memmap for writing
    out = np.memmap(out_path, dtype=np.uint16, mode='r+', shape=(n_total * SEQ_LENGTH,))

    for actual_id, i in pairs:
        tokens = get_sample_tokens_cached(indexed_ds, doc_idx, sample_idx, actual_id, SEQ_LENGTH)
        out[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH] = tokens
    out.flush()

    return ds_id, len(pairs), 0


def extract_train_data(n_samples, out_path):
    """Extract n_samples training samples in cybertron's exact order."""
    print("Loading BlendedDataset indices...")
    dataset_index, sample_index = load_blended_index()
    print(f"  Total blended samples available: {len(dataset_index):,}")
    print(f"  Extracting first {n_samples:,} samples")

    desc = load_blended_description()
    datasets = desc['datasets']
    print(f"  Num sub-datasets: {len(datasets)}")

    # Quick check which datasets are available (without loading all data)
    needed_ds_ids = set(int(x) for x in dataset_index[:n_samples])
    print(f"  Datasets needed for {n_samples:,} samples: {len(needed_ds_ids)}")

    # Verify index files exist for all needed datasets
    missing = 0
    per_ds_shuffle = {}  # track which datasets have cache (None = missing)
    for ds_id in sorted(needed_ds_ids):
        ds_desc = datasets[ds_id]
        h = build_dataset_hash(ds_desc)
        split = ds_desc['index_split']
        shuf_path = f'{DATA_CACHE_PATH}/{h}-MCoreGPTDataset-{split}-shuffle_index.npy'
        if not os.path.exists(shuf_path):
            print(f"  WARNING: cache miss for dataset {ds_id}: {shuf_path}")
            per_ds_shuffle[ds_id] = None
            missing += 1
        else:
            per_ds_shuffle[ds_id] = True
    print(f"  {len(needed_ds_ids) - missing} datasets have valid cache ({missing} missing)")

    # Group blended sample indices by dataset, then sort by actual_id within each
    # dataset for sequential document access (avoids random IO over shuffled order).
    print("Grouping samples by dataset for sequential IO...")
    from collections import defaultdict
    ds_to_blended_idxs = defaultdict(list)
    for i in range(n_samples):
        ds_id = int(dataset_index[i])
        ds_to_blended_idxs[ds_id].append(i)

    # Write output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(n_samples * SEQ_LENGTH,))
    print(f"Writing {n_samples:,} samples → {out_path} ({n_samples * SEQ_LENGTH * 2 / 1e9:.2f} GB)")

    import multiprocessing as mp

    # Build args list sorted by sample count (largest first for better load balancing)
    args_list = []
    skipped = 0
    for ds_id in sorted(ds_to_blended_idxs.keys(),
                        key=lambda d: len(ds_to_blended_idxs[d]), reverse=True):
        if per_ds_shuffle[ds_id] is None:
            print(f"  Dataset {ds_id}: SKIPPED (cache miss)", flush=True)
            skipped += len(ds_to_blended_idxs[ds_id])
            continue
        args_list.append({
            'ds_id': ds_id,
            'blended_idxs': ds_to_blended_idxs[ds_id],
            'ds_desc': datasets[ds_id],
            'out_path': out_path,
            'n_samples': n_samples,
        })

    n_workers = min(16, len(args_list))
    print(f"Processing {len(args_list)} datasets with {n_workers} parallel workers...")

    written = 0
    with mp.Pool(processes=n_workers) as pool:
        for ds_id_done, n_written, n_skip in pool.imap_unordered(_extract_dataset_worker, args_list):
            written += n_written
            skipped += n_skip
            print(f"  Dataset {ds_id_done}: done ({n_written:,} samples)", flush=True)

    out.flush()
    print(f"Done. Wrote {written:,}/{n_samples:,} samples, skipped {skipped:,}")
    if skipped > 0:
        print(f"WARNING: {skipped} samples were skipped due to cache misses")


def extract_val_data(out_path, n_samples=2000):
    """Extract validation data from Pile_test_5k megatron dataset."""
    print(f"\nExtracting {n_samples:,} validation sequences from Pile_test_5k...")

    if not os.path.exists(VAL_DATA_PATH + '.bin'):
        print(f"WARNING: Validation data not found at {VAL_DATA_PATH}")
        print("Creating minimal placeholder...")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.zeros(n_samples * SEQ_LENGTH, dtype=np.uint16).tofile(out_path)
        return

    indexed_ds = open_indexed_dataset(VAL_DATA_PATH)
    print(f"  Validation dataset: {len(indexed_ds)} docs")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(n_samples * SEQ_LENGTH,))

    written = 0
    doc_idx = 0
    buf = np.array([], dtype=np.uint16)

    while written < n_samples and doc_idx < len(indexed_ds):
        doc = indexed_ds.get(doc_idx).astype(np.uint16)
        buf = np.concatenate([buf, doc])
        doc_idx += 1

        while len(buf) >= SEQ_LENGTH and written < n_samples:
            out[written * SEQ_LENGTH:(written + 1) * SEQ_LENGTH] = buf[:SEQ_LENGTH]
            buf = buf[SEQ_LENGTH:]
            written += 1

    out.flush()
    print(f"  Wrote {written:,} validation sequences to {out_path}")


def main():
    global DATA_CACHE_PATH, BLENDED_HASH

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='198', choices=['196', '198'],
                        help='Which cybertron experiment to align with (default: 198)')
    parser.add_argument('--n_train_samples', type=int, default=None,
                        help='Training samples to extract (default: from exp config)')
    parser.add_argument('--n_val_samples', type=int, default=2000)
    parser.add_argument('--out_dir', type=str, default='data/cybertron_baseline')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_val', action='store_true')
    args = parser.parse_args()

    # Set globals from experiment config
    exp_cfg = EXP_CONFIGS[args.exp]
    DATA_CACHE_PATH = exp_cfg['data_cache_path']
    BLENDED_HASH    = exp_cfg['blended_hash']
    if args.n_train_samples is None:
        args.n_train_samples = exp_cfg['n_train_samples']

    print(f"Experiment: scaling_moe_00{args.exp}")
    print(f"  data_cache_path: {DATA_CACHE_PATH}")
    print(f"  blended_hash:    {BLENDED_HASH}")
    print(f"  n_train_samples: {args.n_train_samples:,}")

    os.makedirs(args.out_dir, exist_ok=True)

    # meta.pkl
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        with open(meta_path, 'wb') as f:
            pickle.dump({'vocab_size': VOCAB_SIZE}, f)
        print(f"Wrote {meta_path}")

    if not args.skip_train:
        train_out = os.path.join(args.out_dir, 'train.bin')
        if os.path.exists(train_out):
            print(f"train.bin already exists ({os.path.getsize(train_out)/1e9:.2f} GB). Delete to regenerate.")
        else:
            extract_train_data(args.n_train_samples, train_out)

    if not args.skip_val:
        val_out = os.path.join(args.out_dir, 'val.bin')
        if os.path.exists(val_out):
            print(f"val.bin already exists. Delete to regenerate.")
        else:
            extract_val_data(val_out, args.n_val_samples)

    print(f"\nDone! Output in: {args.out_dir}")


if __name__ == '__main__':
    main()
