"""Check that nano's train.bin sample 32000..32063 matches ref's iter 501 batch.

Ref at iter 501 consumed samples 500*64..501*64-1 = 32000..32063.
Nano reads from train.bin linearly: sample i at offset i * block_size.
"""
import argparse, os, sys
import numpy as np

MEGATRON_PY = '/newcpfs/user/yuchen/llm/megatron_dots3.0_swa'
CYBERTRON_PY = '/newcpfs/user/yuchen/llm/cybertron_dots3.0_swa'

SEQ_LENGTH = 8192
DATA_CACHE = '/prodcpfs/user/data/save/data/lossalign/data_cache'
BLENDED_HASH = '43adec39b46f5eb95d144361a0db6699'


def fetch_ref_sample(i: int) -> np.ndarray:
    """Fetch sample i using the same logic as Megatron's BlendedDataset.__getitem__."""
    sys.path.insert(0, MEGATRON_PY)
    sys.path.insert(0, CYBERTRON_PY)
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    import hashlib, json

    dataset_index = np.load(f'{DATA_CACHE}/{BLENDED_HASH}-BlendedDataset-train-dataset_index.npy', mmap_mode='r')
    dataset_sample_index = np.load(f'{DATA_CACHE}/{BLENDED_HASH}-BlendedDataset-train-dataset_sample_index.npy', mmap_mode='r')
    desc = json.load(open(f'{DATA_CACHE}/{BLENDED_HASH}-BlendedDataset-train-description.txt'))

    ds_id = int(dataset_index[i])
    sub_pos = int(dataset_sample_index[i])
    ds_desc = desc['datasets'][ds_id]
    split = ds_desc['index_split']
    ds_hash = hashlib.md5(json.dumps(ds_desc, indent=4).encode('utf-8'), usedforsecurity=False).hexdigest()

    doc_idx = np.load(f'{DATA_CACHE}/{ds_hash}-MCoreGPTDataset-{split}-document_index.npy', mmap_mode='r')
    shuffle_idx = np.load(f'{DATA_CACHE}/{ds_hash}-MCoreGPTDataset-{split}-shuffle_index.npy', mmap_mode='r')
    sample_idx = np.load(f'{DATA_CACHE}/{ds_hash}-MCoreGPTDataset-{split}-sample_index.npy', mmap_mode='r')

    ds = IndexedDataset(ds_desc['dataset_path'], mmap=True)

    actual_id = int(shuffle_idx[sub_pos])
    # Reconstruct packed sample of seq_len+1 (Megatron grabs seq_len+1 for target shift)
    doc_index_beg, offset_start = sample_idx[actual_id]
    doc_index_end, offset_end = sample_idx[actual_id + 1]
    if doc_index_beg == doc_index_end:
        tok = ds.get(int(doc_idx[int(doc_index_beg)]))[int(offset_start):int(offset_start) + SEQ_LENGTH + 1]
    else:
        tok = list(ds.get(int(doc_idx[int(doc_index_beg)]))[int(offset_start):])
        for di in range(int(doc_index_beg) + 1, int(doc_index_end) + 1):
            doc = ds.get(int(doc_idx[di]))
            if di < int(doc_index_end):
                tok.extend(doc)
            else:
                tok.extend(doc[:int(offset_end) + 1])
        tok = tok[:SEQ_LENGTH + 1]
    tok = np.asarray(tok, dtype=np.int64)
    if len(tok) < SEQ_LENGTH + 1:
        tok = np.pad(tok, (0, SEQ_LENGTH + 1 - len(tok)))
    return tok[:SEQ_LENGTH + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/nanogpt/data/cybertron_baseline/train.bin')
    ap.add_argument('--sample', type=int, default=32000, help='Global sample idx (iter 501 starts at 32000)')
    args = ap.parse_args()

    # nano read
    arr = np.memmap(args.data, dtype=np.int32, mode='r')
    nano_sample = np.array(arr[args.sample * SEQ_LENGTH : args.sample * SEQ_LENGTH + SEQ_LENGTH + 1].astype(np.int64))
    print(f'nano train.bin sample {args.sample} first 16: {nano_sample[:16].tolist()}')
    print(f'nano train.bin sample {args.sample} last 16:  {nano_sample[-16:].tolist()}')

    # ref read
    ref_sample = fetch_ref_sample(args.sample)
    print(f'ref Megatron  sample {args.sample} first 16: {ref_sample[:16].tolist()}')
    print(f'ref Megatron  sample {args.sample} last 16:  {ref_sample[-16:].tolist()}')

    eq = np.array_equal(nano_sample, ref_sample)
    diffs = np.nonzero(nano_sample != ref_sample)[0]
    print(f'\nbitwise equal? {eq}')
    if not eq:
        print(f'diff positions: {len(diffs)} / {len(nano_sample)}')
        if len(diffs) > 0:
            print(f'first diff pos {diffs[0]}: nano={nano_sample[diffs[0]]} ref={ref_sample[diffs[0]]}')


if __name__ == '__main__':
    main()
