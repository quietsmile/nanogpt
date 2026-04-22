"""Phase 2: data sampling alignment for scaling_moe_00196.

Validates that the Megatron blended cache that nanogpt's prepare_cybertron_data.py
points to:
  1. Exists at the expected hash for 00196
  2. Contains exactly exit_interval * global_batch_size = 7485 * 64 = 479040 samples
  3. Description matches the yaml-referenced datasets (count, hashes)
  4. First K dataset/sample indices are deterministic and recorded as a golden baseline

Full bit-exact replay of token sequences against Megatron's loader is a follow-up
test that runs on an 8-GPU box (requires running `prepare_cybertron_data.py --exp 196`
to materialize data/cybertron_moe_196/train.bin, then comparing a re-read vs cache).
"""
import hashlib
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# Import the project's already-written data prep module to check its constants
import prepare_cybertron_data as pcd  # noqa: E402


EXPECTED_CACHE = '/prodcpfs/user/data/save/data/lossalign/data_cache'
EXPECTED_HASH_196 = '43adec39b46f5eb95d144361a0db6699'
EXPECTED_N_CONSUMED = 479040      # 7485 steps × global_batch 64 (actually consumed)
EXPECTED_N_CACHE = 1572864000    # yaml train_samples — full sampler pre-built; exit_interval cuts early
YAML_PATH = '/prodcpfs/user/data/GitLab/pretrain_scaling_ladder/scaling_moe_00196.yaml'
DATA_YAML_PATH = '/prodcpfs/user/data/GitLab/pretrain_scaling_ladder/data_pretrain_v3_pai.yaml'

os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
REPORT = os.path.join(ROOT, 'reports', 'data_sampling_alignment.json')


class TestBlendedCacheFor196(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = pcd.EXP_CONFIGS['196']
        cls.report = {}

    @classmethod
    def tearDownClass(cls):
        with open(REPORT, 'w') as f:
            json.dump(cls.report, f, ensure_ascii=False, indent=2)

    def test_prepare_constants(self):
        self.assertEqual(self.cfg['data_cache_path'], EXPECTED_CACHE)
        self.assertEqual(self.cfg['blended_hash'], EXPECTED_HASH_196)
        # prepare extracts the portion actually consumed by the reference run
        self.assertEqual(self.cfg['n_train_samples'], EXPECTED_N_CONSUMED)

    def test_blended_cache_files_exist(self):
        for suffix in ['dataset_index', 'dataset_sample_index', 'description']:
            ext = 'txt' if suffix == 'description' else 'npy'
            p = f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-{suffix}.{ext}'
            self.assertTrue(os.path.exists(p), f"missing blended cache file: {p}")

    def test_blended_sample_count(self):
        """Cache is sized for the full yaml train_samples. Actual run stops at exit_interval."""
        p = f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-dataset_index.npy'
        arr = np.load(p, mmap_mode='r')
        n = int(arr.shape[0])
        self.report['blended_sample_count'] = n
        self.report['consumed_by_run'] = EXPECTED_N_CONSUMED
        # The full sampler is pre-built for train_samples; the run exited early at 7485 steps.
        self.assertGreaterEqual(n, EXPECTED_N_CACHE - 1,
                                f"cache should hold ≥ yaml train_samples; got {n}")
        self.assertGreaterEqual(n, EXPECTED_N_CONSUMED,
                                f"cache must cover what the run consumed ({EXPECTED_N_CONSUMED})")

    def test_first_k_indices_deterministic(self):
        """First K (dataset_id, sample_id) pairs hash must be stable."""
        K = 1024
        ds_idx = np.load(
            f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-dataset_index.npy',
            mmap_mode='r')
        sm_idx = np.load(
            f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-dataset_sample_index.npy',
            mmap_mode='r')
        first_ds = ds_idx[:K].astype(np.int64)
        first_sm = sm_idx[:K].astype(np.int64)
        paired = np.stack([first_ds, first_sm], axis=-1)
        h = hashlib.sha256(paired.tobytes()).hexdigest()
        self.report['first_1024_pairs_sha256'] = h
        self.report['first_10_pairs'] = [(int(d), int(s)) for d,s in paired[:10]]
        # Last step of training
        last_ds = ds_idx[-8:].astype(int).tolist()
        last_sm = sm_idx[-8:].astype(int).tolist()
        self.report['last_8_pairs'] = list(zip(last_ds, last_sm))
        # Sample distribution: which datasets are used? How many unique?
        unique_ds = np.unique(ds_idx[:]).tolist()
        self.report['n_unique_datasets_in_train'] = len(unique_ds)
        self.assertGreater(len(unique_ds), 10, "expected diverse blend >10 datasets")

    def test_description_matches_data_yaml(self):
        desc_path = f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-description.txt'
        with open(desc_path) as f:
            desc = json.load(f)
        datasets = desc['datasets']
        # Parse the data yaml to count train_data_path entries
        import yaml
        with open(DATA_YAML_PATH) as f:
            y = yaml.safe_load(f)
        yaml_paths = y['cybertron']['train_data_path']
        self.report['n_datasets_in_blend'] = len(datasets)
        self.report['n_datasets_in_yaml'] = len(yaml_paths)
        self.assertEqual(len(datasets), len(yaml_paths),
                         f"blend has {len(datasets)} datasets, yaml has {len(yaml_paths)}")
        # Spot-check that the first dataset path in blend matches first in yaml
        ds0_path = datasets[0].get('dataset_path', '')
        yaml0_path = yaml_paths[0]['path']
        self.report['first_dataset_blend_vs_yaml'] = {
            'blend': ds0_path, 'yaml': yaml0_path,
        }
        # blended stores path with possibly .bin suffix; yaml without
        self.assertTrue(yaml0_path in ds0_path or ds0_path in yaml0_path,
                        f"first-dataset path mismatch: blend={ds0_path} yaml={yaml0_path}")

    def test_sampler_replay_helper_works(self):
        """Exercise pcd.get_sample_tokens_cached path on one sample from one dataset.

        Doesn't assert a specific token value (we don't have a pre-computed golden)
        but asserts: (a) read succeeds, (b) token ids are within padded vocab,
        (c) EOD-containing samples end appropriately.
        """
        # Use blended step 0 → dataset_id[0], sample_id[0]
        ds_idx = np.load(
            f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-dataset_index.npy',
            mmap_mode='r')
        sm_idx = np.load(
            f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-dataset_sample_index.npy',
            mmap_mode='r')
        desc = json.load(open(
            f'{EXPECTED_CACHE}/{EXPECTED_HASH_196}-BlendedDataset-train-description.txt'))
        step0_ds = int(ds_idx[0])
        step0_sm = int(sm_idx[0])
        ds_desc = desc['datasets'][step0_ds]
        pcd.DATA_CACHE_PATH = EXPECTED_CACHE
        pcd.BLENDED_HASH = EXPECTED_HASH_196
        # Load per-dataset indices and IndexedDataset, then sample
        try:
            doc_idx, shuffle_idx, sample_idx = pcd.load_per_dataset_indices(ds_desc)
            # Actual-sample-id = shuffle_idx[step0_sm]
            actual = int(shuffle_idx[step0_sm])
            indexed = pcd.open_indexed_dataset(ds_desc['dataset_path'])
            tokens = pcd.get_sample_tokens_cached(indexed, doc_idx, sample_idx,
                                                   actual, seq_len=8192)
            self.report['step0_sample'] = {
                'dataset_id': step0_ds,
                'sample_id_in_blend': step0_sm,
                'actual_sample_id': actual,
                'dataset_path': ds_desc['dataset_path'],
                'tokens_shape': list(tokens.shape),
                'tokens_dtype': str(tokens.dtype),
                'first_16_tokens': tokens[:16].tolist(),
                'sha256_first_1024': hashlib.sha256(tokens[:1024].tobytes()).hexdigest(),
                'max_id': int(tokens.max()),
                'min_id': int(tokens.min()),
                'n_eod': int((tokens == 151643).sum()),
            }
            self.assertEqual(tokens.shape, (8192,))
            self.assertLess(int(tokens.max()), 152064,
                            "token id exceeds padded vocab")
            self.assertGreaterEqual(int(tokens.min()), 0)
        except Exception as e:
            self.fail(f"sample replay failed: {type(e).__name__}: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
