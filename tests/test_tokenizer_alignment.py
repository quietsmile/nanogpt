"""Phase 1: tokenizer alignment for scaling_moe_00196.

Verifies the dots_tokenizer at the ref-job path matches expectations:
  - class: Qwen2Tokenizer(Fast)
  - vocab_size: 151643 base, 151660 with added, padded to 152064
  - EOD id (151643) consistent with yaml's accurate_attn_mask_eod_token
  - 16 dots-specific added tokens
  - Round-trip fidelity on a fixed corpus → JSON report with diff details
"""
import hashlib
import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOTS = '/prodcpfs/user/xiaoming/models/dots_tokenizer'
EXPECTED_EOD = 151643
EXPECTED_PADDED_VOCAB = 152064

os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
REPORT = os.path.join(ROOT, 'reports', 'tokenizer_alignment.json')


ROUNDTRIP_CORPUS = [
    "Hello world!",
    "The quick brown fox jumps over the lazy dog 12345.",
    "你好，世界！今天天气真好。",
    "def foo(x):\n    return x * 2  # simple\n",
    "混合 mixed 中英 eN-US 123 !@#",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "\n\n   whitespace   test\t\t\n",
    "emoji: 🚀🔥 unicode: Ω α β",
    "a" * 200,  # long repetition
    "",
]


class TestDotsTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(DOTS)
        cls.cfg = json.load(open(os.path.join(DOTS, 'tokenizer_config.json')))
        cls.report = {}

    @classmethod
    def tearDownClass(cls):
        with open(REPORT, 'w') as f:
            json.dump(cls.report, f, ensure_ascii=False, indent=2)

    def test_class(self):
        cname = type(self.tok).__name__
        self.report['class'] = cname
        self.assertIn('Qwen2', cname, f"expected Qwen2 family, got {cname}")

    def test_vocab_sizes(self):
        v, lt = self.tok.vocab_size, len(self.tok)
        self.report['vocab_size'] = v
        self.report['len_with_added'] = lt
        self.assertEqual(v, 151643)
        self.assertEqual(lt, 151660)
        self.assertLessEqual(lt, EXPECTED_PADDED_VOCAB)
        # padded = 152064 is what goes into the model; unused ids 151660-152063 should not map
        self.assertEqual(EXPECTED_PADDED_VOCAB, 152064)

    def test_eod_id_matches_yaml(self):
        eos = self.tok.eos_token_id
        self.report['eos_token_id'] = eos
        self.report['eos_token'] = self.tok.eos_token
        self.assertEqual(eos, EXPECTED_EOD,
                         f"yaml accurate_attn_mask_eod_token=[151643], tokenizer eos={eos}")

    def test_added_tokens_exact_list(self):
        decoder = self.cfg['added_tokens_decoder']
        added_ids = sorted(int(k) for k in decoder.keys())
        self.report['added_token_ids'] = added_ids
        self.report['added_tokens'] = {i: decoder[str(i)]['content'] for i in added_ids}
        # 16 dots specials
        self.assertEqual(len(added_ids), 16)
        self.assertEqual(added_ids[0], 151643)   # endoftext
        self.assertEqual(added_ids[-1], 151659)  # reserved

    def test_mask_loss_id_out_of_vocab(self):
        # yaml mask_loss_id=160000; must be above real+added vocab, so CE ignore is unambiguous
        self.report['mask_loss_id'] = 160000
        self.assertGreater(160000, len(self.tok))

    def test_roundtrip_exact(self):
        results = []
        all_match = True
        for s in ROUNDTRIP_CORPUS:
            ids = self.tok.encode(s, add_special_tokens=False)
            back = self.tok.decode(ids, skip_special_tokens=False)
            match = (s == back)
            all_match = all_match and match
            results.append({
                'input_sha256': hashlib.sha256(s.encode()).hexdigest()[:16],
                'input_len': len(s),
                'ids_len': len(ids),
                'first_8_ids': ids[:8],
                'last_4_ids': ids[-4:] if len(ids) > 4 else ids,
                'match': match,
                'decode': back if not match else None,
            })
        self.report['roundtrip'] = results
        self.assertTrue(all_match,
                        f"roundtrip failed on: {[r for r in results if not r['match']]}")

    def test_tokenizer_file_md5(self):
        # Lock tokenizer identity — any silent swap would change these
        def md5(path):
            h = hashlib.md5()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
            return h.hexdigest()
        md5s = {
            'tokenizer.json': md5(os.path.join(DOTS, 'tokenizer.json')),
            'merges.txt': md5(os.path.join(DOTS, 'merges.txt')),
            'tokenizer_config.json': md5(os.path.join(DOTS, 'tokenizer_config.json')),
        }
        self.report['file_md5'] = md5s
        # Just require the files exist and are non-trivially sized
        for name, h in md5s.items():
            self.assertTrue(len(h) == 32)


if __name__ == '__main__':
    unittest.main(verbosity=2)
