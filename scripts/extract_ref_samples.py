"""Extract the 4 samples that ref's rank-7 mbs captured, by bf16-bitwise embedding match.
Embedding is a lookup: each row = unique vocab id. So bf16-exact row match reverses it."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megatron_to_nano import load_all_megatron_shards


def main():
    dump_dir = '/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps'
    ckpt_dir = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988'

    print('Loading embedding weight...', flush=True)
    meg = load_all_megatron_shards(ckpt_dir)
    embed_w = meg['embedding.word_embeddings.weight']  # [V, H] bf16
    V, H = embed_w.shape
    print(f'  shape=[{V},{H}]', flush=True)

    # Hash each vocab row's raw bytes → vocab id
    print('Building hash map for embedding rows...', flush=True)
    row_to_id = {}
    w_bytes = embed_w.contiguous().view(torch.uint8).view(V, H * 2)  # bf16 = 2 bytes
    for v in range(V):
        key = bytes(w_bytes[v].tolist())
        row_to_id[key] = v
    print(f'  {len(row_to_id)} unique rows out of {V}', flush=True)

    for mbs in [0, 1]:
        ref = torch.load(f'{dump_dir}/embedding-iter5988-mbs{mbs}-forward-output-tp0.1-pp0.1-ep3.4.pt',
                         weights_only=False, map_location='cpu')
        print(f'mbs {mbs}: ref shape {ref.shape}', flush=True)
        S, B, Href = ref.shape
        assert Href == H
        ref_bytes = ref.contiguous().view(torch.uint8).view(S, B, H * 2)
        for b in range(B):
            tok_ids = []
            missing = 0
            for s in range(S):
                key = bytes(ref_bytes[s, b].tolist())
                tid = row_to_id.get(key)
                if tid is None:
                    missing += 1
                    tid = -1
                tok_ids.append(tid)
            out = f'/newcpfs/user/yuchen/karpathy/cybertron_dump/sample_mbs{mbs}_b{b}.pt'
            torch.save({'tokens': torch.tensor(tok_ids, dtype=torch.long), 'mbs': mbs, 'b': b}, out)
            first10 = tok_ids[:10]
            print(f'  mbs{mbs} b{b}: {len(tok_ids)} tokens, {missing} missing, first10={first10}', flush=True)


if __name__ == '__main__':
    main()
