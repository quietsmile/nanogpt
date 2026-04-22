"""Parse train_bg.log for 'step N ... val loss V' lines and merge into nanogpt_train_log.json."""
import json, re, sys

LOG = sys.argv[1] if len(sys.argv) > 1 else '/tmp/train_bg.log'
JSONL = sys.argv[2] if len(sys.argv) > 2 else '/home/claudeuser/nanogpt/reports/nanogpt_train_log.jsonl'
OUT  = sys.argv[3] if len(sys.argv) > 3 else '/home/claudeuser/nanogpt/reports/nanogpt_train_log.json'

pat = re.compile(r'^step (\d+) \(samples ([\d,]+)\): train loss ([\d.]+), val loss ([\d.]+)')
val_points = []   # [(iter, val_loss, train_loss, samples)]
with open(LOG) as f:
    for line in f:
        m = pat.search(line)
        if m:
            iter_num = int(m.group(1))
            samples = int(m.group(2).replace(',', ''))
            train_l = float(m.group(3))
            val_l   = float(m.group(4))
            val_points.append({'iter': iter_num, 'val_loss': val_l,
                               'train_loss_at_eval': train_l, 'samples': samples})

all_entries = [json.loads(l) for l in open(JSONL) if l.strip()]
# train_log.jsonl is append-only across runs; keep only the LATEST run
# (identified by finding the last index where iter resets)
last_reset = 0
for i, e in enumerate(all_entries):
    if e['iter'] == 0 and i > 0:
        last_reset = i
entries = all_entries[last_reset:]
print(f"jsonl has {len(all_entries)} total entries; last-run starts at index {last_reset}, using {len(entries)} entries")
out = {
    'train_loss': [[e['iter'], e['loss']] for e in entries],
    'lr':         [[e['iter'], e['lr']]   for e in entries if 'lr' in e],
    'val_loss':   [[p['iter'], p['val_loss']] for p in val_points],
    'n_entries':  len(entries),
    'val_checkpoints': val_points,
}
if entries:
    out['first'] = entries[0]; out['last'] = entries[-1]
json.dump(out, open(OUT, 'w'), indent=1)
print(f"{len(entries)} train entries, {len(val_points)} val checkpoints → {OUT}")
