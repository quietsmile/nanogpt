"""Monitor + NullMonitor.

Monitor.step() is called once per optim step, AFTER scaler.update() (so optim
state is current) and BEFORE optimizer.zero_grad() (so .grad is still on every
parameter). It captures a single JSONL record per training step.

On non-master ranks the Monitor still participates in any DDP all_reduce ops
but skips the disk write. This keeps collectives balanced across ranks.
"""
import json
import math
import os
from collections import deque

import torch


class NullMonitor:
    def step(self, *a, **kw):
        pass

    def close(self):
        pass

    def register_probe(self, *a, **kw):
        pass

    def maybe_probe(self, *a, **kw):
        pass

    def attach(self, *a, **kw):
        pass

    def detach(self, *a, **kw):
        pass


class Monitor:
    def __init__(self, model, optimizer, out_dir, master=True, ddp=False):
        self.model = model
        self.optimizer = optimizer
        self.master = master
        self.ddp = ddp
        self.out_dir = out_dir

        # cadence knobs (all env-overridable)
        self.m_interval = int(os.environ.get('NANOGPT_MONITOR_M', '50'))
        # probe_interval: run attention probe every N optim steps (0 = off).
        # The probe runs a full eval-mode forward on a fixed batch of tokens,
        # so it's expensive — default to 0 and let the user enable it explicitly
        # (or set via train.py near the eval loop).
        self.probe_interval = int(os.environ.get('NANOGPT_MONITOR_PROBE_INTERVAL', '0'))
        self.verbose = os.environ.get('NANOGPT_MONITOR_VERBOSE', '0') == '1'
        # probe state (populated by register_probe)
        self._probe_tokens = None
        self._probe_snapshots = []

        # running state
        self._loss_ema = None        # (mean, var) exponential moving
        self._ema_alpha = 0.05
        self._last_loss = None
        self._last_samples = None
        self._sample_loss_hist = deque(maxlen=50)  # (samples, loss) pairs

        # frame: populated by forward hooks, drained + cleared each step
        self._frame = {}
        from .hooks import install
        self._hook_handles, self._moe_routers = install(model, self._frame)

        # writer
        self._fp = None
        if self.master:
            log_path = os.path.join(out_dir, 'monitor.jsonl')
            os.makedirs(out_dir, exist_ok=True)
            self._fp = open(log_path, 'a', buffering=1)   # line-buffered
            if self.verbose:
                print(f"[monitor] enabled, writing {log_path} "
                      f"(m_interval={self.m_interval})")

    # ------------------------------------------------------------------ #
    # Per-step collection
    # ------------------------------------------------------------------ #
    def step(self, iter_num, loss, grad_norm, lr, samples=None):
        """Must be called once per optim step, before optimizer.zero_grad()."""
        # Fast-path guard so grad-norm / optim reads happen while grads exist.
        record = {
            'iter': int(iter_num),
            'loss': _scalar(loss),
            'lr': float(lr),
            'grad_norm': _scalar(grad_norm),
        }
        if samples is not None:
            record['samples'] = int(samples)

        # 1. Loss spike: z-score against EMA(mean, var)
        l = record['loss']
        ok = math.isfinite(l)
        if not ok:
            record['nan_inf'] = True
        if ok:
            if self._loss_ema is None:
                self._loss_ema = (l, 1.0)
            else:
                mu, var = self._loss_ema
                a = self._ema_alpha
                mu_new = (1 - a) * mu + a * l
                var_new = (1 - a) * var + a * (l - mu) ** 2
                z = (l - mu) / (max(var, 1e-12) ** 0.5)
                self._loss_ema = (mu_new, var_new)
                record['loss_z'] = float(z)

        # 2. Δloss/Δsamples (token efficiency)
        if samples is not None and self._last_loss is not None and ok:
            ds = samples - self._last_samples
            if ds > 0:
                record['dloss_dsamples'] = float((l - self._last_loss) / ds)
        if ok:
            self._last_loss = l
            self._last_samples = samples

        # 3. grad_norm by parameter group (must run before zero_grad)
        record['gn_by_group'] = _grad_norm_by_group(self.model)

        # 4. grad_norm / loss ratio (scale-invariant stability proxy)
        if ok and record['grad_norm'] > 0:
            record['gn_over_loss'] = float(record['grad_norm'] / max(l, 1e-8))

        # 5. Adam second-moment percentiles (p50/p99 per group)
        record['adam_v'] = _adam_v_stats(self.optimizer)

        # 6. MoE: load + routing stats (may need DDP reduce)
        moe = self._drain_moe()
        if moe:
            record['moe'] = moe

        # 7. final residual stream + block residuals (cheap hook outputs)
        for k in ('final_resid_max', 'final_resid_std'):
            if k in self._frame:
                record[k] = self._frame[k]
        if int(iter_num) % self.m_interval == 0:
            for k in ('block_res_pre', 'block_res_post', 'block_contrib'):
                if k in self._frame:
                    # hook stores {layer_idx: val} — serialize as sorted list
                    v = self._frame[k]
                    record[k] = [v[i] for i in sorted(v)]

        # 8. clear the frame (but keep accumulator keys already drained)
        self._frame.clear()

        # 9. write (master only)
        if self._fp is not None:
            self._fp.write(json.dumps(record, default=_jsonable) + '\n')

    # ------------------------------------------------------------------ #
    def _drain_moe(self):
        """Build per-layer MoE stats: load entropy, dead experts, bias stats,
        router score entropy / top1 share. Performs one all_reduce(sum) over
        per-expert token counts when ddp=True so counts reflect the global
        batch, matching how the router bias is updated."""
        if not self._moe_routers:
            return None
        accum = self._frame.get('_moe_load_accum')
        if not accum:
            return None

        # Stack per-layer loads so we can all_reduce once (fewer collectives).
        layers = sorted(accum.keys())
        stacked = torch.stack([accum[li] for li in layers], dim=0)   # [L, E]
        if self.ddp:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stacked, op=dist.ReduceOp.SUM)

        out = {}
        score_ent = self._frame.get('moe_score_entropy', {})
        top1_sh = self._frame.get('moe_top1_share', {})
        for row, li in enumerate(layers):
            counts = stacked[row].to('cpu')
            total = float(counts.sum().item())
            E = counts.numel()
            if total > 0:
                p = counts / total
                load_ent = float(-(p * (p + 1e-20).log()).sum().item())
                # normalize entropy to [0, 1]: divide by log(E)
                load_ent_norm = load_ent / math.log(E) if E > 1 else 0.0
                # gini (simple)
                gini = _gini(counts.numpy())
                dead = int((counts == 0).sum().item())
                near_dead = int((counts / (total / E) < 0.1).sum().item())
            else:
                load_ent = load_ent_norm = gini = 0.0
                dead = near_dead = E
            entry = {
                'load_entropy_norm': load_ent_norm,
                'load_gini': gini,
                'dead': dead,
                'near_dead': near_dead,
                'tokens_routed': int(total),
            }
            if li in score_ent:
                entry['score_entropy'] = score_ent[li]
            if li in top1_sh:
                entry['top1_share'] = top1_sh[li]
            # bias stats from the router buffer (read-only)
            m = self._moe_routers[li]
            b = m.e_score_correction_bias.detach()
            entry['bias_max'] = float(b.abs().max().item())
            entry['bias_std'] = float(b.float().std().item())
            out[li] = entry
        return out

    # ------------------------------------------------------------------ #
    # Attention probe integration
    # ------------------------------------------------------------------ #
    def register_probe(self, tokens):
        """Register a fixed token tensor to use for the attention probe.

        Call once after model init (train.py). The same tokens are passed to
        every probe invocation so snapshots across iterations are directly
        comparable. Accepts a torch tensor on any device — will be moved to
        the model's device at probe time.
        """
        self._probe_tokens = tokens.detach().clone()

    def maybe_probe(self, iter_num):
        """Run attention probe if iter_num matches probe_interval and tokens
        are registered. Safe to call from train.py eval loop unconditionally —
        it no-ops when the interval is 0 or tokens not set."""
        if self.probe_interval <= 0 or self._probe_tokens is None:
            return
        if int(iter_num) % self.probe_interval != 0:
            return
        if not self.master:
            return
        try:
            from .attn_probe import probe_attention
            device = next(self.model.parameters()).device
            tokens = self._probe_tokens.to(device)
            snap = probe_attention(self.model, tokens, downsample=128,
                                   iter_tag=int(iter_num))
            self._probe_snapshots.append(snap)
            # Persist incrementally — overwrite a full snapshots-list file so
            # a crash mid-run still leaves the prior snapshots recoverable.
            path = os.path.join(self.out_dir, 'attention_maps.json')
            with open(path, 'w') as f:
                json.dump({'snapshots': self._probe_snapshots}, f)
            if self.verbose:
                print(f"[monitor] attn probe @ iter {iter_num}: "
                      f"{snap['n_layer']}L × {snap['n_head']}H, "
                      f"max sink={max(m['sink_strength'] for m in snap['metrics_per_layer']):.3f}")
        except Exception as e:
            if self.verbose:
                print(f"[monitor] attn probe failed @ iter {iter_num}: {e}")

    def close(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _scalar(x):
    if hasattr(x, 'item'):
        try:
            return float(x.item())
        except Exception:
            return float('nan')
    try:
        return float(x)
    except Exception:
        return float('nan')


def _jsonable(o):
    if hasattr(o, 'item'):
        return o.item()
    return str(o)


def _grad_norm_by_group(model):
    from .param_groups import classify
    acc = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n2 = float(g.float().pow(2).sum().item())
        group = classify(name)
        acc[group] = acc.get(group, 0.0) + n2
    return {g: math.sqrt(v) for g, v in acc.items()}


def _adam_v_stats(optimizer):
    # Per-group p50/p99 of Adam's exp_avg_sq (second moment).
    # Returns a dict keyed by the model's semantic param_group classifier when
    # we can map the param back by id; fall back to a single global view.
    vals = []
    try:
        for pgroup in optimizer.param_groups:
            for p in pgroup['params']:
                st = optimizer.state.get(p)
                if st is None or 'exp_avg_sq' not in st:
                    continue
                v = st['exp_avg_sq'].detach()
                # cheap: take log-mean of a sample so percentile computation is bounded
                flat = v.flatten()
                if flat.numel() > 4096:
                    # subsample with stride (deterministic; no RNG)
                    stride = flat.numel() // 4096
                    flat = flat[::stride][:4096]
                vals.append(flat.float())
    except Exception:
        return {}
    if not vals:
        return {}
    cat = torch.cat(vals)
    # quantile requires q tensor on same device as input — keep it portable.
    q = torch.quantile(cat, torch.tensor([0.5, 0.99], device=cat.device))
    return {'p50': float(q[0].item()), 'p99': float(q[1].item())}


def _gini(arr):
    import numpy as np
    a = np.asarray(arr, dtype='float64').flatten()
    if a.sum() == 0:
        return 0.0
    a = np.sort(a)
    n = a.size
    cum = np.cumsum(a)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)
