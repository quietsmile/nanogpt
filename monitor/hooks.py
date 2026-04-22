"""Forward-hook installation. All hooks are read-only: they .detach() outputs
and wrap bodies in torch.no_grad(), so they cannot affect forward numerics or
the autograd graph.

Hooks skip themselves during eval (no grad) to avoid polluting the current-step
frame with val-batch stats.
"""
import torch


def install(model, frame):
    """Register forward hooks on relevant modules of `model`.

    `frame` is a dict-like object owned by the Monitor — hook callbacks write
    into it under well-known keys. Monitor.step() reads + clears this dict once
    per training step.

    Returns a list of RemovableHandle so the caller can detach on shutdown.
    """
    from model import Block, MoERouter, MoEFFN
    handles = []

    # --- Final-residual (ln_f output) magnitude sentinel. We hook ln_f instead
    #     of lm_head because the training path calls F.linear(x.float(), ...)
    #     directly (model.py:831), bypassing self.lm_head.__call__ and hence
    #     any forward hook on it. ln_f output is pre-logit residual — same
    #     scale family — so its max/std are a faithful bf16-saturation proxy.
    ln_f = getattr(getattr(model, 'transformer', None), 'ln_f', None)
    if ln_f is not None:
        def ln_f_hook(module, inputs, output):
            if not torch.is_grad_enabled():
                return
            with torch.no_grad():
                o = output.detach()
                frame['final_resid_max'] = float(o.abs().max().item())
                frame['final_resid_std'] = float(o.float().std().item())
        handles.append(ln_f.register_forward_hook(ln_f_hook))

    # --- Each Block: pre/post residual norm, net contribution ratio ---
    blocks = getattr(getattr(model, 'transformer', None), 'h', None)
    if blocks is not None:
        for li, block in enumerate(blocks):
            def make_hook(layer_idx):
                def block_hook(module, inputs, output):
                    if not torch.is_grad_enabled():
                        return
                    with torch.no_grad():
                        x_pre = inputs[0].detach()
                        x_post = output[0].detach() if isinstance(output, tuple) else output.detach()
                        # mean L2 across (B*T) — divide by sqrt(d) for scale-free
                        d = x_pre.shape[-1]
                        pre = float(x_pre.float().norm(dim=-1).mean().item()) / (d ** 0.5)
                        post = float(x_post.float().norm(dim=-1).mean().item()) / (d ** 0.5)
                        delta = float((x_post - x_pre).float().norm(dim=-1).mean().item()) / (d ** 0.5)
                        frame.setdefault('block_res_pre', {})[layer_idx] = pre
                        frame.setdefault('block_res_post', {})[layer_idx] = post
                        # relative contribution: |Δ| / |pre|
                        frame.setdefault('block_contrib', {})[layer_idx] = delta / (pre + 1e-12)
                return block_hook
            handles.append(block.register_forward_hook(make_hook(li)))

    # --- MoERouter: accumulate per-expert load across micro-steps + capture
    #                score entropy / top-1 share on the last observed call ---
    moe_routers = {}  # layer_idx -> MoERouter module
    for name, m in model.named_modules():
        if not isinstance(m, MoERouter):
            continue
        parts = name.split('.')
        try:
            li = int(parts[parts.index('h') + 1])
        except (ValueError, IndexError):
            li = len(moe_routers)
        moe_routers[li] = m

        def make_router_hook(layer_idx):
            def router_hook(module, inputs, output):
                if not torch.is_grad_enabled():
                    return
                with torch.no_grad():
                    topk_idx, final_weights, scores = output
                    E = module.num_experts
                    # 1. Per-expert load accumulator (sum across micro-steps).
                    #    We cannot reuse module.local_tokens_per_expert — it is
                    #    zeroed by update_expert_bias() before monitor.step().
                    idx_flat = topk_idx.detach().reshape(-1).long()
                    ones = torch.ones_like(idx_flat, dtype=torch.float32)
                    load = torch.zeros(E, dtype=torch.float32, device=idx_flat.device)
                    load.scatter_add_(0, idx_flat, ones)
                    accum = frame.setdefault('_moe_load_accum', {})
                    if layer_idx in accum:
                        accum[layer_idx] = accum[layer_idx] + load
                    else:
                        accum[layer_idx] = load
                    # 2. Cheap per-call stats: entropy of sigmoid scores, top-1 weight share.
                    #    Sample at most 256 tokens for the entropy calc.
                    sample = scores.detach()[:256].float()
                    # normalize per row so we can take entropy
                    p = sample / (sample.sum(dim=-1, keepdim=True) + 1e-20)
                    ent = float(-(p * (p + 1e-20).log()).sum(dim=-1).mean().item())
                    frame.setdefault('moe_score_entropy', {})[layer_idx] = ent
                    fw = final_weights.detach().float()
                    frame.setdefault('moe_top1_share', {})[layer_idx] = float(fw[:, 0].mean().item())
            return router_hook
        handles.append(m.register_forward_hook(make_router_hook(li)))

    # --- MoEFFN: shared-expert vs routed-expert output-norm ratio (best-effort,
    #             only if shared_expert exists; uses module output) ---
    # The MoEFFN returns a single fused tensor, so we can't split post-hoc.
    # Skip for Phase-1. D7 will come via instrumentation inside MoEFFN in Phase-2.

    return handles, moe_routers
