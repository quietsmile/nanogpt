"""Classify a parameter by name into a semantic group.

Groups are ordered from most-specific to least-specific; first match wins.
Keep groups stable across runs so cross-scale comparisons are meaningful.
"""

# (group_name, list of substrings; any match -> that group)
_RULES = [
    ('embedding',      ['transformer.wte', 'transformer.wpe']),
    ('lm_head',        ['lm_head']),
    ('norm',           ['.ln_1', '.ln_2', '.ln_f', '.norm',
                        'q_layernorm', 'k_layernorm']),
    ('router',         ['router.linear']),
    ('shared_expert',  ['shared_expert']),
    # MoE routed-expert tensors in MoEFFN are registered as moe_w_gate/up/down
    ('routed_expert',  ['moe_w_gate', 'moe_w_up', 'moe_w_down']),
    ('attn_qkv',       ['q_proj', 'k_proj', 'v_proj']),
    ('attn_proj',      ['attn.c_proj', 'attn.proj', 'out_proj']),
    ('ffn_gate',       ['gate_proj']),
    ('ffn_up',         ['up_proj']),
    ('ffn_down',       ['down_proj']),
]


def classify(pname: str) -> str:
    for group, needles in _RULES:
        for n in needles:
            if n in pname:
                return group
    return 'other'


def all_groups():
    return [g for g, _ in _RULES] + ['other']
