# from dill import pickle as dill_pickle
import copyreg
import hashlib
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy, deepcopy

import dill
import dill.settings
import torch
import torch.distributed

from cybertron.config.config import CybertronArguments

# torch._logging._internal.set_logs(autograd=logging.DEBUG)


class NoneReducer:
    @staticmethod
    def reduce(pg):
        return (NoneReducer.rebuild, (None,))

    @staticmethod
    def rebuild(state):
        return None


copyreg.pickle(torch._C._distributed_c10d.ProcessGroup, NoneReducer.reduce)
copyreg.pickle(torch.cuda.Stream, NoneReducer.reduce)
copyreg.pickle(torch._C._functions.AccumulateGrad, NoneReducer.reduce)
copyreg.pickle(torch.cuda.Event, NoneReducer.reduce)
copyreg.pickle(torch._C.DispatchKeySet, NoneReducer.reduce)


# dill.detect.trace(True)
dill.settings["byref"] = True
dill.settings["recurse"] = True
dill.settings["ignore"] = True


def get_children_layers(model: torch.nn.Module, name=""):
    named_children = dict(model.named_children())
    children = named_children.values()
    names = named_children.keys()
    if len(children) == 0:
        output_names, output_children = [name], [model]
    else:
        output_names, output_children = [name], [model]
        for n, c in zip(names, children):
            res_n, res_c = get_children_layers(c, n)
            for ni, ci in zip(res_n, res_c):
                full_name = f"{name}.{ni}" if name != "" else ni
                output_names.append(full_name)
                output_children.append(ci)

    return output_names, output_children


def remove_attrs(m, names):
    for n in names:
        if hasattr(m, n):
            delattr(m, n)


def remove_hook_from_module(module: torch.nn.Module, recurse=False):
    # module._forward_hooks.clear()
    # module._forward_pre_hooks.clear()
    # module._forward_pre_hooks_with_kwargs.clear()
    # module._forward_hooks_with_kwargs.clear()
    # module._backward_hooks.clear()
    # module._backward_pre_hooks.clear()

    # if hasattr(module, "_parameters"):
    #     module._parameters = dict(module._parameters)

    if hasattr(module, "_old_forward"):
        module.forward = module._old_forward
        module.forward.__self__.forward = module._old_forward
        delattr(module, "_old_forward")

    # remove_names = ["_hf_hook", "hf_device_map"]
    # remove_attrs(module, remove_names)

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)


mbs_ids = {}


def is_last_rank():
    # MODIFIED: dump from all DP ranks on ep=ep_size-1 to capture all 64 samples
    # per iter instead of just 8. With tp=pp=1, ep=EP-1 is the last EP group;
    # we still only capture one EP group but across all DP ranks.
    try:
        from megatron.core import mpu
        ep = mpu.get_expert_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
        tp = mpu.get_tensor_model_parallel_rank()
        pp = mpu.get_pipeline_model_parallel_rank()
        return (ep == ep_size - 1) and (tp == 0) and (pp == 0)
    except Exception:
        return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def md5_hash_tensor(t):
    if t is None:
        return "None"
    return hashlib.md5(t.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


def hook_fwd_bwd_to_module(
    args: CybertronArguments, model: torch.nn.Module, names=None, prefix="", is_hf=False
):
    def name_fn(name, direction="forward", is_hf=False):
        def fn(module, input_features, output_features):
            from megatron.core import mpu

            tp = mpu.get_tensor_model_parallel_rank()
            pp = mpu.get_pipeline_model_parallel_rank()
            ep = mpu.get_expert_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            ep_size = mpu.get_expert_model_parallel_world_size()
            # MODIFIED: also grab DP rank so dumps from multiple DP ranks don't collide
            try:
                dp = mpu.get_data_parallel_rank()
                dp_size = mpu.get_data_parallel_world_size()
            except Exception:
                dp = 0
                dp_size = 1
            # module_copy = copy(module)
            # remove_hook_from_module(module_copy, recurse=True)

            # flag = torch.distributed.get_rank() == 0 if is_hf else pp == 0
            flag = True
            node = torch._C._current_autograd_node()
            save_md5 = bool(int(os.environ.get("CYBERTRON_BITWISE_COMPARE_MD5", 0)))
            if flag and name is not None and name != "" and name != " " and is_last_rank():
                print(f"===== dump {name} datas {node=}===== {tp=} {pp=} {ep=}")
                if prefix and not os.path.exists(prefix):
                    os.makedirs(prefix, exist_ok=True)

                iteration = args.curr_iteration
                key = (name, direction)
                mbs_ids.setdefault(key, 0)
                # with tensor_reduce_context():
                #     torch.save(module_copy, f"{prefix}{name}-iter{iteration}-{direction}-module-tp{tp}-pp{pp}-ep{ep}.pt", pickle_module=dill)

                print(
                    f"{prefix}{name}-iter{iteration}-mbs{mbs_ids[key]}-{direction}-input-tp{tp}.{tp_size}-pp{pp}.{pp_size}-ep{ep}.{ep_size}-dp{dp}.{dp_size}.md5"
                )
                if save_md5:
                    with open(
                        f"{prefix}{name}-iter{iteration}-mbs{mbs_ids[key]}-{direction}-input-tp{tp}.{tp_size}-pp{pp}.{pp_size}-ep{ep}.{ep_size}-dp{dp}.{dp_size}.md5",
                        "w",
                    ) as writer:
                        for feature in input_features:
                            writer.write(md5_hash_tensor(feature) + "\n")
                    with open(
                        f"{prefix}{name}-iter{iteration}-mbs{mbs_ids[key]}-{direction}-output-tp{tp}.{tp_size}-pp{pp}.{pp_size}-ep{ep}.{ep_size}-dp{dp}.{dp_size}.md5",
                        "w",
                    ) as writer:
                        for feature in output_features:
                            writer.write(md5_hash_tensor(feature) + "\n")
                else:
                    torch.save(
                        input_features,
                        f"{prefix}{name}-iter{iteration}-mbs{mbs_ids[key]}-{direction}-input-tp{tp}.{tp_size}-pp{pp}.{pp_size}-ep{ep}.{ep_size}-dp{dp}.{dp_size}.pt",
                        pickle_module=dill,
                    )
                    torch.save(
                        output_features,
                        f"{prefix}{name}-iter{iteration}-mbs{mbs_ids[key]}-{direction}-output-tp{tp}.{tp_size}-pp{pp}.{pp_size}-ep{ep}.{ep_size}-dp{dp}.{dp_size}.pt",
                        pickle_module=dill,
                    )
                mbs_ids[key] += 1

            # torch.distributed.barrier()

            # if direction == "backward":
            #     return input_features, output_features

            # for param_name, param in module.named_parameters():
            #     torch.save(param, f"{prefix}{name}_param_{param_name}_{rank}.pt")

        return fn

    if isinstance(names, str):
        names = [names]

    all_names, _ = get_children_layers(model)

    new_names = []
    if names is None:
        new_names = all_names
    else:
        for n in all_names:
            for t in names:
                if t.endswith("*"):
                    if n.startswith(t[:-1]):
                        new_names.append(n)
                    if n == t[:-2]:
                        new_names.append(n)
                else:
                    if n == t:
                        new_names.append(n)

    # model.register_forward_hook(name_fn("model"))
    modules = dict(model.named_modules())
    for name in new_names:
        if name in modules.keys():
            modules[name].register_forward_hook(name_fn(name, is_hf=is_hf))
            modules[name].register_full_backward_hook(
                name_fn(name, "backward", is_hf=is_hf), prepend=True
            )

    # # hook params
    # def _make_param_hook(
    #     name,
    #     param: torch.nn.Parameter,
    # ):
    #     """
    #     Creates the all-reduce / reduce-scatter hook for backprop.
    #     """

    #     def param_hook(*unused):

    #         if param.requires_grad:
    #             if param.grad is not None:
    #                 rank = torch.distributed.get_rank()
    #                 file_name = f"{prefix}{name}_grad_{rank}.pt"
    #                 param_copy = param.detach().clone()
    #                 torch.save(param_copy, file_name)

    #     return param_hook

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # Expand so we get access to grad_fn.
    #         param_tmp = param.expand_as(param)
    #         # Get the gradient accumulator function.
    #         grad_acc = param_tmp.grad_fn.next_functions[0][0]
    #         grad_acc.register_hook(_make_param_hook(name, param))
