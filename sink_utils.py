import torch
from torch import nn

from timm.models.vision_transformer import Attention

from vitsinks import sink_heuristics as sh

@torch.no_grad()
def batched_bincount(x, dim, max_value):
    """
    Borrowed from https://discuss.pytorch.org/t/batched-bincount/72819/4,
    with minor modification.
    """
    target = torch.zeros(*x.shape[:-1], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


@torch.no_grad()
def update_seedkey_keepmap(querywise_argmin, k, glbl_to_keep_map, to_evict=3, heads_first=False, has_cls=True):
    """
    `querywise_argmin`: torch.Tensor[B, H, Nq] or torch.Tensor[B, Nq, H] -- Argmin for every query from attention map
    `keys`: torch.Tensor[B, H, Nq, D] or torch.Tensor[B, Nq, H, D] -- Actual keys from QKV
    `glbl_to_keep_map`: torch.Tensor[B, Nk_evict] -- Keepmap for context tokens
    `to_evict`: int -- How many context tokens to evict
    `heads_first`: bool -- True if all shapes have heads before sequence mode
    """
    if to_evict == 0:
        return glbl_to_keep_map

    # Implement for to_evict==1

    if heads_first:
        headmode_idx = 1
        seqmode_idx = 2
    else:
        headmode_idx = 2
        seqmode_idx = 1

    bincount = batched_bincount(querywise_argmin.flatten(1,2), dim=-1, max_value=k.shape[seqmode_idx])
    if has_cls:
        bincount = bincount[..., 1:]
    bincount_argsorted = torch.argsort(bincount, dim=-1, descending=True, stable=True)

    if has_cls:
        if heads_first:
            keys = k[..., 1:, :].mean(dim=headmode_idx)
        else:
            keys = k[..., 1:, :, :].mean(dim=headmode_idx)
    else:
        keys = k.mean(dim=headmode_idx) 

    seedkey_idx = bincount_argsorted[..., 0]
    restkey_idx = bincount_argsorted[..., 1:]
    seedkey_map = seedkey_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k.shape[-1])
    restkey_map = restkey_idx.unsqueeze(-1).expand(-1, -1, k.shape[-1])

    if torch.sum(seedkey_idx == 0) > 0:
        seedkey_idx = bincount_argsorted[..., 1]
        restkey_idx = torch.cat([bincount_argsorted[..., 0].unsqueeze(-1), bincount_argsorted[..., 2:]], dim=-1)
        seedkey_map = seedkey_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k.shape[-1])
        restkey_map = restkey_idx.unsqueeze(-1).expand(-1, -1, k.shape[-1])

    # restkey_map = delete_one(seedkey_idx, k.shape[-2]).unsqueeze(-1).expand(-1, -1, k.shape[-1])
    seedkey = torch.gather(keys, index=seedkey_map, dim=-2)
    restkey = torch.gather(keys, index=restkey_map, dim=-2)
    # impmap = torch.nn.functional.cosine_similarity(restkey, seedkey, dim=-1)
    impmap = restkey @ seedkey.transpose(-1, -2)
    impsort = torch.argsort(impmap[..., 0], dim=-1, descending=True, stable=True)

    if has_cls:
        impsort = torch.cat([torch.zeros(k.shape[0], 1, device=glbl_to_keep_map.device, dtype=glbl_to_keep_map.dtype), impsort], dim=-1)

    lcl_to_keep_map = torch.gather(restkey_idx, index=impsort[..., :-to_evict + 1], dim=-1)
    # lcl_to_keep_map = impsort[..., :-to_evict+1]

    # glbl_to_keep_map = torch.gather(lcl_to_keep_map, index=impsort, dim=-1)

    return lcl_to_keep_map


@torch.no_grad()
def get_topk_drop_mask(attn, k=5, to_evict=3, largest=False):
    _, to_keep_map = sh._topk_nothres(attn, k=k, to_evict=to_evict, largest=largest)
    return to_keep_map

@torch.no_grad()
def update_keepmap(attn_map, glbl_to_keep_map, k=5, to_evict=3, algorithm="topk", largest=False, has_cls=True):

    if int(to_evict) == 0:
        return glbl_to_keep_map

    if algorithm == "topk":
        func = sh._topk_nothres
    elif algorithm == "arithmetic_mean":
        func = sh._mean_colwise_thres
    elif algorithm == "geometric_mean":
        func = sh._geomean_colwise_thres
    else:
        raise ValueError()

    B, H, Nq, Nc = attn_map.shape

    if not float(to_evict).is_integer():
        to_evict = round(Nc * to_evict)
        assert Nc > to_evict, f"Cannot evict more context than what is present, {to_evict=}, {Nc=}!"
    to_evict = int(to_evict)

    _, lcl_to_keep_map = func(attn_map, k=k, to_evict=to_evict, largest=largest, has_cls=has_cls)
    
    # glbl_to_keep_map: [B, Nc], the global evict map will have indices from the original Nc.
    # lcl_to_keep_map: [B, Nc - to_evict], the local evict map will have indices in the Nc it had for the layer it was in.

    glbl_to_keep_map = torch.gather(glbl_to_keep_map, dim=-1, index=lcl_to_keep_map)
    Nc_evicted = glbl_to_keep_map.shape[1]    

    if has_cls:
        glbl_to_keep_map = torch.cat([torch.zeros(B, 1, device=glbl_to_keep_map.device, dtype=glbl_to_keep_map.dtype), glbl_to_keep_map], dim=-1)

    return glbl_to_keep_map

@torch.no_grad()
def bench_update_keepmap(attn_map, glbl_to_keep_map, k=5, to_evict=3, algorithm="topk", largest=False, has_cls=True, timing_dict=None, key_prefix=""):

    if int(to_evict) == 0:
        return glbl_to_keep_map, timing_dict

    if algorithm == "topk":
        func = sh._topk_nothres
    elif algorithm == "arithmetic_mean":
        func = sh._mean_colwise_thres
    elif algorithm == "geometric_mean":
        func = sh._geomean_colwise_thres
    else:
        raise ValueError()

    B, H, Nq, Nc = attn_map.shape

    if not float(to_evict).is_integer():
        to_evict = round(Nc * to_evict)
        assert Nc > to_evict, f"Cannot evict more context than what is present, {to_evict=}, {Nc=}!"
    to_evict = int(to_evict)
    if timing_dict is not None:
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()

    _, lcl_to_keep_map = func(attn_map, k=k, to_evict=to_evict, largest=largest, has_cls=has_cls)
    if timing_dict is not None:
        end.record()
        torch.cuda.synchronize()
        timing_dict[key_prefix + "sorting"] = start.elapsed_time(end)
    
    # glbl_to_keep_map: [B, Nc], the global evict map will have indices from the original Nc.
    # lcl_to_keep_map: [B, Nc - to_evict], the local evict map will have indices in the Nc it had for the layer it was in.
    if timing_dict is not None:
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()

    glbl_to_keep_map = torch.gather(glbl_to_keep_map, dim=-1, index=lcl_to_keep_map)
    Nc_evicted = glbl_to_keep_map.shape[1]    
    if timing_dict is not None:
        end.record()
        torch.cuda.synchronize()
        timing_dict[key_prefix + "gather"] = start.elapsed_time(end)
    
    if timing_dict is not None:
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()

    if has_cls:
        glbl_to_keep_map = torch.cat([torch.zeros(B, 1, device=glbl_to_keep_map.device, dtype=glbl_to_keep_map.dtype), glbl_to_keep_map], dim=-1)
    if timing_dict is not None:
        end.record()
        torch.cuda.synchronize()
        timing_dict[key_prefix + "cls_append"] = start.elapsed_time(end)

    return glbl_to_keep_map, timing_dict

@torch.no_grad()
def sort_x_importance(x, attn_map, to_evict=3, k=5, algorithm="topk", largest=False, has_cls=True, timing_dict=None, key_prefix=""):
    if int(to_evict) == 0:
        return x 

    if algorithm == "topk":
        func = sh._topk_nothres
    elif algorithm == "arithmetic_mean":
        func = sh._mean_colwise_thres
    elif algorithm == "geometric_mean":
        func = sh._geomean_colwise_thres
    else:
        raise ValueError()

    B, H, Nq, Nc = attn_map.shape

    if not float(to_evict).is_integer():
        to_evict = round(Nc * to_evict)
        assert Nc > to_evict, f"Cannot evict more context than what is present, {to_evict=}, {Nc=}!"
    to_evict = int(to_evict)
    if timing_dict is not None:
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()

    if timing_dict is not None and algorithm == "arithmetic_mean":
        func = sh._bench_mean_colwise_thres
        to_del_map, to_keep_map, timing_dict = func(attn_map, k=k, to_evict=to_evict, largest=largest, has_cls=has_cls, timing_dict=timing_dict, key_prefix=key_prefix)
    else:
        to_del_map, to_keep_map = func(attn_map, k=k, to_evict=to_evict, largest=largest, has_cls=has_cls)
    new_order = torch.cat([to_keep_map, to_del_map], axis=-1)

    if timing_dict is not None:
        end.record() # = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        timing_dict[key_prefix + "heuristic_sort"] = start.elapsed_time(end)

    if new_order.shape[-1] < x.shape[-1] - 1 and has_cls:
        added_token_idxs = torch.arange(start=new_order.shape[-1] + 1, end=x.shape[-2],
            device=new_order.device, dtype=new_order.dtype).unsqueeze(0).expand(B, -1)
        new_order = torch.cat([new_order, added_token_idxs], dim=-1)
        new_order = torch.cat([torch.zeros(B, 1, device=new_order.device, dtype=new_order.dtype), new_order], dim=-1)
    
    if new_order.shape[-1] < x.shape[-1] and not has_cls:
        added_token_idxs = torch.arange(start=new_order.shape[-1], end=x.shape[-2],
            device=new_order.device, dtype=new_order.dtype).unsqueeze(0).expand(B, -1)
        new_order = torch.cat([new_order, added_token_idxs], dim=-1)

    new_index = new_order.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    if timing_dict is not None:
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    
    with ProfilerContext(timing_dict=timing_dict, key="gather", key_prefix=key_prefix) as pc:
        x = torch.gather(x, dim=-2, index=new_index)
    timing_dict = pc()
    
    # if timing_dict is not None:
    #     end.record() # = torch.cuda.Event(enable_timing=True)
    #     torch.cuda.synchronize()
    #     timing_dict[key_prefix + "gather"] = start.elapsed_time(end)
    

    if timing_dict is not None:
        return x, timing_dict
    return x
    
@torch.no_grad()
def prune_x(x, glbl_to_keep_map):
    glbl_to_keep_index = glbl_to_keep_map.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    pruned_x = torch.gather(x, dim=-2, index=glbl_to_keep_index)
    return pruned_x
 

@torch.no_grad()
def split_qkv_weights_two(qkv_lin):
    out_feats, in_feats = qkv_lin.weight.data.shape
    new_out_feats = out_feats // 3
    has_bias = qkv_lin.bias is not None

    q = nn.Linear(in_feats, new_out_feats, bias=has_bias)
    kv = nn.Linear(in_feats, 2 * new_out_feats, bias=has_bias)

    q.weight.data = qkv_lin.weight.data[:new_out_feats]
    kv.weight.data = qkv_lin.weight.data[new_out_feats:]

    if has_bias:
        q.bias.data = qkv_lin.bias.data[:new_out_feats]
        kv.bias.data = qkv_lin.bias.data[new_out_feats:]
   
    return q, kv 

@torch.no_grad()
def split_qkv_weights(qkv_lin: torch.nn.Linear):
    out_feats, in_feats = qkv_lin.weight.data.shape
    new_out_feats = out_feats // 3
    has_bias = qkv_lin.bias is not None    

    q = nn.Linear(in_feats, new_out_feats, bias=has_bias)
    k = nn.Linear(in_feats, new_out_feats, bias=has_bias)
    v = nn.Linear(in_feats, new_out_feats, bias=has_bias)

    for idx, lin in enumerate([q, k, v]):

        lin.weight.data = qkv_lin.weight.data[idx * new_out_feats: (idx + 1) * new_out_feats]
        if has_bias:
            lin.bias.data = qkv_lin.bias.data[idx * new_out_feats: (idx + 1) * new_out_feats]

    return q, k, v


class ProfilerContext:
    def __init__(self, timing_dict, key, key_prefix=""):
        self.timing_dict = timing_dict
        if timing_dict is None:
            self.no_op = True
        else:
            self.no_op = False
        self.key = key
        self.key_prefix = key_prefix
        self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        assert isinstance(key, str) and isinstance(key_prefix, str), "`key` and `key_prefix` should be strings"

    def __enter__(self):
        if self.no_op:
            return lambda: None
        torch.cuda.synchronize()
        self.start.record()
        return lambda: self.timing_dict

    def __exit__(self, *args, **kwargs):
        if self.no_op:
            return False
        self.end.record()
        torch.cuda.synchronize()
        self.timing_dict[self.key_prefix + self.key] = self.start.elapsed_time(self.end)


