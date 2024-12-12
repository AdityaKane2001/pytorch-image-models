import torch
from torch import nn

from timm.models.vision_transformer import Attention

from vitsinks import sink_heuristics as sh

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
def sort_x_importance(x, attn_map, to_evict=3, k=5, algorithm="topk", largest=False, has_cls=True):
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

    to_del_map, to_keep_map = func(attn_map, k=k, to_evict=to_evict, largest=largest, has_cls=has_cls)
    new_order = torch.cat([to_keep_map, to_del_map], axis=-1)


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

    x = torch.gather(x, dim=-2, index=new_index)

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

