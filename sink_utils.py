import torch
from torch import nn

from timm.models.vision_transformer import Attention

from vitsinks import sink_heuristics as sh


@torch.no_grad()
def get_topk_drop_mask(attn, k=5, to_evict=3, largest=False):
    _, to_keep_map = sh._topk_nothres(attn, k=k, to_evict=to_evict, largest=largest)
    return to_keep_map

@torch.no_grad()
def update_keepmap(attn_map, glbl_to_keep_map, k=5, to_evict=3, largest=False, has_cls=True):

    if to_evict == 0:
        return glbl_to_keep_map

    B, H, Nq, Nc = attn_map.shape
    _, lcl_to_keep_map = sh._topk_nothres(attn_map, k=k, to_evict=to_evict, largest=largest, has_cls=has_cls)
    
    # glbl_to_keep_map: [B, Nc], the global evict map will have indices from the original Nc.
    # lcl_to_keep_map: [B, Nc - to_evict], the local evict map will have indices in the Nc it had for the layer it was in.

    glbl_to_keep_map = torch.gather(glbl_to_keep_map, dim=-1, index=lcl_to_keep_map)
    Nc_evicted = glbl_to_keep_map.shape[1]    

    if has_cls:
        glbl_to_keep_map = torch.cat([torch.zeros(B, 1, device=glbl_to_keep_map.device, dtype=glbl_to_keep_map.dtype), glbl_to_keep_map], dim=-1)

    return glbl_to_keep_map


@torch.no_grad()
def prune_x(x, glbl_to_keep_map):
    glbl_to_keep_index = glbl_to_keep_map.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    pruned_x = torch.gather(x, dim=-2, index=glbl_to_keep_index)
    return pruned_x
    

def get_eviction_schedule(start_layer=0, end_layer=1, num_layers=10, to_evict=3, slope=1):
    """
    For every entry in `to_evicts`:
    <0: Use full context.
    =0: Use same context as last layer (maybe partially evicted).
    >0: Evict more context from last layer.
    """

    to_evicts = [-1 for _ in range(num_layers)]

    assert start_layer <= end_layer and end_layer < num_layers

    for layer_idx in range(num_layers):
        if layer_idx >= start_layer and layer_idx < end_layer:
            to_evicts[layer_idx] = to_evict

    return to_evicts

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

@torch.no_grad()
def replace_qkv_with_unbound(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, Attention) and hasattr(module, "qkv"):
            q, k, v = split_qkv_weights(module.qkv)
            module.q = q
            module.k = k
            module.v = v
    return model 
