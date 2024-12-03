import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models.vision_transformer import VisionTransformer, Block, Attention

from sink_utils import update_keepmap, prune_x, split_qkv_weights, split_qkv_weights_two

@torch.no_grad()
def replace_qkv_with_unbound(model: torch.nn.Module, num_gemms=3):
    for name, module in model.named_modules():
        if isinstance(module, Attention) and hasattr(module, "qkv"):
            if num_gemms == 3:
                q, k, v = split_qkv_weights(module.qkv)
                module.q = q
                module.k = k
                module.v = v
                module.num_gemms = 3
            elif num_gemms == 2:
                q, kv = split_qkv_weights_two(module.qkv)
                module.q = q
                module.kv = kv
                module.num_gemms = 2
            elif num_gemms == 1:
                module.num_gemms = 1
            else:
                raise AttributeError("num_gemms should be 1,2 or 3")
    return model 

class UnboundQKVAttention(Attention):   
    def forward(self,x):
        B, N, C = x.shape
        if self.num_gemms == 1:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

        elif self.num_gemms == 2:
            q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(x).view(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

        elif self.num_gemms == 3:
            q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        else:
            raise AttributeError("either of one,two, three gemm should be true")
        
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            ##############################################
            attn = self.attn_logits_identity(attn)
            ##############################################
            
            attn = attn.softmax(dim=-1)
            ##############################################
            attn = self.attn_map_identity(attn)
            ##############################################
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SinkAttention(Attention):
    def check_patching(self):
        has_q = hasattr(self, "q")
        has_k = hasattr(self, "k")
        has_v = hasattr(self, "v")
        has_kv = hasattr(self, "kv")
        has_qkv_unbound = has_q and ((has_k and has_v) or has_kv)

        cfg_keys = list(self.eviction_config.keys())
        has_cls = "has_cls" in cfg_keys
        has_evict_k = "k" in cfg_keys
        has_to_evict = "to_evict" in cfg_keys
        has_largest = "largest" in cfg_keys
        has_algorithm = "algorithm" in cfg_keys
        has_pruning_params = has_cls and has_evict_k and has_to_evict and has_largest and has_algorithm
        
        return has_qkv_unbound, has_k and has_v, has_kv, has_pruning_params

    def check_eviction_cfg(self):
        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config
        
        if not has_pruning_params:
            raise ValueError("Patching not done properly!")

        assert has_qkv_unbound, "QKV not unbound!"

        if ecfg["to_evict"] == 0 and to_keep_map is None:
            raise ValueError("Got self.to_evict > 0, but to_keep_map was not supplied.")

        if ecfg["to_evict"] > 0:
            if past_attn is None:
                raise ValueError("Got self.to_evict > 0, but past_attn was not supplied.")
            elif to_keep_map is None:
                raise ValueError("Got self.to_evict > 0, but to_keep_map was not supplied.")

        if not (has_kandv or has_kv):
            raise AttributeError(f"{self.__class__} instance should have k and v, or kv.")         
            

    def forward(self, x: torch.Tensor, to_keep_map: torch.Tensor = None, past_attn: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        self.check_eviction_cfg()

        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config

        if ecfg["to_evict"] >= 0:
        
            to_keep_map = update_keepmap(past_attn, to_keep_map, k=ecfg["k"], to_evict=ecfg["to_evict"], 
                 largest=ecfg["largest"], has_cls=ecfg["has_cls"], algorithm=ecfg["algorithm"])
            
            pruned_x = prune_x(x, to_keep_map) 
            

            q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            if has_kandv: 
                k = self.k(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = self.v(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            elif has_kv:
                kv = self.kv(x).view(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
        
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            to_keep_map = torch.arange(N, device=q.device).unsqueeze(0).expand(B, -1)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
 
        return x, to_keep_map, attn


class SinkBlock(Block):
    def forward(self, x) -> torch.Tensor:
        if isinstance(x, tuple):
            x, to_keep_map, past_attn = x
        else:
            to_keep_map = None
            past_attn = None        

        skip_x = x
        x = self.norm1(x)

        x, to_keep_map, past_attn = self.attn(x, to_keep_map, past_attn)
        x = self.ls1(x)
        x = self.drop_path1(x)
        x = x + skip_x
        
        # Unrolled
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, to_keep_map, past_attn


class SinkVisionTransformer(VisionTransformer):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x, to_keep_map, past_attn = x
        x = self.norm(x)
        return x
    

def get_eviction_model(model, k=5, algorithm="topk", eviction_policy=None):
    if eviction_policy is not None and max(eviction_policy) > 0:
        model.__class__ = SinkVisionTransformer
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.__class__ = SinkAttention
        
            if isinstance(module, Block):
                module.__class__ = SinkBlock
        
        for layer_idx in range(len(model.blocks)):
            model.blocks[layer_idx].attn.eviction_config = dict(
                largest=False,
                has_cls = model.cls_token is not None,
                k=k,
                to_evict=eviction_policy[layer_idx],
                algorithm=algorithm
            ) 
        
    return model

def get_qkv_unbound_model(model, num_gemms=2):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.__class__ = UnboundQKVAttention
    return model

def parse_limits(num_layers=12, start_layer=0, end_layer=1):
    
    _ret_start = None
    _ret_end = None

    if isinstance(start_layer, float):
        if start_layer.is_integer(): 
            start_layer = int(start_layer)
        else:
            start_layer = round(start_layer * num_layers)   
    
    if isinstance(end_layer, float):
        if end_layer.is_integer(): 
            end_layer = int(end_layer)
        else:
            end_layer = round(end_layer * num_layers)   

    start_layer = int(start_layer)
    end_layer = int(end_layer)

    if start_layer >= 0:
        _ret_start = start_layer
    elif start_layer < 0:
        _ret_start = num_layers + start_layer

    if end_layer >= 0:
        _ret_end = end_layer
    elif end_layer < 0:
        _ret_end = num_layers + end_layer


    assert _ret_start >= 0  and _ret_start < num_layers, f"Start layer incorrect, {num_layers=}, {start_layer=}, {_ret_start=}" 
    assert _ret_end >= 0  and _ret_end < num_layers, f"End layer incorrect, {num_layers=}, {end_layer=}, {_ret_end=}" 
    assert _ret_start <= _ret_end, f"Start layer greater than end: {_ret_start=}, {_ret_end=}"

    return _ret_start, _ret_end
    

def get_constantly_decreasing_eviction_policy(num_layers, start_layer=0, end_layer=1, after_end=0, to_evict=3, step=0):
    """
    For every entry in `to_evicts`:
    <0: Use full context.
    =0: Use same context as last layer (maybe partially evicted).
    >0: Evict more context from last layer.
    """
    
    start_layer, end_layer = parse_limits(num_layers, start_layer, end_layer)

    eviction_policy = [-1 for _ in range(num_layers)]
    
    assert end_layer < num_layers and end_layer > start_layer, "Incorrect values for start and end layers."

    for layer_idx in range(start_layer, end_layer):
        eviction_policy[layer_idx] = to_evict + step * layer_idx

    for layer_idx in range(end_layer, num_layers):
        eviction_policy[layer_idx] = after_end

    print(f"{eviction_policy=}")

    return eviction_policy


def get_instant_pruning_eviction_policy(num_layers, eviction_policy_str, to_evict=3):
    # Here eviction policy will be a set of floats, where eviction will happen only at those 
    # blocks. The latter blocks will have policy set to `0`, except the last layer
    
    prune_spots = list(map(float, eviction_policy_str.split(",")))
    prune_spots = sorted([parse_limits(num_layers, prune_spots[i], -1)[0] for i in range(len(prune_spots))])
    
    eviction_policy = [-1 for _ in range(num_layers)]
    
    for layer_idx in range(prune_spots[0], num_layers - 1):
        eviction_policy[layer_idx] = 0
  
    for prune_idx in prune_spots:
        eviction_policy[prune_idx] = to_evict
 
    print(f"{eviction_policy=}")


def patch_model_for_kv_eviction(model, k=5, algorithm="topk", eviction_policy=None, to_evict=3, 
    start_layer=0, end_layer=1, step=0, after_end=0, num_gemms=2):
    
    if eviction_policy is not None and isinstance(eviction_policy, str):
        eviction_policy = get_instant_pruning_eviction_policy(len(model.blocks), eviction_policy)
    elif k > 0:
        eviction_policy = get_constantly_decreasing_eviction_policy(len(model.blocks), start_layer=start_layer, 
            end_layer=end_layer, to_evict=to_evict, step=step, after_end=after_end)

    model = replace_qkv_with_unbound(model, num_gemms=num_gemms)

    if k == 0:
        model = get_qkv_unbound_model(model, num_gemms=num_gemms)
    else:
        model = get_eviction_model(model, k=k, algorithm=algorithm, eviction_policy=eviction_policy)
    return model
