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

# from sink_utils import update_keepmap, prune_x, sort_x_importance, split_qkv_weights, split_qkv_weights_two
from sink_utils import *

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

class BenchUnboundQKVAttention(Attention): 
    def forward(self, x, timing_dict=None, key_prefix=""):
        B, N, C = x.shape
        if self.num_gemms == 1:
            with ProfilerContext(timing_dict, "1gemm", key_prefix) as pc:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
            timing_dict = pc()
        elif self.num_gemms == 2:
            with ProfilerContext(timing_dict, "2gemm", key_prefix) as pc:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                kv = self.kv(x).view(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
            timing_dict = pc()

        elif self.num_gemms == 3:
            with ProfilerContext(timing_dict, "3gemm", key_prefix) as pc:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = self.k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = self.v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()

        else:
            raise AttributeError("either of one,two, three gemm should be true")
        with ProfilerContext(timing_dict, "qknorm", key_prefix) as pc:
            q, k = self.q_norm(q), self.k_norm(k)
        timing_dict = pc()

        with ProfilerContext(timing_dict, "attn", key_prefix) as pc:
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
        timing_dict = pc()

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


def delete_one(delete_index, max_value):
    nzmask = torch.arange(max_value, device=delete_index.device, dtype=delete_index.dtype).unsqueeze(0).expand(delete_index.shape[0], -1)
    expanded_delete_index = delete_index.unsqueeze(-1).expand(-1, max_value)
    return nzmask[nzmask != expanded_delete_index].view(delete_index.shape[0], max_value - 1)


class ExperimentalGranularBenchUnboundQKVAttention(Attention): 
    def forward(self, x, timing_dict=None, key_prefix=""):
        B, N, C = x.shape
        if self.num_gemms == 1:
            with ProfilerContext(timing_dict, "qkv_proj", key_prefix) as pc:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "1gemm_unbind", key_prefix) as pc:
                q, k, v = qkv.unbind(0)
            timing_dict = pc()
        elif self.num_gemms == 2:
            with ProfilerContext(timing_dict, "q_proj", key_prefix) as pc:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "kv_proj", key_prefix) as pc:
                kv = self.kv(x).view(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "2gemm_unbind", key_prefix) as pc:
                k, v = kv.unbind(0)
            timing_dict = pc()

        elif self.num_gemms == 3:
            with ProfilerContext(timing_dict, "q_proj", key_prefix) as pc:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "k_proj", key_prefix) as pc:
                k = self.k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "v_proj", key_prefix) as pc:
                v = self.v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()

        else:
            raise AttributeError("either of one,two, three gemm should be true")
        with ProfilerContext(timing_dict, "qknorm", key_prefix) as pc:
            q, k = self.q_norm(q), self.k_norm(k)
        timing_dict = pc()

        with ProfilerContext(timing_dict, "attn", key_prefix) as pc:
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
        timing_dict = pc()


        attn_argmins = torch.argmin(attn, dim=-1)

        # torch.cuda.synchronize()
        # with ProfilerContext(timing_dict, "seedkey_prune", key_prefix):
        #     bincount = batched_bincount(attn_argmins.flatten(1,2), dim=-1, max_value=k.shape[-2])
        #     seedkey_idx = torch.argmax(bincount, dim=-1)
        #     restkey_map = delete_one(seedkey_idx, k.shape[-2]).unsqueeze(-1).expand(-1, -1, k.shape[-1])
        #     seedkey_map = seedkey_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k.shape[-1])
        #     keys = torch.mean(k, dim=1)
        #     seedkey = torch.gather(keys, index=seedkey_map, dim=-2)
        #     restkey = torch.gather(keys, index=restkey_map, dim=-2)
        #     impmap = torch.nn.functional.cosine_similarity(restkey, seedkey)
        #     impsort = torch.argsort(impmap, dim=-1, descending=True)
        #     to_keep_map = impsort[..., 32:].unsqueeze(-1).unsqueeze(1).expand(-1, k.shape[1], -1, k.shape[-1])
        #     # to_prune_map = impsort[..., 32:, :].unsqueeze(-1).expand(k.shape[-1])
        #     keep_keys = torch.gather(k, index=to_keep_map, dim=-2)
        # timing_dict = pc()
        
        # headmode_idx = 1 if not self.fused_attn else 2 
        # seqmode_idx = 2 if not self.fused_attn else 1
        Nc = N
        
        with ProfilerContext(timing_dict, "seedkey_prune_argsort", key_prefix):
            bincount = batched_bincount(attn_argmins.flatten(1,2), dim=-1, max_value=k.shape[-2])
            bincount_argsorted = torch.argsort(bincount, dim=-1, descending=True, stable=True)
            keys = k.mean(dim=1)
            
            seedkey_idx = bincount_argsorted[..., 0]
            restkey_map = bincount_argsorted[..., 1:].unsqueeze(-1).expand(-1, -1, k.shape[-1])
            # restkey_map = delete_one(seedkey_idx, k.shape[-2]).unsqueeze(-1).expand(-1, -1, k.shape[-1])
            seedkey_map = seedkey_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k.shape[-1])
            seedkey = torch.gather(keys, index=seedkey_map, dim=-2)
            restkey = torch.gather(keys, index=restkey_map, dim=-2)
            impmap = torch.nn.functional.cosine_similarity(restkey, seedkey)
            impsort = torch.argsort(impmap, dim=-1, descending=True)
            to_keep_map = impsort[..., 32:].unsqueeze(-1).unsqueeze(1).expand(-1, k.shape[1], -1, k.shape[-1])
            # to_prune_map = impsort[..., 32:, :].unsqueeze(-1).expand(k.shape[-1])
            keep_keys = torch.gather(k, index=to_keep_map, dim=-2)
        timing_dict = pc()            
        
        torch.cuda.synchronize()
        
        with ProfilerContext(timing_dict, "headwise_seedkey_prune_argsort", key_prefix):
            # attn_argmins: torch.Tensor[B, H, Nq]
            bincount = batched_bincount(attn_argmins, dim=-1, max_value=Nc)
            bincount_argsorted = torch.argsort(bincount, dim=-1, descending=True, stable=True)
            # bincount_argsorted: torch.Tensor[B, H, Nc]
            seedkey_map = bincount_argsorted[..., 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            restkey_map = bincount_argsorted[..., 1:].unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            seedkey = torch.gather(k, index=seedkey_map, dim=-2)
            restkey = torch.gather(k, index=restkey_map, dim=-2)
            impmap = torch.nn.functional.cosine_similarity(restkey, seedkey, dim=-1)
            impsort = torch.argsort(impmap, dim=-1, descending=True)
            to_keep_map = impsort[..., 32:].unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            # to_prune_map = impsort[..., 32:, :].unsqueeze(-1).expand(k.shape[-1])
            keep_keys = torch.gather(k, index=to_keep_map, dim=-2)
        timing_dict = pc()            

        torch.cuda.synchronize()

        with ProfilerContext(timing_dict, "batched_bincount", key_prefix):
            bincount = batched_bincount(attn_argmins, dim=-1, max_value=Nc)
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "batched_argsort", key_prefix):
            bincount_argsorted = torch.argsort(bincount, dim=-1, descending=True, stable=True)
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "seedkey_makemap", key_prefix):
            seedkey_map = bincount_argsorted[..., 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "restkey_makemap", key_prefix):
            restkey_map = bincount_argsorted[..., 1:].unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "seedkey_gather", key_prefix):
            seedkey = torch.gather(k, index=seedkey_map, dim=-2)
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "restkey_gather", key_prefix):
            restkey = torch.gather(k, index=restkey_map, dim=-2)
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "cossim", key_prefix):
            impmap = torch.nn.functional.cosine_similarity(restkey, seedkey, dim=-1)
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "cossim_argsort", key_prefix):
            impsort = torch.argsort(impmap, dim=-1, descending=True)
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "unsqueeze_expand_many_elem", key_prefix):
            to_keep_map = impsort[..., 32:].unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            # to_prune_map = impsort[32:].unsqueeze(-1).expand(k.shape[-1])
        timing_dict = pc()            
        with ProfilerContext(timing_dict, "final_key_gather", key_prefix):
            keep_keys = torch.gather(k, index=to_keep_map, dim=-2)
        timing_dict = pc()            

        torch.cuda.synchronize()
        
        # with ProfilerContext(timing_dict, "batched_bincount", key_prefix):
        #     bincount = batched_bincount(attn_argmins.flatten(1,2), dim=-1, max_value=k.shape[-2])
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "seedkey_argmax", key_prefix):
        #     seedkey_idx = torch.argmax(bincount, dim=-1)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "seedkey_argsort", key_prefix):
        #     seedkey_sorted= torch.argsort(bincount, dim=-1, descending=True)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "seedkey_argsort_stable", key_prefix):
        #     seedkey_sorted= torch.argsort(bincount, dim=-1, stable=True, descending=True)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "delete_one", key_prefix):
        #     restkey_map = delete_one(seedkey_idx, k.shape[-2]).unsqueeze(-1).expand(-1, -1, k.shape[-1])
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "unsqueeze_expand_single_elem", key_prefix):
        #     seedkey_map = seedkey_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k.shape[-1])
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "keys_mean", key_prefix):
        #     keys = k.mean(dim=1)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "seedkey_gather", key_prefix):
        #     seedkey = torch.gather(keys, index=seedkey_map, dim=-2)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "restkey_gather", key_prefix):
        #     restkey = torch.gather(keys, index=restkey_map, dim=-2)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "cossim", key_prefix):
        #     impmap = torch.nn.functional.cosine_similarity(restkey, seedkey)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "cossim_argsort", key_prefix):
        #     impsort = torch.argsort(impmap, dim=-1, descending=True)
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "unsqueeze_expand_many_elem", key_prefix):
        #     to_keep_map = impsort[..., 32:].unsqueeze(-1).unsqueeze(1).expand(-1, k.shape[1], -1, k.shape[-1])
        #     # to_prune_map = impsort[32:].unsqueeze(-1).expand(k.shape[-1])
        # timing_dict = pc()            
        # with ProfilerContext(timing_dict, "final_key_gather", key_prefix):
        #     # to_prune_map = impsort[32:].unsqueeze(-1).expand(k.shape[-1])
        #     keep_keys = torch.gather(k, index=to_keep_map, dim=-2)
        # timing_dict = pc()            

        torch.cuda.synchronize()
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GranularBenchUnboundQKVAttention(Attention): 
    def forward(self, x, timing_dict=None, key_prefix=""):
        B, N, C = x.shape
        if self.num_gemms == 1:
            with ProfilerContext(timing_dict, "qkv_proj", key_prefix) as pc:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "1gemm_unbind", key_prefix) as pc:
                q, k, v = qkv.unbind(0)
            timing_dict = pc()
        elif self.num_gemms == 2:
            with ProfilerContext(timing_dict, "q_proj", key_prefix) as pc:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "kv_proj", key_prefix) as pc:
                kv = self.kv(x).view(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "2gemm_unbind", key_prefix) as pc:
                k, v = kv.unbind(0)
            timing_dict = pc()

        elif self.num_gemms == 3:
            with ProfilerContext(timing_dict, "q_proj", key_prefix) as pc:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "k_proj", key_prefix) as pc:
                k = self.k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()
            with ProfilerContext(timing_dict, "v_proj", key_prefix) as pc:
                v = self.v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            timing_dict = pc()

        else:
            raise AttributeError("either of one,two, three gemm should be true")
        with ProfilerContext(timing_dict, "qknorm", key_prefix) as pc:
            q, k = self.q_norm(q), self.k_norm(k)
        timing_dict = pc()

        

        with ProfilerContext(timing_dict, "attn", key_prefix) as pc:
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
        timing_dict = pc()


        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class SeedkeyAttention(Attention):
    def check_patching(self):
        cfg_keys = list(self.eviction_config.keys())
        has_cls = "has_cls" in cfg_keys
        has_evict_k = "k" in cfg_keys
        has_to_evict = "to_evict" in cfg_keys
        has_largest = "largest" in cfg_keys
        has_algorithm = "algorithm" in cfg_keys
        has_pruning_params = has_cls and has_evict_k and has_to_evict and has_largest and has_algorithm
        
        return has_pruning_params

    def check_eviction_cfg(self, to_keep_map, querywise_argmin):
        has_pruning_params = self.check_patching()
        ecfg = self.eviction_config
        
        if not has_pruning_params:
            raise ValueError("Patching not done properly!")

        if ecfg["to_evict"] == 0 and to_keep_map is None:
            raise ValueError("Got self.to_evict == 0, but to_keep_map was not supplied.")

        if ecfg["to_evict"] > 0:
            if querywise_argmin is None:
                raise ValueError("Got self.to_evict > 0, but querywise_argmin was not supplied.")
            elif to_keep_map is None:
                raise ValueError("Got self.to_evict > 0, but to_keep_map was not supplied.")

            
    def forward(self, x: torch.Tensor, to_keep_map: torch.Tensor = None, querywise_argmin: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        self.check_eviction_cfg(to_keep_map, querywise_argmin)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        ecfg = self.eviction_config
        if ecfg["to_evict"] >= 0:
            to_keep_map = update_seedkey_keepmap(querywise_argmin, k, to_keep_map, to_evict=ecfg["to_evict"],
                has_cls=ecfg["has_cls"], heads_first=True)
            to_keep_index = to_keep_map.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
            
            k = torch.gather(k, index=to_keep_index, dim=-2)
            v = torch.gather(v, index=to_keep_index, dim=-2)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            to_keep_map = torch.arange(N, device=q.device).unsqueeze(0).expand(B, -1)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        try:        
            querywise_argmin = torch.argmin(attn, dim=-1)
        except:
            print(f"Argmin failed with {attn.shape=}")
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
 
        return x, to_keep_map, querywise_argmin 

class SeedkeyBlock(Block):
    def forward(self, x)-> torch.Tensor:
        if isinstance(x, tuple):
            x, to_keep_map, querywise_argmin = x
        else:
            to_keep_map = None
            querywise_argmin = None        

        skip_x = x
        x = self.norm1(x)

        x, to_keep_map, querywise_argmin = self.attn(x, to_keep_map, querywise_argmin)
        x = self.ls1(x)
        x = self.drop_path1(x)
        x = x + skip_x
        
        # Unrolled
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, to_keep_map, querywise_argmin
        

class SinkAttention(Attention):
    def check_patching(self):
        has_q = hasattr(self, "q")
        has_k = hasattr(self, "k")
        has_v = hasattr(self, "v")
        has_kv = hasattr(self, "kv")
        has_qkv_unbound = has_q and ((has_k and has_v) or has_kv)

        assert (has_k and has_v) ^ (has_kv), "SinkAttention needs to have exactly one of {k, v} and kv."

        cfg_keys = list(self.eviction_config.keys())
        has_cls = "has_cls" in cfg_keys
        has_evict_k = "k" in cfg_keys
        has_to_evict = "to_evict" in cfg_keys
        has_largest = "largest" in cfg_keys
        has_algorithm = "algorithm" in cfg_keys
        has_pruning_params = has_cls and has_evict_k and has_to_evict and has_largest and has_algorithm
        
        return has_qkv_unbound, has_k and has_v, has_kv, has_pruning_params

    def check_eviction_cfg(self, to_keep_map, past_attn):
        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config
        
        if not has_pruning_params:
            raise ValueError("Patching not done properly!")

        assert has_qkv_unbound, "QKV not unbound!"

        if ecfg["to_evict"] == 0 and to_keep_map is None:
            raise ValueError("Got self.to_evict == 0, but to_keep_map was not supplied.")

        if ecfg["to_evict"] > 0:
            if past_attn is None:
                raise ValueError("Got self.to_evict > 0, but past_attn was not supplied.")
            elif to_keep_map is None:
                raise ValueError("Got self.to_evict > 0, but to_keep_map was not supplied.")

        if not (has_kandv or has_kv):
            raise AttributeError(f"{self.__class__} instance should have k and v, or kv.")         
            

    def forward(self, x: torch.Tensor, to_keep_map: torch.Tensor = None, past_attn: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        self.check_eviction_cfg(to_keep_map, past_attn)

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
                kv = self.kv(pruned_x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
        
        else:
            if has_kandv: 
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = self.k(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = self.v(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            elif has_kv:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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

class BenchSinkAttention(SinkAttention):
    def forward(self, x: torch.Tensor, to_keep_map: torch.Tensor = None, past_attn: torch.Tensor = None, timing_dict=None, key_prefix="") -> torch.Tensor:
        B, N, C = x.shape

        self.check_eviction_cfg(to_keep_map, past_attn)

        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config

        if ecfg["to_evict"] >= 0:
        
            to_keep_map, timing_dict = bench_update_keepmap(past_attn, to_keep_map, k=ecfg["k"], to_evict=ecfg["to_evict"], 
                 largest=ecfg["largest"], has_cls=ecfg["has_cls"], algorithm=ecfg["algorithm"], timing_dict=timing_dict, key_prefix=key_prefix)
            
            with ProfilerContext(timing_dict, "gatherop", key_prefix) as pc:            
                pruned_x = prune_x(x, to_keep_map)
            timing_dict = pc()


            q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            if timing_dict is not None:
                qend.record()
                torch.cuda.synchronize()
                timing_dict[key_prefix + "q_proj"] = qstart.elapsed_time(qend)

            if has_kandv: 
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                k = self.k(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "k_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                v = self.v(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "v_proj"] = qstart.elapsed_time(qend)
            elif has_kv:
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                kv = self.kv(pruned_x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "kv_proj"] = qstart.elapsed_time(qend)
            
            if timing_dict is not None:
                end.record()
                torch.cuda.synchronize()
                timing_dict[key_prefix + "qkv_proj"] = start.elapsed_time(end)
        
        else:
            if timing_dict is not None:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            
            if has_kandv: 
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_q_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                k = self.k(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_k_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                v = self.v(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_v_proj"] = qstart.elapsed_time(qend)
            elif has_kv:
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_q_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_kv_proj"] = qstart.elapsed_time(qend)
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
            
            if timing_dict is not None:
                end.record()
                torch.cuda.synchronize()
                timing_dict[key_prefix + "noprune_qkv_proj"] = start.elapsed_time(end)
            
            to_keep_map = torch.arange(N, device=q.device).unsqueeze(0).expand(B, -1)

        if timing_dict is not None:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()

        q, k = self.q_norm(q), self.k_norm(k)
        if timing_dict is not None:
            end.record()
            torch.cuda.synchronize()
            timing_dict[key_prefix + "qk_norm"] = start.elapsed_time(end)

        q = q * self.scale
        if timing_dict is not None:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        if timing_dict is not None:
            end.record()
            torch.cuda.synchronize()
            timing_dict[key_prefix + "attn"] = start.elapsed_time(end)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
 
        return x, to_keep_map, attn



class SortedSinkAttention(SinkAttention):
    def check_eviction_cfg(self, to_keep_map, past_attn):
        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config
        
        if not has_pruning_params:
            raise ValueError("Patching not done properly!")

        assert has_qkv_unbound, "QKV not unbound!"

        if ecfg["to_evict"] > 0:
            if past_attn is None:
                raise ValueError("Got self.to_evict > 0, but past_attn was not supplied.")

        if not (has_kandv or has_kv):
            raise AttributeError(f"{self.__class__} instance should have k and v, or kv.")         
    
    def forward(self, x, to_keep_map, past_attn):
        B, N, C = x.shape 
        self.check_eviction_cfg(to_keep_map, past_attn)

        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config

        if ecfg["to_evict"] >= 0:

            if ecfg["to_evict"] > 0:
                # to_keep_map = update_keepmap(past_attn, to_keep_map, k=ecfg["k"], to_evict=ecfg["to_evict"], 
                #     largest=ecfg["largest"], has_cls=ecfg["has_cls"], algorithm=ecfg["algorithm"])
                
                x = sort_x_importance(x, past_attn, to_evict=ecfg["to_evict"], k=ecfg["k"], largest=ecfg["largest"], has_cls=ecfg["has_cls"], algorithm=ecfg["algorithm"])

                pruned_N = past_attn.shape[-1] - ecfg["to_evict"]

                to_keep_map = None
            

            if ecfg["to_evict"] == 0:
                pruned_N = past_attn.shape[-1]

            pruned_x = x[..., :pruned_N,:]
            
            q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            if has_kandv: 
                k = self.k(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = self.v(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            elif has_kv:
                kv = self.kv(pruned_x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
            
        else:
            if has_kandv: 
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = self.k(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = self.v(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            elif has_kv:
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
                kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
            
            to_keep_map = None # torch.arange(N, device=q.device).unsqueeze(0).expand(B, -1)

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


class BenchSortedSinkAttention(SinkAttention):
    def check_eviction_cfg(self, to_keep_map, past_attn):
        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config
        
        if not has_pruning_params:
            raise ValueError("Patching not done properly!")

        assert has_qkv_unbound, "QKV not unbound!"

        if ecfg["to_evict"] > 0:
            if past_attn is None:
                raise ValueError("Got self.to_evict > 0, but past_attn was not supplied.")

        if not (has_kandv or has_kv):
            raise AttributeError(f"{self.__class__} instance should have k and v, or kv.")         
    
    def forward(self, x, to_keep_map, past_attn, timing_dict=None, key_prefix=""):
        B, N, C = x.shape 
        self.check_eviction_cfg(to_keep_map, past_attn)

        has_qkv_unbound, has_kandv, has_kv, has_pruning_params = self.check_patching()
        ecfg = self.eviction_config

        if ecfg["to_evict"] >= 0:
            if ecfg["to_evict"] > 0:
                # to_keep_map = update_keepmap(past_attn, to_keep_map, k=ecfg["k"], to_evict=ecfg["to_evict"], 
                #     largest=ecfg["largest"], has_cls=ecfg["has_cls"], algorithm=ecfg["algorithm"])
                
                # if timing_dict is not None:
                #     torch.cuda.synchronize()
                #     start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                #     start.record()

                if timing_dict is not None:
                    x, timing_dict = sort_x_importance(x, past_attn, to_evict=ecfg["to_evict"], k=ecfg["k"], largest=ecfg["largest"], has_cls=ecfg["has_cls"], algorithm=ecfg["algorithm"], timing_dict=timing_dict, key_prefix=key_prefix)
               
                else:
                    x = sort_x_importance(x, past_attn, to_evict=ecfg["to_evict"], k=ecfg["k"], largest=ecfg["largest"], has_cls=ecfg["has_cls"], algorithm=ecfg["algorithm"])

                # if timing_dict is not None:
                #     end.record() # = torch.cuda.Event(enable_timing=True)
                #     torch.cuda.synchronize()
                #     timing_dict[key_prefix + "sorting"] = start.elapsed_time(end)
                

                pruned_N = past_attn.shape[-1] - ecfg["to_evict"]

                to_keep_map = None

            if timing_dict is not None:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()


            if ecfg["to_evict"] == 0:
                pruned_N = past_attn.shape[-1]

            pruned_x = x[..., :pruned_N,:]
            
            if timing_dict is not None:
                end.record() # = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                timing_dict[key_prefix + "slicing"] = start.elapsed_time(end)
            if timing_dict is not None:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            
            if timing_dict is not None:
                torch.cuda.synchronize()
                qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                qstart.record()

            q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            if timing_dict is not None:
                qend.record()
                torch.cuda.synchronize()
                timing_dict[key_prefix + "q_proj"] = qstart.elapsed_time(qend)

            if has_kandv: 
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                k = self.k(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "k_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                v = self.v(pruned_x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "v_proj"] = qstart.elapsed_time(qend)
            elif has_kv:
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                kv = self.kv(pruned_x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "kv_proj"] = qstart.elapsed_time(qend)
            
            if timing_dict is not None:
                end.record()
                torch.cuda.synchronize()
                timing_dict[key_prefix + "qkv_proj"] = start.elapsed_time(end)
        
        else:
            if timing_dict is not None:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            
            if has_kandv: 
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_q_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                k = self.k(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_k_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                v = self.v(x).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_v_proj"] = qstart.elapsed_time(qend)
            elif has_kv:
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_q_proj"] = qstart.elapsed_time(qend)
                if timing_dict is not None:
                    torch.cuda.synchronize()
                    qstart, qend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    qstart.record()
                kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                if timing_dict is not None:
                    qend.record()
                    torch.cuda.synchronize()
                    timing_dict[key_prefix + "noprune_kv_proj"] = qstart.elapsed_time(qend)
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
            
            if timing_dict is not None:
                end.record()
                torch.cuda.synchronize()
                timing_dict[key_prefix + "noprune_qkv_proj"] = start.elapsed_time(end)
            
            to_keep_map = None # torch.arange(N, device=q.device).unsqueeze(0).expand(B, -1)

        if timing_dict is not None:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()

        q, k = self.q_norm(q), self.k_norm(k)
        if timing_dict is not None:
            end.record()
            torch.cuda.synchronize()
            timing_dict[key_prefix + "qk_norm"] = start.elapsed_time(end)

        q = q * self.scale
        if timing_dict is not None:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        if timing_dict is not None:
            end.record()
            torch.cuda.synchronize()
            timing_dict[key_prefix + "attn"] = start.elapsed_time(end)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
 
        return x, to_keep_map, attn


class SinkBlock(Block):
    def forward(self, x)-> torch.Tensor:
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

class BenchSinkBlock(Block):
    def forward(self, x, timing_dict=None, key_prefix="") -> torch.Tensor:
        if isinstance(x, tuple):
            x, to_keep_map, past_attn = x
        else:
            to_keep_map = None
            past_attn = None        

        skip_x = x
        x = self.norm1(x)

        x, to_keep_map, past_attn = self.attn(x, to_keep_map, past_attn, timing_dict=timing_dict, key_prefix=key_prefix)
        x = self.ls1(x)
        x = self.drop_path1(x)
        x = x + skip_x
        
        # Unrolled
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, to_keep_map, past_attn


class BenchUnboundQKVBlock(Block):
    def forward(self, x, timing_dict=None, key_prefix="") -> torch.Tensor:
        skip_x = x
        x = self.norm1(x)

        x = self.attn(x, timing_dict=timing_dict, key_prefix=key_prefix)
        x = self.ls1(x)
        x = self.drop_path1(x)
        x = x + skip_x
        
        # Unrolled
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class BenchSinkVisionTransformer(VisionTransformer):
    def forward_features(self, x: torch.Tensor, timing_dict=None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        for idx, block in enumerate(self.blocks):
            x = block(x, timing_dict=timing_dict, key_prefix=f"block{idx}_")

        x, to_keep_map, past_attn = x
        x = self.norm(x)
        return x
    
    def forward(self, x: torch.Tensor, timing_dict=None) -> torch.Tensor:
        x = self.forward_features(x, timing_dict=timing_dict)
        x = self.forward_head(x)
        return x, timing_dict

class BenchUnboundQKVVisionTransformer(VisionTransformer):
    def forward_features(self, x: torch.Tensor, timing_dict=None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        for idx, block in enumerate(self.blocks):
            x = block(x, timing_dict=timing_dict, key_prefix=f"block{idx}_")

        x = self.norm(x)
        return x
    
    def forward(self, x: torch.Tensor, timing_dict=None) -> torch.Tensor:
        x = self.forward_features(x, timing_dict=timing_dict)
        x = self.forward_head(x)
        return x, timing_dict


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
    
class SeedkeyVisionTransformer(VisionTransformer):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x, to_keep_map, querywise_argmin = x
        x = self.norm(x)
        return x

def get_eviction_model(model, k=5, algorithm="topk", eviction_policy=None, profile=False):
    if eviction_policy is not None and max(eviction_policy) > 0:
        model.__class__ = SeedkeyVisionTransformer # SinkVisionTransformer if not profile else BenchSinkVisionTransformer
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.__class__ = SeedkeyAttention
                # module.__class__ = SortedSinkAttention if not profile else BenchSortedSinkAttention
                # module.__class__ = SinkAttention if not profile else BenchSinkAttention
        
            if isinstance(module, Block):
                module.__class__ = SeedkeyBlock # SinkBlock if not profile else BenchSinkBlock
        
        for layer_idx in range(len(model.blocks)):
            model.blocks[layer_idx].attn.eviction_config = dict(
                largest=False,
                has_cls = model.cls_token is not None,
                k=k,
                to_evict=eviction_policy[layer_idx],
                algorithm=algorithm
            ) 
        
    return model

def get_qkv_unbound_model(model, num_gemms=2, profile=False):
    print(f"Getting unbound model with {profile=}")
    model.__class__ = VisionTransformer if not profile else BenchUnboundQKVVisionTransformer
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.__class__ = UnboundQKVAttention if not profile else ExperimentalGranularBenchUnboundQKVAttention 
        if isinstance(module, Block):
            module.__class__ = Block if not profile else BenchUnboundQKVBlock
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


def get_instant_pruning_eviction_policy(num_layers, eviction_policy_str, to_evict=3, seedkey=True):
    # Here eviction policy will be a set of floats, where eviction will happen only at those 
    # blocks. The latter blocks will have policy set to `0`, except the last layer
    
    prune_spots = list(map(float, eviction_policy_str.split(",")))
    prune_spots = sorted([parse_limits(num_layers, prune_spots[i], -1)[0] for i in range(len(prune_spots))])
    
    eviction_policy = [-1 for _ in range(num_layers)]
    
    for layer_idx in range(prune_spots[0], num_layers - 1):
        eviction_policy[layer_idx] = 0
  
    for idx, prune_idx in enumerate(prune_spots):
        if seedkey:    
            eviction_policy[prune_idx] = to_evict * (idx + 1)
        else:
            eviction_policy[prune_idx] = to_evict

    print(f"{eviction_policy=}")
    return eviction_policy

def patch_model_for_kv_eviction(model, k=5, algorithm="topk", eviction_policy=None, to_evict=3, 
    start_layer=0, end_layer=1, step=0, after_end=0, num_gemms=2, profile=False):

    seedkey = algorithm == "seedkey"   
 
    if eviction_policy is not None and isinstance(eviction_policy, str):
        eviction_policy = get_instant_pruning_eviction_policy(len(model.blocks), eviction_policy, to_evict=to_evict, seedkey=seedkey)
    elif k > 0:
        eviction_policy = get_constantly_decreasing_eviction_policy(len(model.blocks), start_layer=start_layer, 
            end_layer=end_layer, to_evict=to_evict, step=step, after_end=after_end)

    model = replace_qkv_with_unbound(model, num_gemms=num_gemms)

    if k == 0:
        model = get_qkv_unbound_model(model, num_gemms=num_gemms, profile=profile)
    else:
        model = get_eviction_model(model, k=k, algorithm=algorithm, eviction_policy=eviction_policy, profile=profile)
    return model
