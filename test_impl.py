import torch
import timm

from patch import patch_model_for_kv_eviction 


if __name__ == "__main__":
    model = timm.models.create_model("vit_large_patch14_clip_224.openai_ft_in12k_in1k")
    model = patch_model_for_kv_eviction(
        model, 
        algorithm="topk",
        k=5,
        to_evict=10,
        start_layer=0.5,
        end_layer=-1,
        after_end=-1,
        step=0 
    )

    model.to("cuda")
    inputs = torch.rand((2, 3, 224, 224), device="cuda")
    
    model.eval()
    with torch.no_grad():
        model(inputs)
