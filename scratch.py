import timm

print(timm.list_models("vit*_in1k", pretrained=True))
