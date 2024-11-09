python -u validate.py --data-dir /data/akane38/ImageNet/ \
    --model "vit_small_patch16_224.augreg_in21k_ft_in1k" \
    --evict-k=5\
    --evict-start=6\
    --evict-end=11\
    --evict-num=8\
    --evict-after-end=-1\
    --pretrained
