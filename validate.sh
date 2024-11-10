modellist=(\
"eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"\
"eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"\
"eva_giant_patch14_560.m30m_ft_in22k_in1k"\
"eva02_large_patch14_448.mim_in22k_ft_in1k"\
"eva_giant_patch14_336.m30m_ft_in22k_in1k"\
"eva02_large_patch14_448.mim_m38m_ft_in1k"\
"eva_giant_patch14_336.clip_ft_in1k"\
"eva_large_patch14_336.in22k_ft_in22k_in1k"\
"eva_giant_patch14_224.clip_ft_in1k"\
)

for MODEL in "${modellist[@]}"
do 
    echo "##################### $MODEL"
    echo "======== unpruned"
    python3 -u validate.py --data-dir /data/datasets/ImageNet/ \
        --model $MODEL \
        --evict-algo="none"\
        --evict-k=0\
        --evict-start=12\
        --evict-end=22\
        --evict-num=8\
        --evict-after-end=-1\
        --amp\
        --num-gpu=8\
        --pretrained
    
    echo "======= arithmetic mean"
    python3 -u validate.py --data-dir /data/datasets/ImageNet/ \
        --model $MODEL \
        --evict-algo="arithmetic_mean"\
        --evict-k=1000\
        --evict-start=12\
        --evict-end=22\
        --evict-num=8\
        --evict-after-end=-1\
        --amp\
        --num-gpu=8\
        --pretrained
    
    
    for k in $(seq 1 8)
    do
    echo "======= top-k k=$k"
    python3 -u validate.py --data-dir /data/datasets/ImageNet/ \
        --model=$MODEL\
        --evict-algo="topk"\
        --evict-k=$k\
        --evict-start=12\
        --evict-end=22\
        --evict-num=8\
        --evict-after-end=-1\
        --amp\
        --num-gpu=8\
        --pretrained
    done
done
