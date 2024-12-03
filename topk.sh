python3 -u validate.py --data-dir /data/datasets/ImageNet/ \
    --model=$1\
    --evict-algo="topk"\
    --evict-policy="0.25,0.5,0.75"\
    --evict-k=$3\
    --evict-start=-1\
    --evict-end=-1\
    --evict-num=$2\
    --evict-after-end=-1\
    --amp\
    --results-format="json"\
    --batch-size=1024\
    --workers=32\
    --num-gpu=8\
    --savedir="/home/users/akane/kv-evict-results-3spot-policy/$1/topk/"\
    --pretrained

