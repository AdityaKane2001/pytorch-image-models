python3 -u validate.py --data-dir /data/datasets/ImageNet/ \
   --model=$1\
   --evict-algo="none"\
   --evict-k=0\
   --evict-start=-1\
   --evict-end=-1\
   --evict-num=0\
   --evict-after-end=-1\
   --amp\
   --results-format="json"\
   --batch-size=1024\
   --workers=32\
   --num-gpu=8\
   --evict-num-gemms=2\
   --savedir="/home/users/akane/kv-evict-3spot-policy-sortx/$1/unpruned/"\
   --pretrained

