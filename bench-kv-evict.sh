readarray -t modellist < /workspace/akane/node-benchlist.txt

echo "#####################"

for MODEL in "${modellist[@]}"
do
    echo "#############################################################"
    echo "=========================="
    echo "Running $MODEL unpruned..."
    python3 -u benchmark-kv-evict.py \
       --model=$MODEL\
       --evict-algo="none"\
       --evict-k=-1\
       --evict-start=-1\
       --evict-end=-1\
       --evict-num=-1\
       --evict-after-end=-1\
       --amp\
       --savedir="/home/users/akane/kv-evict-bench/"

    for num in $(seq 1 16)
    do
        echo "=========================="
        echo "Running $MODEL arithmetic mean with num=$num..."
        python3 -u benchmark-kv-evict.py \
           --model=$MODEL\
           --evict-algo="arithmetic_mean"\
           --evict-k=1000\
           --evict-start=0.5\
           --evict-end=-1\
           --evict-num=$num\
           --evict-after-end=-1\
           --amp\
           --savedir="/home/users/akane/kv-evict-bench/"
        echo "=========================="
        echo "Running $MODEL geometric mean with num=$num..."
        python3 -u benchmark-kv-evict.py \
           --model=$MODEL\
           --evict-algo="geometric_mean"\
           --evict-k=1000\
           --evict-start=0.5\
           --evict-end=-1\
           --evict-num=$num\
           --evict-after-end=-1\
           --amp\
           --savedir="/home/users/akane/kv-evict-bench/"

        for k in $(seq 1 5)
        do
            echo "=========================="
            echo "Running $MODEL topk with num=$num and k=$k..."
            python3 -u benchmark-kv-evict.py \
               --model=$MODEL\
               --evict-algo="topk"\
               --evict-k=$k\
               --evict-start=0.5\
               --evict-end=-1\
               --evict-num=$num\
               --evict-after-end=-1\
               --amp\
               --savedir="/home/users/akane/kv-evict-bench/"
        done
    done
done

