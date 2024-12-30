readarray -t modellist < /workspace/akane/node-modellist.txt

for MODEL in "${modellist[@]}"
do
    echo "#############################################################"
    echo "Running $MODEL unpruned..."
    ./unpruned.sh $MODEL 2>&1 | tee -a outfiles/$MODEL-unpruned.txt
    echo "=========================="

    # Since this is the 3-spot algorithm, we will be trying a larger magnitude of 
    # kv eviction number

    nums=(2 4 8 16 32 48 64)

    for num in ${nums[@]}
    do
        echo "=========================="
        ./seedkey.sh $MODEL $num 2>&1 | tee -a outfiles/$MODEL-seedkey-$num.txt
        # echo "=========================="
        # echo "Running $MODEL arithmetic mean with num=$num..."
        # ./arithmetic_mean.sh $MODEL $num 2>&1 | tee -a outfiles/$MODEL-arithmetic-mean-$num.txt
        # echo "=========================="
        # echo "Running $MODEL geometric mean with num=$num..."
        # ./geometric_mean.sh $MODEL $num 2>&1 | tee -a outfiles/$MODEL-geometric-mean-$num.txt

        # for k in $(seq 1 5)
        # do
        #     echo "=========================="
        #     echo "Running $MODEL topk with num=$num and k=$k..."
        #     ./topk.sh $MODEL $num $k 2>&1 | tee -a outfiles/$MODEL-topk-$num-$k.txt
        # done
    done
done
