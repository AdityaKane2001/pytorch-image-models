readarray -t modellist < /workspace/akane/node-modellist.txt

for MODEL in "${modellist[@]}"
do
    ./unpruned.sh $MODEL 2>&1 | tee -a outfiles/$MODEL-unpruned.txt

    for num in $(seq 1 16)
    do
        ./arithmetic_mean.sh $MODEL $num 2>&1 | tee -a outfiles/$MODEL-arithmetic-mean-$num.txt
        ./geometric_mean.sh $MODEL $num 2>&1 | tee -a outfiles/$MODEL-geometric-mean-$num.txt

        for k in $(seq 1 5)
        do
            ./topk.sh $MODEL $num $k 2>&1 | tee -a outfiles/$MODEL-topk-$num-$k.txt
    done
done
