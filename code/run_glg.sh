cpu=70
# iterate over the datasets
for dataset in "NCI1"; do
# for dataset in "BAMultiShapes" "NCI1"; do
# for dataset in "MUTAG" "Mutagenicity"; do
    # iterate over the seeds
    seeds=(45 357 796)
    if [ "$dataset" == "NCI1" ]; then
        seeds=(45 1225 1983)
    fi
    for seed in "${seeds[@]}"; do
        # iterate over the sizes
        for size in 0.25 0.5 0.75 1.0; do
            command="taskset -c $cpu python glg_${dataset}.py -d cpu -s $seed -r 0 --size $size -e PGExplainer"
            log="../logs/acc/${dataset}_PGExplainer_size${size}_seed${seed}_run${run}.log"
            echo "$command &> $log"
            $command &> $log
        done
    done
done
