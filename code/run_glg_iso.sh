cpu=70
log="../logs/acc_iso.log"
# rm $log
# iterate over the datasets
# for dataset in "BAMultiShapes" "NCI1"; do # Run with conda env 'pyg'
for dataset in "MUTAG" "Mutagenicity"; do # Run with conda env 'glg'
    # iterate over the seeds
    seeds=(45 357 796)
    if [ "$dataset" == "NCI1" ]; then
        seeds=(45 1225 1983)
    fi
    for seed in "${seeds[@]}"; do
        # iterate over the sizes
        for size in 0.25 0.5 0.75 1.0; do
            command="taskset -c $cpu python acc_isomorphism.py -d $dataset -e PGExplainer \
            --split test --size $size -s $seed -r 0"
            printf "\n>>> $dataset, $size, $seed\n" >> $log
            $command >> $log
        done
    done
done
