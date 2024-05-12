# iterate over the datasets
cpu=70
for dataset in "BAMultiShapes" "NCI1"; do
# for dataset in "MUTAG" "Mutagenicity"; do
    # iterate over the seeds
    seeds=(45 357 796)
    if [ "$dataset" == "NCI1" ]; then
        seeds=(45 1225 1983)
    fi
    for seed in "${seeds[@]}"; do
        # iterate over the sizes
        for size in 0.25 0.5 0.75 1.0; do
            command="taskset -c $cpu python pgexplainer.py -d $dataset -s $seed --size $size"
            echo "${command}"
            $command
        done
    done
done
