dataset="NCI1"

pg_data="../our_data/${dataset}/"
glg_data="../../GraphTrail/data/${dataset}/"

# nci1 seeds: 45 1225 1983
# others: 357 45 796
for seed in 45 1225 1983
do
    # copy train, val, test indices
    cp "${glg_data}/GIN/add/1.0/${seed}/train_indices.pkl" "${pg_data}/train_indices_size1.0_${seed}.pkl"
    cp "${glg_data}/GIN/add/1.0/${seed}/val_indices.pkl" "${pg_data}/val_indices_${seed}.pkl"
    cp "${glg_data}/GIN/add/1.0/${seed}/test_indices.pkl" "${pg_data}/test_indices_${seed}.pkl"

    # copy models
    cp "${glg_data}/GIN/add/1.0/${seed}/model.pt" "${pg_data}/model_${seed}_gin_sum.pt"
done
