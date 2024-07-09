"""
Compute classification metrics for GLG and GLG-iso. GLG has been commented out as
local_explanations.py omits some graphs failing certain conditions leading to a mismatch against the
labels. Hence, GLG's metrics are computed in glg*.py files directly and stored in glg's logs.
"""
from sklearn.metrics import classification_report
import torch

DATASETS = ["MUTAG", "Mutagenicity"] # ["BAMultiShapes", "NCI1"]
ARCHS = ["gat", "gcn", "gin"]
POOLS = ["sum", "max", "mean"]
SIZES = [0.05, 0.25, 0.5, 0.75, 1.0]

ARCHS = ["gat"]
POOLS = ["sum"]
SIZES = [1.0]
sample = 1.0 # Use ctrees and shapley values from this split of the training set.

for dataset in DATASETS:
    for arch in ARCHS:
        for pool in POOLS:
            for size in SIZES:
                if dataset == "NCI1":
                    SEEDS = [45, 1225, 1983]
                else:
                    SEEDS = [45, 357, 796]
                for seed in SEEDS:
                    file = f"{dataset}_PGExplainer_{arch}_{pool}_size{size}_seed{seed}_run0_test.pt"
                    print(file, ">" * 25)
                    try:
                        gnn_test_pred = torch.LongTensor(torch.load(f"../our_data/gnn_predictions/{file}"))
                        glg_test_pred = torch.load(f"../our_data/glg_predictions/{file}")
                        glg_iso_test_pred = torch.load(f"../our_data/glg_iso_predictions/{file}")
                    except FileNotFoundError:
                        print("FileNotFound")
                        continue

                    # print(gnn_test_pred.size())
                    # print(glg_test_pred.size())
                    # print(glg_iso_test_pred.size())

                    # glg sets predictions to -1 if they satisfy both formuale.
                    # This leads to division by 3 instead of 2 when averaging the metrics.
                    # To fix this, set -1 prediction opposite to its corresponding ground truth.
                    for i in range(len(gnn_test_pred)):
                        # if glg_test_pred[i] == -1:
                        #     glg_test_pred[i] =  abs(gnn_test_pred[i] - 1)
                        if glg_iso_test_pred[i] == -1:
                            glg_iso_test_pred[i] = abs(gnn_test_pred[i] - 1)

                    print("--- Test")
                    print("gnn_test_pred:", torch.unique(gnn_test_pred,  return_counts=True))

                    # print("--- GLG")
                    # print("glg_test_pred:", torch.unique(glg_test_pred, return_counts=True))
                    # print(classification_report(y_pred=glg_test_pred,  y_true=gnn_test_pred))

                    print("--- GLG iso")
                    print("glg_iso_test_pred:", torch.unique(glg_iso_test_pred, return_counts=True))
                    print(classification_report(y_pred=glg_iso_test_pred,  y_true=gnn_test_pred))
