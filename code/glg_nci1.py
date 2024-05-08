import json
import os
from argparse import ArgumentParser
from pickle import dump

import matplotlib.pyplot as plt
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader

import utils
import models
from local_explanations import *

parser = ArgumentParser()
parser.add_argument("-t", "--trained", action="store_true",
                    help="Pass the flag if GLG has already been trained.")
parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "0", "1", "2", "3"])
parser.add_argument("-e", "--explainer", type=str, choices=["GNNExplainer", "PGExplainer"], required=True)
args = parser.parse_args()
print(args)


# * Read hyper-parameters and data
DATASET_NAME = "NCI1"
with open("../config/" + DATASET_NAME + "_params.json") as json_file:
    hyper_params = json.load(json_file)

PATH_GLG_MODEL = f"../our_data/trained_glg_models/{DATASET_NAME}_{args.explainer}.pt"
PATH_GLG_PLOT = f"../our_data/plots/{DATASET_NAME}_{args.explainer}.png"
PATH_CONCEPTS = f"../our_data/concepts/{DATASET_NAME}_{args.explainer}.pkl"

from local_explanations import read_nci1


# * ----- Data
adjs_train , \
edge_weights_train , \
ori_adjs_train , \
ori_classes_train , \
belonging_train , \
summary_predictions_train , \
le_classes_train ,\
embeddings_train = read_nci1(explainer=args.explainer,
                                     evaluate_method=False, 
                                     split="TRAIN")

adjs_val , \
edge_weights_val , \
ori_adjs_val , \
ori_classes_val , \
belonging_val , \
summary_predictions_val , \
le_classes_val ,\
embeddings_val = read_nci1(explainer=args.explainer,
                                   evaluate_method=False, 
                                   split="VAL")

adjs_test , \
edge_weights_test , \
ori_adjs_test , \
ori_classes_test , \
belonging_test , \
summary_predictions_test , \
le_classes_test ,\
embeddings_test = read_nci1(explainer=args.explainer,
                                    evaluate_method=False, 
                                    split="TEST")

device = torch.device(f'cuda:{args.device}' if args.device != 'cpu' else 'cpu')
transform = None

dataset_train = utils.LocalExplanationsDataset("", adjs_train, "embeddings", transform=transform, y=le_classes_train, belonging=belonging_train, task_y=ori_classes_train, precomputed_embeddings=embeddings_train)
dataset_val   = utils.LocalExplanationsDataset("", adjs_val,   "embeddings", transform=transform, y=le_classes_val,   belonging=belonging_val,   task_y=ori_classes_val,   precomputed_embeddings=embeddings_val)
dataset_test  = utils.LocalExplanationsDataset("", adjs_test,  "embeddings", transform=transform, y=le_classes_test,  belonging=belonging_test,  task_y=ori_classes_test,  precomputed_embeddings=embeddings_test)

"""
# y: le_class created through isomorphism tests.
# task_y: gnn prediction
# le_id: A connected component's id
# graph_id: Graph id of the graph to which the connected component belongs.
"""


# * ----- Train GLGExplainer
train_group_loader = utils.build_dataloader(dataset_train, belonging_train, num_input_graphs=128)
val_group_loader   = utils.build_dataloader(dataset_val,   belonging_val,   num_input_graphs=256)
test_group_loader  = utils.build_dataloader(dataset_test,  belonging_test,  num_input_graphs=256)

torch.manual_seed(42)

# Computes the graph embedding of single, disconnected local explanations.
le_model = models.LEEmbedder(
    num_features=hyper_params["num_le_features"], 
    activation=hyper_params["activation"], 
    num_hidden=hyper_params["dim_prototypes"]
).to(device)

# e-len model. Generates the truth table.
len_model = models.LEN(
    hyper_params["num_prototypes"], 
    hyper_params["LEN_temperature"], 
    remove_attention=hyper_params["remove_attention"]
).to(device)

expl = models.GLGExplainer(
    len_model, 
    le_model, 
    device, 
    hyper_params=hyper_params,
    classes_names=["C0", "C1"],
    dataset_name=DATASET_NAME,
    num_classes=2 # len(train_group_loader.dataset.data.task_y.unique()) #! This is not safe.
).to(device)


if not args.trained:
    print("\n>>> Training GLGExplainer")
    expl.iterate(train_group_loader, val_group_loader, plot=False)
    torch.save(expl.state_dict(), PATH_GLG_MODEL)
else:
    expl.load_state_dict(torch.load(PATH_GLG_MODEL))

expl.eval()

print("\n>>> Inspect train set")
expl.inspect(train_group_loader, plot=True)

print("\n>>> Inspect test set")
expl.inspect(test_group_loader, plot=True)


# * ----- Materialize Prototypes
print("\n>>> Visualization")

# change assign function to a non-discrete one just to compute distance between local expls. and prototypes
# useful to show the materialization of prototypes based on distance 
# expl.hyper["assign_func"] = "sim"

loader = DataLoader(dataset_train, batch_size=64, shuffle=False)
# Iterate over graphs in dataset_train
le_embeddings = torch.tensor([], device=device)
le_idxs = torch.tensor([], device=device)
for data in loader:
    data = data.to(device)
    # Pass them through the le_model to get their embedding
    embs = expl.le_model(x=data.x, edge_index=data.edge_index, batch=data.batch)
    le_embeddings = torch.concat([le_embeddings, embs], dim=0)
    le_idxs = torch.concat([le_idxs, data.le_id], dim=0)

# Calculate and store the distance to expl.prototypes
concepts_assignment = utils.prototype_assignement(
    assign_func=hyper_params["assign_func"],
    le_embeddings=le_embeddings,
    prototype_vectors=expl.prototype_vectors,
    temp=1
)

proto_names = {
    0: "C0",
    1: "C1",
}

torch.manual_seed(42)
fig = plt.figure(figsize=(17,4))
n = 0
for p in range(expl.hyper["num_prototypes"]):
    idxs = le_idxs[concepts_assignment.argmax(-1) == p]
    # idxs = idxs[torch.randperm(len(idxs))] # for random examples
    sa = concepts_assignment[concepts_assignment.argmax(-1) == p]
    idxs = idxs[torch.argsort(sa[:, p], descending=True)]

    for ex in range(5):
        n += 1
        plt.subplot(expl.hyper["num_prototypes"],5,n)
        utils.plot_molecule(dataset_train[int(idxs[ex])], composite_plot=True)

for p in range(expl.hyper["num_prototypes"]):
    plt.subplot(expl.hyper["num_prototypes"],5,5*p + 1)
    plt.ylabel(f"$P_{p}$\n {proto_names[p]}", size=25, rotation="horizontal", labelpad=50)
plt.savefig(PATH_GLG_PLOT)


# * ----- For calculating accuracy, find the closest local explanation to a prototype as its replacement
def glg_to_nx(subgraph):
    """Convert pyg objects to networkx graphs"""
    subgraph.x = subgraph.x.argmax(dim=-1)
    subgraph_nx = to_networkx(
        data=subgraph,
        node_attrs=["x"],
        # The local explanations don't save the edge features.
        # Here, the edge_attrs are the weights assigned by the explainer to individual edges.
        edge_attrs=None,
        to_undirected=True,
    )    
    return subgraph_nx

concepts = {}
# Iterate over the #protypes.
for p in range(expl.hyper["num_prototypes"]):
    # Indices of subgraphs belonging to prototype p's cluster.
    indices_p = le_idxs[concepts_assignment.argmax(-1) == p]
    # Soft assignments of the local explantions to prototype p.
    sa = concepts_assignment[concepts_assignment.argmax(-1) == p]
    # Get the index of the local explanation closest to prototype p.
    idx = int(indices_p[torch.argsort(sa[:, p], descending=True)][0])
    # Get the subgraph.
    subgraph = dataset_train[idx]
    # Store it as a nx graph in a dictionary
    concepts[p] = glg_to_nx(subgraph)

with open(PATH_CONCEPTS, "wb") as file:
    dump(concepts, file)
