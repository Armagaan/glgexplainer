from argparse import ArgumentParser
from pickle import dump

from local_explanations import *
import utils
import models
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

parser = ArgumentParser()
parser.add_argument("-t", "--trained", action="store_true",
                    help="Pass the flag if GLG has already been trained.")
parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "0", "1", "2", "3"])
parser.add_argument("-e", "--explainer", type=str, choices=["GNNExplainer", "PGExplainer"], required=True)
args = parser.parse_args()
print(args)

# * Read hyper-parameters and data
DATASET_NAME = "BAMultiShapes"
with open("../config/" + DATASET_NAME + "_params.json") as json_file:
    hyper_params = json.load(json_file)

PATH_GLG_MODEL = f"../our_data/trained_glg_models/{DATASET_NAME}_{args.explainer}.pt"
PATH_GLG_PLOT = f"../our_data/plots/{DATASET_NAME}_{args.explainer}.png"
PATH_CONCEPTS = f"../our_data/concepts/{DATASET_NAME}_{args.explainer}.pkl"

adjs_train, \
edge_weights_train, \
ori_classes_train, \
belonging_train, \
summary_predictions_train, \
le_classes_train = read_bamultishapes(evaluate_method=False, remove_mix=False, min_num_include=5, split="TRAIN")

adjs_val, \
edge_weights_val, \
ori_classes_val, \
belonging_val, \
summary_predictions_val, \
le_classes_val = read_bamultishapes(evaluate_method=False, remove_mix=False, min_num_include=5, split="VAL")

adjs_test, \
edge_weights_test, \
ori_classes_test, \
belonging_test, \
summary_predictions_test, \
le_classes_test = read_bamultishapes(evaluate_method=False, remove_mix=False, min_num_include=5, split="TEST")


# * Dataset
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
])     

dataset_train = utils.LocalExplanationsDataset("", adjs_train, "same", transform=transform, y=le_classes_train, belonging=belonging_train, task_y=ori_classes_train)
dataset_val   = utils.LocalExplanationsDataset("", adjs_val,   "same", transform=transform, y=le_classes_val,   belonging=belonging_val,   task_y=ori_classes_val)
dataset_test  = utils.LocalExplanationsDataset("", adjs_test,  "same", transform=transform, y=le_classes_test,  belonging=belonging_test,  task_y=ori_classes_test)


# * Train GLGExplainer
train_group_loader = utils.build_dataloader(dataset_train, belonging_train, num_input_graphs=128)
val_group_loader   = utils.build_dataloader(dataset_val,   belonging_val,   num_input_graphs=256)
test_group_loader  = utils.build_dataloader(dataset_test,  belonging_test,  num_input_graphs=256)


torch.manual_seed(42)
len_model = models.LEN(
    hyper_params["num_prototypes"],
    hyper_params["LEN_temperature"],
    remove_attention=hyper_params["remove_attention"]
).to(device)

le_model = models.LEEmbedder(
    num_features=hyper_params["num_le_features"],
    activation=hyper_params["activation"],
    num_hidden=hyper_params["dim_prototypes"]
).to(device)

expl = models.GLGExplainer(
    len_model,
    le_model, 
    device=device, 
    hyper_params=hyper_params,
    classes_names=bamultishapes_classes_names,
    dataset_name=DATASET_NAME,
    num_classes=4#len(train_group_loader.dataset.data.task_y.unique())
).to(device)

if not args.trained:
    print("\n>>> Training GLGExplainer")
    expl.iterate(train_group_loader, val_group_loader, plot=False)
    torch.save(expl.state_dict(), PATH_GLG_MODEL)
else:
    expl.load_state_dict(torch.load(PATH_GLG_MODEL))

expl.eval()

print(">>> Inspect train set")
expl.inspect(train_group_loader, plot=True)

print(">>> Inspect test set")
expl.inspect(test_group_loader, plot=True)


# * Materialize prototypes
print("\n>>> Visualize")
(
    x_train,
    emb,
    concepts_assignment,
    y_train_1h,
    le_classes,
    le_idxs,
    belonging
) = expl.get_concept_vector(test_group_loader, return_raw=True)

proto_names = {
    0: "BA",
    1: "Wheel",
    2: "Mix",
    3: "Grid",
    4: "House",
    5: "Grid",
}
torch.manual_seed(42)
fig = plt.figure(figsize=(15,5*1.8))
n = 0
for p in range(expl.hyper["num_prototypes"]):
    idxs = le_idxs[concepts_assignment.argmax(-1) == p]
    #idxs = idxs[torch.randperm(len(idxs))]    # random 
    sa = concepts_assignment[concepts_assignment.argmax(-1) == p]
    idxs = idxs[torch.argsort(sa[:, p], descending=True)]
    for ex in range(min(5, len(idxs))):
        n += 1
        ax = plt.subplot(expl.hyper["num_prototypes"],5,n)      
        G = to_networkx(dataset_test[int(idxs[ex])], to_undirected=True)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=20, ax=ax, node_color="orange")
        ax.axis("on")
        plt.box(False)
        
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
    try:
        idx = int(indices_p[torch.argsort(sa[:, p], descending=True)][0])
        # Get the subgraph.
        subgraph = dataset_train[idx]
        # Store it as a nx graph in a dictionary
        concepts[p] = glg_to_nx(subgraph)
    except IndexError:
        concepts[p] = None

with open(PATH_CONCEPTS, "wb") as file:
    dump(concepts, file)
