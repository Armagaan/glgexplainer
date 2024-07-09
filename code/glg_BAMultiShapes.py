from argparse import ArgumentParser
from pickle import dump, load
from time import process_time

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
parser.add_argument("-s", "--seed", type=int, required=True, help="Training sets from multiple seeds"
                    " are avaialbe. Supply the one to be used.")
parser.add_argument("-r", "--run", type=int, default=-1, help="GLG produces different results when"
                    "run multiple time. Use this to save the GLG's trained models under different"
                    "runs.")
parser.add_argument("--size", type=float, default=1.0, help="Percentage of training dataset.")
parser.add_argument("-a", "--arch", type=str, help="gnn architecture", choices=["gcn", "gin", "gat"], required=True)
parser.add_argument("-p", "--pooling", type=str, help="gnn pooling layer", choices=["sum", "mean", "max"], required=True)
args = parser.parse_args()
print(args)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# * Read hyper-parameters and data
DATASET_NAME = "BAMultiShapes"
with open("../config/" + DATASET_NAME + "_params.json") as json_file:
    hyper_params = json.load(json_file)

SUFFIX = f"{DATASET_NAME}_{args.explainer}_{args.arch}_{args.pooling}_size{args.size}_seed{args.seed}_run{args.run}"
PATH_GLG_MODEL = f"../our_data/trained_glg_models/{SUFFIX}.pt"
PATH_GLG_PLOT = f"../our_data/plots/{SUFFIX}.png"
PATH_GLG_PLOT_RANDOM = f"../our_data/plots_random/{SUFFIX}.png"
PATH_CONCEPTS = f"../our_data/concepts/{SUFFIX}.pkl"
PATH_FORMULAE = f"../our_data/formulae/{SUFFIX}.pkl"
PATH_TRAIN_PREDICTIONS = f"../our_data/glg_predictions/{SUFFIX}_train.pt"
PATH_TEST_PREDICTIONS = f"../our_data/glg_predictions/{SUFFIX}_test.pt"

start_time = process_time()

adjs_train, \
edge_weights_train, \
ori_classes_train, \
belonging_train, \
summary_predictions_train, \
le_classes_train = read_bamultishapes(evaluate_method=False,
                                      remove_mix=False,
                                      min_num_include=5,
                                      split="TRAIN",
                                      size=args.size,
                                      seed=args.seed,
                                      model=args.arch,
                                      pooling=args.pooling)

adjs_val, \
edge_weights_val, \
ori_classes_val, \
belonging_val, \
summary_predictions_val, \
le_classes_val = read_bamultishapes(evaluate_method=False,
                                    remove_mix=False,
                                    min_num_include=5,
                                    split="VAL",
                                    size=args.size,
                                    seed=args.seed,
                                    model=args.arch,
                                    pooling=args.pooling)

adjs_test, \
edge_weights_test, \
ori_classes_test, \
belonging_test, \
summary_predictions_test, \
le_classes_test = read_bamultishapes(evaluate_method=False,
                                     remove_mix=False,
                                     min_num_include=5,
                                     split="TEST",
                                     size=args.size,
                                     seed=args.seed,
                                     model=args.arch,
                                     pooling=args.pooling)


# * Dataset
device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
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
    num_classes=2 #len(train_group_loader.dataset.data.task_y.unique())
).to(device)

if not args.trained:
    print("\n>>> Training GLGExplainer")
    expl.iterate(train_group_loader, val_group_loader, plot=False)
    torch.save(expl.state_dict(), PATH_GLG_MODEL)
    with open(PATH_FORMULAE, "wb") as file:
        dump(dict(explanations=expl.explanations, explanations_raw=expl.explanations_raw), file)
else:
    expl.load_state_dict(torch.load(PATH_GLG_MODEL))
    with open(PATH_FORMULAE, "rb") as file:
        exps_dict = load(file)
        expl.explanations = exps_dict["explanations"]
        expl.explanations_raw = exps_dict["explanations_raw"]


expl.eval()

print(">>> Inspect train set")
train_pred = expl.inspect(train_group_loader, testing_formulae=True)
torch.save(train_pred, PATH_TRAIN_PREDICTIONS)

print(">>> Inspect test set")
test_pred = expl.inspect(test_group_loader, testing_formulae=True)
torch.save(test_pred, PATH_TEST_PREDICTIONS)

end_time = process_time()
print(f"[TIME]: {end_time - start_time} s.ms")

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
    0: "P0",
    1: "P1",
    2: "P2",
    3: "P3",
    4: "P4",
    5: "P5",
}

# * closest local explanations
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

# * random explanations
fig = plt.figure(figsize=(15,5*1.8))
n = 0
for p in range(expl.hyper["num_prototypes"]):
    # idxs = le_idxs[concepts_assignment.argmax(-1) == p]
    idxs = le_idxs[torch.randperm(len(le_idxs))]    # random 
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
    plt.ylabel(f"{proto_names[p]}", size=25, rotation="horizontal", labelpad=50)
plt.savefig(PATH_GLG_PLOT_RANDOM)


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
