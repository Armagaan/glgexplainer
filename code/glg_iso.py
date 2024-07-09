"""Compute the accuracy replacing distance a prototype by the subgraph closest to it."""
import os
from argparse import ArgumentParser
from pickle import load
from inspect import signature

import networkx as nx
import torch
import torch_geometric as pyg
from torch_explain.logic.metrics import test_explanations, test_explanation

import gnns

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices=["MUTAG", "Mutagenicity", "BAMultiShapes", "NCI1"])
parser.add_argument("-e", "--explainer", type=str, choices=["PGExplainer", "GNNExplainer"], required=True)
parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
parser.add_argument("-s", "--seed", type=int, required=True, help="Training sets from multiple seeds"
                    " are avaialbe. Supply the one to be used.")
parser.add_argument("-r", "--run", type=int, default=-1, help="GLG produces different results when"
                    "run multiple time. Use this to save the GLG's trained models under different"
                    "runs.")
parser.add_argument("--size", type=float, default=1.0, help="Percentage of training dataset.")
parser.add_argument("-a", "--arch", type=str, help="gnn architecture", choices=["gcn", "gin", "gat"], required=True)
parser.add_argument("-p", "--pooling", type=str, help="gnn pooling layer", choices=["sum", "mean", "max"], required=True)

args = parser.parse_args()

SUFFIX = f"{args.dataset}_{args.explainer}_{args.arch}_{args.pooling}_size{args.size}_seed{args.seed}_run{args.run}"
PATH_CONCEPTS = f"../our_data/concepts/{SUFFIX}.pkl"
PATH_FORMULAE = f"../our_data/formulae/{SUFFIX}.pkl"
PATH_MODEL = f"../our_data/{args.dataset}/model_{args.seed}_{args.arch}_{args.pooling}.pt"
PATH_TRAIN_PREDICTIONS = f"../our_data/glg_iso_predictions/{SUFFIX}_train.pt"
PATH_TEST_PREDICTIONS = f"../our_data/glg_iso_predictions/{SUFFIX}_test.pt"
PATH_GNN_TRAIN_PREDICTIONS = f"../our_data/gnn_predictions/{SUFFIX}_train.pt"
PATH_GNN_TEST_PREDICTIONS = f"../our_data/gnn_predictions/{SUFFIX}_test.pt"

if args.split == "train":
    PATH_DATA = f"../our_data/{args.dataset}/{args.split}_indices_size{args.size}_{args.seed}.pkl"
else:
    PATH_DATA = f"../our_data/{args.dataset}/{args.split}_indices_{args.seed}.pkl"


# * ----- Data
with open(PATH_CONCEPTS, "rb") as file:
    concepts = load(file)
    # for c in concepts.values():
    #     if c is not None:
    #         print(c.nodes)

if args.dataset == "BAMultiShapes":
    if args.arch == "gin":
        dataset = pyg.datasets.BAMultiShapesDataset(root="../our_data/gin_data/BAMultiShapes")
    else:
        dataset = pyg.datasets.BAMultiShapesDataset(root="../our_data/BAMultiShapes")
else:
    if args.arch == "gin":
        dataset = pyg.datasets.TUDataset(root="../our_data/gin_data/", name=args.dataset)
    else:
        dataset = pyg.datasets.TUDataset(root="../our_data/", name=args.dataset)

with open(PATH_DATA, "rb") as file:
    indices = load(file)

loader = pyg.loader.DataLoader(dataset[indices], batch_size=64, shuffle=False)


# * ----- Model
model = eval(f"gnns.{args.arch.upper()}_{args.dataset}(pooling='{args.pooling}')")

if not os.path.exists(PATH_MODEL):
    print("Model weights not found")
    exit(1)

try:
    state_dict = torch.load(PATH_MODEL)
    model.load_state_dict(state_dict)
except RuntimeError:
    print("Model trained on different pyg version")
    exit(1)

model.eval()
model_signature_params = [i.name for i in signature(model.forward).parameters.values()]

def predict_proba(loader):
    pred_proba_list = []
    for data in loader:
        if hasattr(data, "edge_attr") and "edge_attr" in model_signature_params and data.edge_attr is not None:
            pred_proba = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        else:
            pred_proba = model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )
        # pred_proba is a list of lists
        # Hence merge the two rather than appending.
        pred_proba_list += pred_proba.tolist()
    return torch.Tensor(pred_proba_list)

def predict(loader):
    predictions = []
    for data in loader:
        if hasattr(data, "edge_attr") and "edge_attr" in model_signature_params and data.edge_attr is not None:
            out= model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        else:
            out= model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )
        pred = out.argmax(dim=1)
        predictions += pred.tolist()
    return predictions

gnn_pred_proba  = predict_proba(loader)
gnn_pred = gnn_pred_proba.argmax(dim=1).tolist()


# * ----- Accuracy
def pyg_to_nx(graph):
    graph.x = graph.x.argmax(dim=-1)
    graph_nx = pyg.utils.to_networkx(
        data=graph,
        node_attrs=["x"],
        edge_attrs=None, # Ignore, as the concept graphs don't have edge features.
        to_undirected=True,
    )
    return graph_nx

concept_vectors = torch.zeros(size=(len(dataset[indices]), len(concepts)), dtype=torch.long)
# iterate over the graphs
for g_id, graph in enumerate(dataset[indices]):
    graph_nx = pyg_to_nx(graph)
    # iterate over the concepts
    for c_id in concepts:
        if concepts[c_id] is None: # A prototype that had zero subgraphs assigned to it.
            continue
        # if concept is subgraph isomorphic, set the corresponding entry to 1.
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            G1=graph_nx, G2=concepts[c_id],
            node_match= nx.algorithms.isomorphism.categorical_node_match(attr=["x"], default=[0]),
        )
        if matcher.subgraph_is_isomorphic():
            concept_vectors[g_id][c_id] = 1

with open(PATH_FORMULAE, "rb") as file:
    class_present = []
    exps_dict = load(file)["explanations"]
    formulae = []
    for i, f in enumerate(exps_dict):
        if f is None or f == "":
            continue
        formulae.append(f)
        class_present.append(i)
    print("Formulae:")
    for i, f in zip(class_present, formulae):
        print(i, ":", f)
    print()

if len(class_present) == 0:
    print("No formulae")
    exit(0)
elif len(class_present) == 1:
    target_class = class_present[0]
    acc, glg_iso_preds = test_explanation(
        formula=formulae[0],
        x=concept_vectors,
        y=torch.LongTensor([[1, 0] if i == 0 else [0, 1] for i in gnn_pred]),
        target_class=target_class,
        mask=torch.arange(len(indices), dtype=torch.long),
        material=False
    )
else:
    acc, glg_iso_preds = test_explanations(
        formulas=formulae,
        x=concept_vectors,
        y=torch.LongTensor([[1, 0] if i == 0 else [0, 1] for i in gnn_pred]),
        mask=torch.arange(len(indices), dtype=torch.long),
        material=False
    )

if args.split == "train":
    torch.save(glg_iso_preds, PATH_TRAIN_PREDICTIONS)
    torch.save(gnn_pred, PATH_GNN_TRAIN_PREDICTIONS)
else:
    torch.save(glg_iso_preds, PATH_TEST_PREDICTIONS)
    torch.save(gnn_pred, PATH_GNN_TEST_PREDICTIONS)
