from argparse import ArgumentParser
from pickle import load

import networkx as nx
from sklearn.metrics import f1_score
import torch
import torch_geometric as pyg
from torch_explain.logic.metrics import test_explanations

from gnns import GAT_Mutagenicity

parser = ArgumentParser()
parser.add_argument("-e", "--explainer", type=str, choices=["PGExplainer", "GNNExplainer"], required=True)
parser.add_argument("-s", "--split", type=str, choices=["train", "val", "test"], default="test")
args = parser.parse_args()


# * ----- Data
with open(f"../our_data/concepts/Mutagenicity_{args.explainer}.pkl", "rb") as file:
    concepts = load(file)

dataset = pyg.datasets.TUDataset(root="../our_data/", name="Mutagenicity")

PATH = "../our_data/Mutagenicity/"
with open(f"{PATH}/{args.split}_indices.pkl", "rb") as file:
    indices = load(file)

loader = pyg.loader.DataLoader(dataset[indices], batch_size=64, shuffle=False)


# * ----- Model
model = GAT_Mutagenicity()
model.load_state_dict(torch.load(f"../our_data/Mutagenicity/model.pt"))
model.eval()

def predict_proba(loader):
    pred_proba_list = []
    for data in loader:
        pred_proba = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        )
        # pred_proba is a list of lists
        # Hence merge the two rather than appending.
        pred_proba_list += pred_proba.tolist()
    return torch.Tensor(pred_proba_list)

def predict(loader):
    predictions = []
    for data in loader:
        out= model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        )
        pred = out.argmax(dim=1)
        predictions += pred.tolist()
    return predictions

pred_proba  = predict_proba(loader)
pred = pred_proba.argmax(dim=1).tolist()


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
        # if concept is subgraph isomorphic, set the corresponding entry to 1.
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            G1=graph_nx, G2=concepts[c_id],
            node_match= nx.algorithms.isomorphism.categorical_node_match(attr=["x"], default=[0]),
        )
        if matcher.subgraph_is_isomorphic():
            concept_vectors[g_id][c_id] = 1

formulae_PGExplainer = [
    "(~feature0000000001 & feature0000000000) | (feature0000000001 & feature0000000000)", # 0
    "(feature0000000001 & ~feature0000000000)" # only 1
]

formulae_GNNExplainer = [
    "(feature0000000000 & feature0000000001) | (~feature0000000000 & feature0000000001)", # 1
    "(feature0000000000 & ~feature0000000001)" # only 0
]

acc, preds = test_explanations(
    formulas=eval(f"formulae_{args.explainer}"),
    x=concept_vectors,
    y=torch.LongTensor([[1, 0] if i == 0 else [0, 1] for i in pred]),
    mask=torch.arange(len(indices), dtype=torch.long),
    material=False
)
f1_score_ = f1_score(y_true=pred, y_pred=preds.tolist(), average="weighted")
# for i,j in zip(concept_vectors, preds):
#     print(i, j)

c0, c1 = torch.LongTensor(pred).unique(return_counts=True)[1].tolist()
print(f"Class distribution: {c0 / (c0 + c1):.4f}, {c1 / (c0 + c1):.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 score: {f1_score_:.4f}")
