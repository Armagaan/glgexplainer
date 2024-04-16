"""Explain Mutagenicity"""
import os
import pickle
import shutil
from argparse import ArgumentParser

import torch
import torch_geometric as pyg
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from tqdm import tqdm

from gnns import GAT_Mutagenicity

parser = ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=45)
args = parser.parse_args()
print(args)
pyg.seed_everything(args.seed)


# * ----- Data & Model
dataset = pyg.datasets.TUDataset(root="../our_data/", name="Mutagenicity")

PATH = "../our_data/Mutagenicity/"
with open(f"{PATH}/train_indices.pkl", "rb") as file:
    train_indices = pickle.load(file)
with open(f"{PATH}/val_indices.pkl", "rb") as file:
    val_indices = pickle.load(file)
with open(f"{PATH}/test_indices.pkl", "rb") as file:
    test_indices = pickle.load(file)

train_loader = pyg.loader.DataLoader(dataset[train_indices], batch_size=64, shuffle=False)
val_loader   = pyg.loader.DataLoader(dataset[val_indices],   batch_size=64, shuffle=False)
test_loader  = pyg.loader.DataLoader(dataset[test_indices],  batch_size=64, shuffle=False)

model = GAT_Mutagenicity()
model.load_state_dict(torch.load(f"{PATH}/model.pt"))
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

train_pred_proba = predict_proba(train_loader)
val_pred_proba = predict_proba(val_loader)
test_pred_proba = predict_proba(test_loader)

train_pred = train_pred_proba.argmax(dim=1).tolist()
val_pred = val_pred_proba.argmax(dim=1).tolist()
test_pred = test_pred_proba.argmax(dim=1).tolist()


# * ----- GNNExplainer
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(),
    explanation_type="model",
    model_config=ModelConfig(
        mode="multiclass_classification",
        task_level="graph",
        return_type="raw"
    ),
    node_mask_type=None,
    edge_mask_type="object",
    threshold_config=None,
)

"""
# * ----- Explain one graph
idx = 10
graph = dataset[idx]
explanation = explainer(
    x=graph.x,
    edge_index=graph.edge_index,
    target=None, # None, documentation instructs target=None when explanation_type="model"
    index=None, # None, since we're doing graph classification
    # additional kwargs for the model:
    edge_attr=graph.edge_attr,
    batch=None,
)
"""


# * ----- Write the explanations to disk
path_dir = '../our_data/local_explanations/GNNExplainer/Mutagenicity/GCN_TF/'

for key in ["TRAIN", "TEST", "VAL"]:
    shutil.rmtree(f"{path_dir}/{key}", ignore_errors=True)
    os.makedirs(f"{path_dir}/{key}/0")
    os.makedirs(f"{path_dir}/{key}/1")
    os.makedirs(f"{path_dir}/{key}/features")

for split in ["train", "val", "test"]:
    path_dir_split = f'{path_dir}/{split.upper()}/'
    for i, index in enumerate(tqdm(
        eval(f"{split}_indices"), colour="blue", desc=f"Explaining {split} graphs"
    )):
        graph = dataset[index]
        explanation = explainer(
            x=graph.x,
            edge_index=graph.edge_index,
            target=None, # None, documentation instructs target=None when explanation_type="model"
            index=None, # None, since we're doing graph classification
            # additional kwargs for the model:
            edge_attr=graph.edge_attr,
            batch=None,
        )
        adj_tensor = pyg.utils.to_dense_adj(
            explanation.edge_index,
            edge_attr=explanation.edge_mask
        ).squeeze(0)
        adj_arr = adj_tensor.numpy()
        node_features = graph.x.numpy()

        ground_truth = graph.y.item()
        gnn_pred = eval(f"{split}_pred[i]")
        path = f"{path_dir_split}/{ground_truth}/{gnn_pred}_{index}.pkl"
        path_f = f"{path_dir_split}/features/{gnn_pred}_{index}.pkl"

        with open(path, "wb") as file:
            pickle.dump(adj_arr, file)
        with open(path_f, "wb") as file:
            pickle.dump(node_features, file)
