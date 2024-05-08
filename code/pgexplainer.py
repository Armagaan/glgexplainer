import os
import pickle
import shutil
from argparse import ArgumentParser

import torch
import torch_geometric as pyg
from torch_geometric.explain import Explainer, ModelConfig, PGExplainer
from tqdm import tqdm

import gnns

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", type=str,
                    choices=["MUTAG", "Mutagenicity", "NCI1", "BAMultiShapes"], required=True)
parser.add_argument("-t", "--trained", action="store_true")
parser.add_argument("-s", "--seed", type=int, default=45)
args = parser.parse_args()
print(args)
pyg.seed_everything(args.seed)

print(">>> NOTE: Use 'pyg' conda environment for BAMultiShapes and NCI1. Use 'glg' for the rest.")

# * ----- Data & Model
if args.dataset == "BAMultiShapes":
    dataset = pyg.datasets.BAMultiShapesDataset(root="../our_data/BAMultiShapes")
else:
    dataset = pyg.datasets.TUDataset(root="../our_data/", name=args.dataset)

PATH = f"../our_data/{args.dataset}/"
with open(f"{PATH}/train_indices.pkl", "rb") as reader:
    train_indices = pickle.load(reader)
with open(f"{PATH}/val_indices.pkl", "rb") as reader:
    val_indices = pickle.load(reader)
with open(f"{PATH}/test_indices.pkl", "rb") as reader:
    test_indices = pickle.load(reader)

train_loader = pyg.loader.DataLoader(dataset[train_indices], batch_size=64, shuffle=False)
val_loader   = pyg.loader.DataLoader(dataset[val_indices],   batch_size=64, shuffle=False)
test_loader  = pyg.loader.DataLoader(dataset[test_indices],  batch_size=64, shuffle=False)

if args.dataset == "MUTAG":
    model = gnns.GAT_MUTAG()
elif args.dataset == "Mutagenicity":
    model = gnns.GAT_Mutagenicity()
elif args.dataset == "NCI1":
    model = gnns.GIN_NCI1()
elif args.dataset == "BAMultiShapes":
    model = gnns.GIN_BAMultiShapesDataset()

model.load_state_dict(torch.load(f"{PATH}/model.pt"))
model.eval()

def predict_proba(loader):
    pred_proba_list = []
    for data in loader:
        model_args = dict(x=data.x, edge_index=data.edge_index, batch=data.batch)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            model_args["edge_attr"] = data.edge_attr
        pred_proba = model(**model_args)
        # pred_proba is a list of lists
        # Hence merge the two rather than appending.
        pred_proba_list += pred_proba.tolist()
    return torch.Tensor(pred_proba_list)

def predict(loader):
    predictions = []
    for data in loader:
        model_args = dict(x=data.x, edge_index=data.edge_index, batch=data.batch)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            model_args["edge_attr"] = data.edge_attr
        out= model(**model_args)
        pred = out.argmax(dim=1)
        predictions += pred.tolist()
    return predictions

train_pred_proba = predict_proba(train_loader)
val_pred_proba   = predict_proba(val_loader)
test_pred_proba  = predict_proba(test_loader)

train_pred = train_pred_proba.argmax(dim=1).tolist()
val_pred = val_pred_proba.argmax(dim=1).tolist()
test_pred = test_pred_proba.argmax(dim=1).tolist()


# * ----- PGExplainer
"""
>>> Values from PGExplainer paper
To train PGExplainer, we also adopt the Adam optimizer with the initial learning rate of 3.0 × 10−3.
The coefficient of size regularization is set to 0.05 and entropy regularization is 1.0. The epoch T
is set to 30 for all datasets. In this task, we find that relatively high temperatures work well in
practice. τ0 is set to 5.0 and τT is set to 2.0.

epochs = 30
lr = 3e-3
t0, te = 5.0, 2.0
coff_size = 0.05
coff_ent = 1.0

>>> Values from PGExplainer repository
epochs = 30
lr = 0.01
t0, te = 5.0, 1.0
coff_size = 0.01
coff_ent = 0.01
"""
coeffs = {
        'epochs': 30,
        'lr': 3e-3,
        'temp': [5.0, 2.0], # temp_0, temp_T
        'edge_size': 0.05, # coefficient of size reg
        'edge_ent': 1.0, # coefficient of entropy reg
    }
explainer = Explainer(
    model=model,
    algorithm=PGExplainer(**coeffs),
    # 'PGExplainer' only supports phenomenon explanations
    # But we'll supply gnn predictions instead of ground truth. So it is the same.
    # CHECK : PGExplainer predicts model output inside itself. 
    explanation_type="phenomenon",
    model_config=ModelConfig(
        mode="multiclass_classification",
        task_level="graph",
        return_type="raw",
    ),
    node_mask_type=None,
    edge_mask_type="object",
    threshold_config=None,
)


# * ----- Train PGExplainer
PATH_DIR = f"../our_data/local_explanations/PGExplainer/{args.dataset}/GCN/"
os.makedirs(PATH_DIR, exist_ok=True)

if args.trained:
    print("Loading trained explainer")
    explainer = torch.load(f"{PATH_DIR}/explainer.pt")
else:
    model.eval()
    # Train in batches
    for epoch in tqdm(range(coeffs['epochs']), colour="green", desc="Training"):
        for data in train_loader:
            model_args = dict(x=data.x, edge_index=data.edge_index, batch=data.batch)
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                model_args["edge_attr"] = data.edge_attr
            target = model(**model_args).argmax(dim=1)
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                target=target,
                **model_args,
            )
    torch.save(explainer, f"{PATH_DIR}/explainer.pt")


# * ----- Write the explanations to disk
for key in ["TRAIN", "TEST", "VAL"]:
    shutil.rmtree(f"{PATH_DIR}/{key}", ignore_errors=True)
    os.makedirs(f"{PATH_DIR}/{key}/0")
    os.makedirs(f"{PATH_DIR}/{key}/1")
    os.makedirs(f"{PATH_DIR}/{key}/features")

for split in ["train", "val", "test"]:
    path_dir_split = f'{PATH_DIR}/{split.upper()}/'
    for i, index in enumerate(tqdm(
        eval(f"{split}_indices"), colour="blue", desc=f"Explaining {split} graphs"
    )):
        graph = dataset[index]
        graph_args = dict(x=graph.x, edge_index=graph.edge_index, batch=None)
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            graph_args["edge_attr"] = graph.edge_attr
        explanation = explainer(
            **graph_args,
            target=eval(f"{split}_pred_proba[i]"),
            index=None,
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
