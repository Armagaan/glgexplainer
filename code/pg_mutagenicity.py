"""Explain Mutagenicity"""
import os
import pickle
import shutil

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.explain import Explainer, ModelConfig, PGExplainer
from tqdm import tqdm

from gnns import GAT_Mutagenicity

SEED = 7
pyg.seed_everything(SEED)


# * ----- Data & Model
dataset = pyg.datasets.TUDataset(root="../our_data/", name="Mutagenicity")

path = "../our_data/Mutagenicity/"
with open(f"{path}/train_indices.pkl", "rb") as reader:
    train_indices = pickle.load(reader)
with open(f"{path}/test_indices.pkl", "rb") as reader:
    test_indices = pickle.load(reader)
val_indices = test_indices.copy()

train_loader = pyg.loader.DataLoader(dataset[train_indices], batch_size=64, shuffle=False)
test_loader = pyg.loader.DataLoader(dataset[test_indices], batch_size=64, shuffle=False)

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

train_pred_proba = predict_proba(train_loader)
test_pred_proba = predict_proba(test_loader)
val_pred_proba = test_pred_proba.clone()

train_pred = train_pred_proba.argmax(dim=1).tolist()
test_pred = test_pred_proba.argmax(dim=1).tolist()
val_pred = val_pred_proba.argmax(dim=1).tolist()


# * ----- PGExplainer
"""
>>> Values from PGExplainer paper
To train PGExplainer, we also adopt the Adam optimizer with the initial learning rate of 3.0 × 10−3.
The coefficient of size regularization is set to 0.05 and entropy regularization is 1.0. The epoch T
is set to 30 for all datasets. In this task, we find that relatively high temperatures work well in
practice. τ0 is set to 5.0 and τT is set to 2.0.

>>> Values from PGExplainer repository
lr 0.01
coff_t0 5.0
coff_te 1.0
coff_size 0.01
coff_ent 0.01
"""

coeffs = {
        'epochs': 30,
        'lr': 3e-3,
        'edge_size': 0.05, # coefficient of size reg
        'edge_ent': 1.0, # coefficient of entropy reg
        'temp': [5.0, 2.0], # temp_0, temp_T
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
        return_type="probs",
    ),
    node_mask_type=None,
    edge_mask_type="object",
    threshold_config=None
)


# * ----- Train PGExplainer
TRAINED = False
path_dir = '../our_data/local_explanations/PGExplainer/Mutagenicity/GCN_TF/'

if TRAINED == True:
    print("Loading trained explainer")
    explainer = torch.load(f"{path_dir}/explainer.pt")
else:
    model.eval()
    # Train in batches
    for epoch in tqdm(range(coeffs["epochs"]), colour="green", desc="Training"):
        for data in train_loader:
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                target=model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch
                ).argmax(dim=1),
                batch=data.batch,
            )
    torch.save(explainer, f"{path_dir}/explainer.pt")


# * ----- Evaluate the explainer
fid_plus_list = []
fid_minus_list = []

model.eval()
for data in train_loader:
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        target=model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        ).argmax(dim=1),
        batch=data.batch,
    )
    fid_plus, fid_minus = pyg.explain.metric.fidelity(
        explainer=explainer,
        explanation=explanation
    )
    fid_plus_list.append(fid_plus)
    fid_minus_list.append(fid_minus)

fid_plus_arr = np.array(fid_plus_list)
fid_minus_arr = np.array(fid_minus_list)

print(">>> Fidelity")
print(
    f"Plus | min:{fid_plus_arr.min()}"
    f" | mean: {fid_plus_arr.mean()}"
    f" | std: {fid_plus_arr.std()}"
    f" | max: {fid_plus_arr.max()}"
)
print(
    f"Minus | min:{fid_minus_arr.min()}"
    f" | mean: {fid_minus_arr.mean()}"
    f" | std: {fid_minus_arr.std()}"
    f" | max: {fid_minus_arr.max()}"
)


# * ----- Write the explanations to disk
for key in ["TRAIN", "TEST", "VAL"]:
    shutil.rmtree(f"{path_dir}/{key}")
    os.makedirs(f"{path_dir}/{key}/0")
    os.makedirs(f"{path_dir}/{key}/1")
    os.makedirs(f"{path_dir}/{key}/features")

    """
    shutil.rmtree(path_dir+key)
    os.makedirs(path_dir+key+"/0")
    os.makedirs(path_dir+key+"/1")
    os.makedirs(path_dir+key+"/features")
    """

for split in ["train", "val", "test"]:
    path_dir_split = f'{path_dir}/{split.upper()}/'
    for i, index in enumerate(tqdm(
        eval(f"{split}_indices"), colour="blue", desc=f"Explaining {split} graphs"
    )):
        graph = dataset[index]
        explanation = explainer(
            graph.x,
            graph.edge_index,
            edge_attr = graph.edge_attr,
            target=eval(f"{split}_pred_proba[i]"),
            index=None,
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

        """
        if graph.y.item() == 1:
            if eval(f"{split}_pred[i]") == 1:
                path = path_dir + f'1/1_{index}.pkl'
                path_f = path_dir + f'features/1_{index}.pkl'
            else:
                path = path_dir + f'1/0_{index}.pkl'
                path_f = path_dir + f'features/0_{index}.pkl'
        else:
            if eval(f"{split}_pred[i]") == 1:
                path = path_dir + f'0/1_{index}.pkl'
                path_f = path_dir + f'features/1_{index}.pkl'
            else:
                path = path_dir + f'0/0_{index}.pkl'
                path_f = path_dir + f'features/0_{index}.pkl'
        """

        with open(path, "wb") as file:
            pickle.dump(adj_arr, file)
        with open(path_f, "wb") as file:
            pickle.dump(node_features, file)
