import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from local_explanations import *
import utils
import models
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

TRAINED = False # Is GLG trained?

# * Read hyper-parameters and data
DATASET_NAME = "BAMultiShapes"
with open("../config/" + DATASET_NAME + "_params.json") as json_file:
    hyper_params = json.load(json_file)

user = "us"
if user == "author":
    SAVE_GLG_MODEL_PATH = "../trained_models/bamultishapes.pt"
    SAVE_GLG_PLOT_PATH = "../plots/bamultishapes.png"
else:
    SAVE_GLG_MODEL_PATH = "../our_data/trained_models/bamultishapes.pt"
    SAVE_GLG_PLOT_PATH = "../our_data/plots/bamultishapes.png"


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    num_classes=len(train_group_loader.dataset.data.task_y.unique())
).to(device)

if not TRAINED:
    print("\n>>> Training GLGExplainer")
    expl.iterate(train_group_loader, val_group_loader, plot=False)
    torch.save(expl.state_dict(), SAVE_GLG_MODEL_PATH)
else:
    expl.load_state_dict(torch.load(SAVE_GLG_MODEL_PATH))

expl.eval()

print(">>> Inspect train set")
expl.inspect(train_group_loader, plot=True)

print(">>> Inspect test set")
expl.inspect(test_group_loader, plot=True)


# * Materialize prototypes
# change assign function to a non-discrete one just to compute distance between local expls. and prototypes
# useful to show the materialization of prototypes based on distance 
expl.hyper["assign_func"] = "sim"
(
    x_train,
    emb,
    concepts_assignement,
    y_train_1h,
    le_classes,
    le_idxs,
    belonging
) = expl.get_concept_vector(test_group_loader, return_raw=True)        
expl.hyper["assign_func"] = "discrete"

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
    idxs = le_idxs[concepts_assignement.argmax(-1) == p]
    #idxs = idxs[torch.randperm(len(idxs))]    # random 
    sa = concepts_assignement[concepts_assignement.argmax(-1) == p]
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

print(">>> Visualize")
plt.savefig(SAVE_GLG_PLOT_PATH)
