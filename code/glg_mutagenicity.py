
import json
import os

import matplotlib.pyplot as plt
import torch
import torch_geometric.transforms as T

import utils
import models
from local_explanations import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TRAINED = True # Is GLG trained?

# * Read hyper-parameters and data
DATASET_NAME = "Mutagenicity"
with open("../config/" + DATASET_NAME + "_params.json") as json_file:
    hyper_params = json.load(json_file)

user = "us"
if user == "author":
    SAVE_GLG_MODEL_PATH = "../trained_models/mutagenicity_with_nh2.pt"
    SAVE_GLG_PLOT_PATH = "../plots/mutagenicity_with_nh2.png"
    manual_cut = hyper_params["manual_cut"]
else:
    SAVE_GLG_MODEL_PATH = "../our_data/trained_models/mutagenicity_with_nh2.pt"
    SAVE_GLG_PLOT_PATH = "../our_data/plots/mutagenicity_with_nh2.png"
    manual_cut = None # Armgaan: Our explanations have smaller values than the authors'


from local_explanations import read_mutagenicity

adjs_train , \
edge_weights_train , \
ori_adjs_train , \
ori_classes_train , \
belonging_train , \
summary_predictions_train , \
le_classes_train ,\
embeddings_train = read_mutagenicity(evaluate_method=False, 
                                     manual_cut=manual_cut,
                                     split="TRAIN")

adjs_val , \
edge_weights_val , \
ori_adjs_val , \
ori_classes_val , \
belonging_val , \
summary_predictions_val , \
le_classes_val ,\
embeddings_val = read_mutagenicity(evaluate_method=False, 
                                   manual_cut=manual_cut,
                                   split="VAL")

adjs_test , \
edge_weights_test , \
ori_adjs_test , \
ori_classes_test , \
belonging_test , \
summary_predictions_test , \
le_classes_test ,\
embeddings_test = read_mutagenicity(evaluate_method=False, 
                                    manual_cut=manual_cut,
                                    split="TEST")

device = "cpu" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = None

#  ERROR : adjs_train is empty. 
dataset_train = utils.LocalExplanationsDataset("", adjs_train, "embeddings", transform=transform, y=le_classes_train, belonging=belonging_train, task_y=ori_classes_train, precomputed_embeddings=embeddings_train)
dataset_val   = utils.LocalExplanationsDataset("", adjs_val,   "embeddings", transform=transform, y=le_classes_val,   belonging=belonging_val,   task_y=ori_classes_val,   precomputed_embeddings=embeddings_val)
dataset_test  = utils.LocalExplanationsDataset("", adjs_test,  "embeddings", transform=transform, y=le_classes_test,  belonging=belonging_test,  task_y=ori_classes_test,  precomputed_embeddings=embeddings_test)

# print(dataset_train[0])
"""
# y: le_class created through isomorphism tests.
# task_y: gnn prediction
# le_id: A connected component's id
# graph_id: Graph id of the graph to which the connected component belongs.
"""

# * Train GLGExplainer
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
    classes_names=mutag_classes_names,
    dataset_name=DATASET_NAME,
    num_classes=len(train_group_loader.dataset.data.task_y.unique()) # This is not safe.
).to(device)


if not TRAINED:
    print("\n>>> Training GLGExplainer")
    expl.iterate(train_group_loader, val_group_loader, plot=False)
    torch.save(expl.state_dict(), SAVE_GLG_MODEL_PATH)
else:
    expl.load_state_dict(torch.load(SAVE_GLG_MODEL_PATH))

expl.eval()

print("\n>>> Inspect train set")
expl.inspect(train_group_loader, plot=True)

print("\n>>> Inspect test set")
expl.inspect(test_group_loader, plot=True)


# * Materialize Prototypes
print("\n>>> Visualization")
# change assign function to a non-discrete one just to compute distance between local expls. and prototypes
# useful to show the materialization of prototypes based on distance 
expl = expl
expl.hyper["assign_func"] = "sim"

(
    x_train,
    emb,
    concepts_assignement,
    y_train_1h,
    le_classes,
    le_idxs,
    belonging
) = expl.get_concept_vector(train_group_loader, return_raw=True)

expl.hyper["assign_func"] = "discrete"

proto_names = {
    0: "Others",
    1: "NO2",
}

torch.manual_seed(42)

fig = plt.figure(figsize=(17,4))
n = 0
for p in range(expl.hyper["num_prototypes"]):
    idxs = le_idxs[concepts_assignement.argmax(-1) == p]
    idxs = idxs[torch.randperm(len(idxs))] # for random examples
    sa = concepts_assignement[concepts_assignement.argmax(-1) == p]
    idxs = idxs[torch.argsort(sa[:, p], descending=True)]

    for ex in range(5):
        n += 1
        plt.subplot(expl.hyper["num_prototypes"],5,n)        
        utils.plot_molecule(dataset_train[int(idxs[ex])], composite_plot=True)

for p in range(expl.hyper["num_prototypes"]):
    plt.subplot(expl.hyper["num_prototypes"],5,5*p + 1)
    plt.ylabel(f"$P_{p}$\n {proto_names[p]}", size=25, rotation="horizontal", labelpad=50)

plt.savefig(SAVE_GLG_PLOT_PATH)
