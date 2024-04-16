from copy import deepcopy
from random import seed as rseed
from os import environ
import pickle
import torch
import torch_geometric as pyg

from gnns import GAT_Mutagenicity

SEED = 7
DEVICE = "cuda:3"
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
pyg.seed_everything(SEED)

def set_seed(seed: int = 42) -> None:
    rseed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    environ["PYTHONHASHSEED"] = str(seed)


TRAINED = False
PATH = f"../our_data/Mutagenicity"

dataset = pyg.datasets.TUDataset(name="Mutagenicity", root="../our_data/")

with open(f"{PATH}/train_indices.pkl", "rb") as file:
    train_indices = pickle.load(file)
with open(f"{PATH}/val_indices.pkl", "rb") as file:
    val_indices = pickle.load(file)
with open(f"{PATH}/test_indices.pkl", "rb") as file:
    test_indices = pickle.load(file)

train_loader = pyg.loader.DataLoader(dataset[train_indices], batch_size=128, shuffle=True)
val_loader   = pyg.loader.DataLoader(dataset[val_indices]  , batch_size=128, shuffle=False)
test_loader  = pyg.loader.DataLoader(dataset[test_indices] , batch_size=128, shuffle=False)

model = GAT_Mutagenicity()
model = model.to(DEVICE)

opt = torch.optim.Adam(params=model.parameters(), lr=1e-2)
loss_func = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:
        data = data.to(DEVICE)
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )
        loss = loss_func(out, data.y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return

def test(loader):
    model.eval()
    correct = 0
    loss = 0
    for data in loader:
        data = data.to(DEVICE)
        with torch.no_grad():
            out = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch,
            )
            loss += loss_func(out, data.y)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    acc = correct / len(loader.dataset)
    loss /= len(loader.dataset)
    return acc, loss

if TRAINED:
    best_weights = torch.load(f"{PATH}/model.pt")
else:
    best_epoch, best_loss, best_weights = 0, float("inf"), None
    for epoch in range(1, 1000):
        train()
        train_acc, train_loss = test(train_loader)
        val_acc, val_loss = test(val_loader)
        # print(
        #     f"Epoch {epoch:03d}"
        #     f" | Acc:  Train: {train_acc:.4f};  Val: {val_acc:.4f}"
        #     f" | Loss: Train: {train_loss:.4f}; Val: {val_loss:.4f}"
        # )
        if val_loss <= best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc
            best_weights = deepcopy(model.state_dict())
        # if epoch - best_epoch > 20:
        #     print("Early stopping!")
        #     break
    torch.save(best_weights, f"{PATH}/model.pt")
    print(f"\nBest epoch: {best_epoch} | val acc: {best_acc} | val loss: {best_loss}")
    print(f"Saved weights from epoch {best_epoch} to disk")

print(f"\nLoading weights from disk")
best_weights = torch.load(f"{PATH}/model.pt")
model.load_state_dict(best_weights)
test_acc, test_loss = test(test_loader)
print(f"Test acc: {test_acc:.4f}")
