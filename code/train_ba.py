import pickle
import torch
import torch_geometric as pyg

from gnns import GIN_BAMultiShapes

pyg.seed_everything(7)
TRAINED = True

path = f"../our_data/BAMultiShapes/"
dataset = pyg.datasets.BAMultiShapesDataset(root=path)

with open(f"{path}/train_indices.pkl", "rb") as file:
    train_indices = pickle.load(file)
with open(f"{path}/test_indices.pkl", "rb") as file:
    test_indices = pickle.load(file)
split_idx = int(0.8 * len(train_indices))
val_indices = train_indices[split_idx:]
train_indices = train_indices[:split_idx]

train_loader = pyg.loader.DataLoader(dataset[train_indices], batch_size=64, shuffle=False)
val_loader   = pyg.loader.DataLoader(dataset[val_indices]  , batch_size=64, shuffle=False)
test_loader  = pyg.loader.DataLoader(dataset[test_indices] , batch_size=64, shuffle=False)

model = GIN_BAMultiShapes()

opt = torch.optim.Adam(params=model.parameters(), lr=1e-2)
loss_func = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:
        out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
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
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            loss += loss_func(out, data.y)

        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc = correct / len(loader.dataset)
    loss /= len(loader.dataset)
    return acc, loss

if TRAINED:
    best_weights = torch.load(f"{path}/model.pt")
else:
    best_epoch, best_loss, best_weights = 0, float("inf"), None
    for epoch in range(1, 101):
        train()
        train_acc, train_loss = test(train_loader)
        val_acc, val_loss = test(val_loader)
        print(
            f"Epoch {epoch:03d}"
            f" | Acc:  Train: {train_acc:.4f};  Val: {val_acc:.4f}"
            f" | Loss: Train: {train_loss:.4f}; Val: {val_loss:.4f}"
        )
        if val_loss <= best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_weights = model.state_dict().copy()
    torch.save(best_weights, f"{path}/model.pt")
    print(f"\nSaved weights from epoch {best_epoch} to disk\n")

print(f"Loading weights")
model.load_state_dict(best_weights)
test_acc, test_loss = test(test_loader)
print(f"Test acc: {test_acc:.4f}")
