"""
These models have a softmax layer unlike our method.
This is done as PGExplainer expects probabilities
"""
import torch
from torch.nn.functional import dropout
import torch_geometric as pyg


class GNN_MUTAG(torch.nn.Module):
    def __init__(self, hidden_dim = 32):
        torch.manual_seed(7)
        super().__init__()
        num_node_features=7
        num_edge_features=4
        num_classes=2
        gat_args = dict(
            edge_dim = num_edge_features,
            add_self_loops = False,
        )
        self.conv1 = pyg.nn.conv.GATConv(num_node_features, hidden_dim, **gat_args)
        self.conv2 = pyg.nn.conv.GATConv(hidden_dim, hidden_dim, **gat_args)
        self.conv3 = pyg.nn.conv.GATConv(hidden_dim, hidden_dim, **gat_args)
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch): 
        out = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        out = self.conv2(out, edge_index, edge_attr=edge_attr).relu()
        node_embs = self.conv3(out, edge_index, edge_attr=edge_attr)

        # emb = pyg.nn.pool.global_add_pool(out, batch)
        graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)

        out = dropout(graph_emb, p=0.5, training=self.training)
        out = self.lin(out)
        out = torch.nn.functional.softmax(out, dim=1) # * Added as PG expects probabilities

        return out


class GNN_Mutagenicity(torch.nn.Module):
    def __init__(self, hidden_dim = 32):
        torch.manual_seed(7)
        super().__init__()
        num_node_features=14
        num_edge_features=3
        num_classes=2
        gat_args = dict(
            edge_dim = num_edge_features,
            add_self_loops = False,
        )
        self.conv1 = pyg.nn.conv.GATConv(num_node_features, hidden_dim, **gat_args)
        self.conv2 = pyg.nn.conv.GATConv(hidden_dim, hidden_dim, **gat_args)
        self.conv3 = pyg.nn.conv.GATConv(hidden_dim, hidden_dim, **gat_args)
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch): 
        out = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        out = self.conv2(out, edge_index, edge_attr=edge_attr).relu()
        node_embs = self.conv3(out, edge_index, edge_attr=edge_attr)

        graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)

        out = dropout(graph_emb, p=0.5, training=self.training)
        out = self.lin(out)
        out = torch.nn.functional.softmax(out, dim=1) # * Added as PG expects probabilities

        return out


class GNN_BAMultiShapesDataset(torch.nn.Module):
    def __init__(self, hidden_dim = 32):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops = False,
        )
        num_node_features = 10 
        num_classes = 2
        self.conv1 = pyg.nn.conv.GCNConv(num_node_features, hidden_dim, **gat_args)
        self.conv2 = pyg.nn.conv.GCNConv(hidden_dim, hidden_dim, **gat_args)
        self.conv3 = pyg.nn.conv.GCNConv(hidden_dim, hidden_dim, **gat_args)
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch): 
        out = self.conv1(x, edge_index).relu()
        out = self.conv2(out, edge_index).relu()
        node_embs = self.conv3(out, edge_index)

        graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)

        out = dropout(graph_emb, p=0.5, training=self.training)
        out = self.lin(out)
        out = torch.nn.functional.softmax(out, dim=1) # * Added as PG expects probabilities

        return out
