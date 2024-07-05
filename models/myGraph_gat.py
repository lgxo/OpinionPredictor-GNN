import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, ModuleList
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class myGraph_gat(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, layers, readout_type, is_selfloop, is_edge_attr, dropout=0.5) -> None:
        super(myGraph_gat, self).__init__()
        self.readout_type = readout_type
        self.is_edge_attr = is_edge_attr
        self.layers = layers

        self.conv_lst = ModuleList(
            [GATConv(
                in_features,
                hidden_features,
                add_self_loops=is_selfloop,
            )] + [
                GATConv(
                    hidden_features,
                    hidden_features,
                    add_self_loops=is_selfloop,
                ) for _ in range(self.layers-1)
            ]
        )

        self.drop = Dropout(dropout)
        self.lin1 = Linear(hidden_features, out_features)

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        if self.is_edge_attr:
            for module in self.conv_lst[:-1]:
                x = module(x, edge_index, edge_attr)
                x = x.relu()
            x = self.conv_lst[-1](x, edge_index, edge_attr)
        else:
            for module in self.conv_lst[:-1]:
                x = module(x, edge_index)
                x = x.relu()
            x = self.conv_lst[-1](x, edge_index)

        if self.readout_type == "global_mean_pool":
            x = global_mean_pool(x, batch)
        elif self.readout_type == "global_max_pool":
            x = global_max_pool(x, batch)
        else:
            raise ValueError("Invalid readout type.")

        x = self.drop(x)
        x = self.lin1(x)

        return {
            "logits": x,
        }
