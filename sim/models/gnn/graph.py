from typing import Any, Dict, List
import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing

class myGNN(MessagePassing):
    def __init__(self, aggr: str | List[str] | Aggregation | None = 'sum', *, aggr_kwargs: Dict[str, Any] | None = None, flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1) -> None:
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)