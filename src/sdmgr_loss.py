import torch
from mmdet.models.builder import LOSSES
from mmdet.models.losses import accuracy
from torch import nn

import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input,
         target, ignore_index=self.ignore_index, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


@LOSSES.register_module()
class SDMGRLoss(nn.Module):
    """The implementation the loss of key information extraction proposed in
    the paper: Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction.

    https://arxiv.org/abs/2103.14470.
    """

    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=-100):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def forward(self, node_preds, edge_preds, gts):
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].contiguous().view(-1))
        node_gts = torch.cat(node_gts).long()
        edge_gts = torch.cat(edge_gts).long()

        node_valids = torch.nonzero(
            node_gts != self.ignore, as_tuple=False).view(-1)
        edge_valids = torch.nonzero(edge_gts != -1, as_tuple=False).view(-1)
        return dict(
            loss_node=self.node_weight * self.loss_node(node_preds, node_gts),
            loss_edge=self.edge_weight * self.loss_edge(edge_preds, edge_gts),
            acc_node=accuracy(node_preds[node_valids], node_gts[node_valids]),
            acc_edge=accuracy(edge_preds[edge_valids], edge_gts[edge_valids]))
