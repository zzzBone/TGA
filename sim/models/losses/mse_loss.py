import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

def physical_error(pred, inputs, dt):
    state = inputs['state'].squeeze(1).transpose(-1, -2)
    rigid_mask = inputs['rigid_mask'].squeeze(1)
    fluid_mask = inputs['fluid_mask'].squeeze(1)
    # bs, 2, 3, 3
    stat = inputs['stat']
    # bs, 3
    mean_p = stat[:, 0, :, 0]
    std_p = stat[:, 0, :, 1]
    mean_v = stat[:, 1, :, 0]
    std_v = stat[:, 1, :, 1]

    rigid_mask = rigid_mask.unsqueeze(2)
    fluid_mask = fluid_mask.unsqueeze(2)

    p_0 = (state[:, :, :3] * std_p.unsqueeze(1) + mean_p.unsqueeze(1))
    v_pred = (pred - p_0) / dt

    v_gt = state[:, :, 3:6] * std_v.unsqueeze(1) + mean_v.unsqueeze(1)

    loss = F.mse_loss(v_pred ** 2, v_gt ** 2, reduction='none')

    return loss


def mean_squared_error(pred, label, dt, weight=None, reduction='mean', avg_factor=None, **kwargs):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    inputs = kwargs.get('inputs', None)
    
    # element-wise losses
    # pred : [bs, n, c]  label: [bs, n, c]
    loss = F.mse_loss(pred, label, reduction='none')
    # loss2 = physical_error(pred, inputs, dt)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class MSELoss(nn.Module):
    """Cross entropy loss

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = mean_squared_error

    def forward(self,
                cls_score,
                label,
                dt,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.criterion(
            cls_score,
            label,
            dt,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
