"""The loss aggregator."""

import torch
from util_tools import is_dict, get_attr_from, get_valid_args, is_tensor, get_ddp_module
from util_tools import Odict
from util_tools import get_msg_mgr
from .losses.triplet import TripletLoss


class TripletLossAggregator():
    def __init__(self, margin = 0.2, loss_term_weight = 1.0) -> None:
        """
        Initialize the loss aggregator.

        Args:
            loss_cfg: Config of losses. List for multiple losses.
        """
        self.loss_func = TripletLoss(margin, loss_term_weight).cuda()

    def __call__(self, training_feats):
        """Compute the sum of all losses.

        The input is a dict of features. The key is the name of loss and the value is the feature and label. If the key not in 
        built losses and the value is torch.Tensor, then it is the computed loss to be added loss_sum.

        Args:
            training_feats: A dict of features. The same as the output["training_feat"] of the model.
        """
        loss_sum = .0
        loss_info = Odict()

        for k, v in training_feats.items():
            loss_func = self.loss_func
            loss, info = loss_func(**v)
            for name, value in info.items():
                loss_info['scalar/%s/%s' % (k, name)] = value
            loss = loss.mean() * loss_func.loss_term_weight
            loss_sum += loss

        return loss_sum, loss_info
