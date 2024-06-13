"""
This class implements Exponential Moving Average (EMA) for updating parameters.

Usage::

    ema = EMA(label, num_classes=None, alpha=0.9)
    ema.update(data, index, curve=None, iter_range=None, step=None)
    ema.max_loss(label)

Arguments::

    label (torch.Tensor): The label tensor.
    num_classes (int, optional): Number of classes. Defaults to None.
    alpha (float, optional): The smoothing factor. Defaults to 0.9.
    data (torch.Tensor): The data tensor.
    index (torch.Tensor): The index tensor.
    curve (torch.Tensor, optional): The curve tensor. Defaults to None.
    iter_range (int, optional): The range of iterations. Defaults to None.
    step (int, optional): The step size. Defaults to None.
    Modified from https://github.com/alinlab/LfF/blob/master/util.py
"""

import io
import torch
import numpy as np
import torch.nn as nn

class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9):
        """
        Initialize the EMA object.

        Args:
            label (torch.Tensor): The label tensor.
            num_classes (int, optional): Number of classes. Defaults to None.
            alpha (float, optional): The smoothing factor. Defaults to 0.9.
        """
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        """
        Update the parameters.

        Args:
            data (torch.Tensor): The data tensor.
            index (torch.Tensor): The index tensor.
            curve (torch.Tensor, optional): The curve tensor. Defaults to None.
            iter_range (int, optional): The range of iterations. Defaults to None.
            step (int, optional): The step size. Defaults to None.

        Returns:
            None
        """
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        """
        Get the maximum loss.

        Args:
            label: The label.

        Returns:
            float: The maximum loss.
        """
        label_index = torch.where(self.label == label)[0]
        if label_index.size(0) == 0:
            return 1
        return self.parameter[label_index].max()

