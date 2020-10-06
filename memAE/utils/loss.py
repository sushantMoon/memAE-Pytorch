from __future__ import absolute_import, print_function

import torch
from torch import nn
from torch.nn import functional as F

class MemAELoss(nn.Module):
    def __init__(self, regularization_parameter=0.0002):
        super(MemAELoss, self).__init__()
        self.reg_param = regularization_parameter

    def forward(self, prediction, ground_truth, training=False, testing=False, validating=False):
        attention_weights = prediction['att']
        loss = None
        if training:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'])
            regularizer =  F.softmax(attention_weights, dim=1) * F.log_softmax(attention_weights, dim=1)
            loss += (-1.0 * self.reg_param * regularizer.sum())
        if validating:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'])
        if testing:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'], reduction='none')
        return loss