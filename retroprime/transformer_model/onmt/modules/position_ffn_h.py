"""
Position feed-forward network from "Attention is All You Need"
"""

import torch.nn as nn

import onmt

from onmt.modules.hyperbolic import cLinear


class PositionwiseFeedForward_h(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, c, dropout=0.1):
        super(PositionwiseFeedForward_h, self).__init__()
        # self.w_1 = nn.Linear(d_model, d_ff)
        # self.w_2 = nn.Linear(d_ff, d_model)
        self.w_1 = cLinear(d_model, d_ff, c)
        self.w_2 = cLinear(d_ff, d_model, c)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x
