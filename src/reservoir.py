import torch
import torch.nn as nn
import torch.nn.init as init
import math

class ESN(nn.Module):
    def __init__(
        self,
        size_in,
        size_res,
        size_out,
        reset_weight_res
    ):
        """
        Args:
            size_in (int): size of input layer.
            size_res (int): size of reservor layer.
            size_out (int): size of output layer.
            reset_weight_res(callable): calleable object to reset
                reset weight of reservoir layer.
        """
        super(ESN, self).__init__()
        self.size_in = size_in
        self.size_res = size_res
        self.size_out = size_out
        self.weight_in = torch.Tensor(size_res, size_in)
        self.weight_res = torch.Tensor(size_res, size_res)
        self.bias = torch.Tensor(size_res, 1)
        self.state = torch.Tensor(size_res, 1)
        self.reset_weight_res = reset_weight_res
        self.reset_parameters()
        self.Linear = nn.Linear(size_res, size_out)

    def reset_parameters(self):
        """
        Reset this parameter of ESN.
        """
        self._reset_weight_in(self.weight_in)
        self.reset_weight_res(self.weight_res)
        self._reset_bias()
        self._reset_state()

    def _reset_weight_in(self, weight):
        """
        Reset weight of input layer.
        """
        init.kaiming_uniform_(weight, a=math.sqrt(5))

    def _reset_bias(self):
        """
        Reset bias of reservoir unit.
        """
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_in)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def _reset_state(self):
        """
        Reset state to zeros.
        """
        init.zeros_(self.state)

    def forward(self, x):
        """
        Args:
            x: input
        Math:
        `x(t + 1) = W_{in}u + W_{res}x(t) + b`
        Where:
            x(t + 1): State of next reservoir layer.
            W_{in}: weight_in
        """
        self.state = torch.tanh(
            self.weight_in @ x +
            self.weight_res @ self.state +
            self.bias
        )
        return self.Linear(self.state)


def reset_weight_res(weight_res):
    """
    Improvisation callable for argument of ESN.
    """
    init.kaiming_uniform_(weight_res, a=math.sqrt(5))

