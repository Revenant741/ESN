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
            size_in (int): 入力層のサイズ
            size_res (int): リザバー層のサイズ
            size_out (int): 出力層のサイズ
            reset_weight_res (callable): リザバー層の重みを引数で受け取り,
                それをリセットする呼び出し可能なオブジェクトです.
                関数や__call__を実装したオブジェクトなどがそれに該当します.
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
        ESNのパラメータをリセットします.
        """
        self._reset_weight_in(self.weight_in)
        self.reset_weight_res(self.weight_res)
        self._reset_bias()
        self._reset_state()

    def _reset_weight_in(self, weight):
        """
        入力層の重みをリセットする.
        """
        init.kaiming_uniform_(weight, a=math.sqrt(5))

    def _reset_bias(self):
        """
        リザバー層のバイアスをリセット
        """
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_in)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def _reset_state(self):
        """
        状態をゼロでリセット
        """
        init.zeros_(self.state)

    def forward(self, x):
        """
        Math:
        x(t) = W_{in}u(t) + W_{res}x(t - 1) + b
        y = W_{out}x(t)

        Where:
            x(t): リザバー層の状態
            W_{in}: 入力層の重み
            u(t): 入力データ
            W_{res}: リザバー層の重み
            x(t + 1): 一つ前のリザバー層
            b: リザバー層のバイアス
            y: ESNの出力
        """
        self.state = torch.tanh(
            self.weight_in @ x +
            self.weight_res @ self.state +
            self.bias
        )
        return self.Linear(self.state.T)


def reset_weight_res(weight_res):
    """
    ESNの第４引数のための即興の関数
    (とりあえずランダムでリセット)
    """
    init.kaiming_uniform_(weight_res, a=math.sqrt(5))
