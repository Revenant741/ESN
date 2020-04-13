import torch
import torch.nn as nn
import torch.nn.init as init
import math
import random

class LeakyESN(nn.Module):
    def __init__(
        self,
        size_in,
        size_res,
        size_out,
        leaky
    ):
        """
        Args:
            size_in (int): 入力層のサイズ
            size_res (int): リザバー層のサイズ
            size_out (int): 出力層のサイズ
        """
        super(LeakyESN, self).__init__()
        self.size_in = size_in
        self.size_res = size_res
        self.size_out = size_out
        self.leaky = leaky
        self.register_buffer('weight_in', torch.Tensor(size_in, size_res))
        self.register_buffer('weight_res', torch.Tensor(size_res,size_res))
        self.register_buffer('bias', torch.Tensor(size_res))
        self.register_buffer('state', torch.Tensor(size_res))
        self.register_buffer('noise', torch.Tensor(size_res))
        self.reset_parameters()
        self.Linear = nn.Linear(size_res, size_out)

    def reset_parameters(self):
        self._reset_weight_in()
        self._reset_weight_res()
        self._reset_bias()
        self._reset_state()

    def _reset_weight_in(self):
        init.kaiming_uniform_(self.weight_in, a=math.sqrt(5))

    def _reset_weight_res(self):
        """
        ESNの第４引数のための即興の関数
        (とりあえずランダムでリセットする.)
        """
        adjency = torch.Tensor([random.randint(0, 1) for _ in range(self.size_res**2)])
        init.kaiming_uniform_(self.weight_res, a=math.sqrt(5))
        self.weight_res *= adjency.view(self.size_res, self.size_res)
        init.kaiming_uniform_(self.weight_res, a=math.sqrt(5))

    def _reset_bias(self):
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_in)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def _reset_state(self):
        init.zeros_(self.state)

    def _reset_noise(self):
        init.uniform_(self.noise, 0, 1e-3)

    def forward(self, x):
        """
        Math:
        ~x(t) = W_{in}u(t) + W_{res}x(t - 1) + b
        x(t) = tanh((1 - a) * x(t - 1) + a * ~x(t))
        y = W_{out}x(t)

        Where:
            ~x(t):省略のための出力
            x(t): リザバー層の状態
            W_{in}: 入力層の重み
            u(t): 入力データ
            W_{res}: リザバー層の重み
            x(t + 1): 一つ前のリザバー層
            b: リザバー層のバイアス
            a: leaky
            y: ESNの出力

        Tips:
            全結合層で引数の状態を転置しているのは
            (size_res x 1)の形でESNの状態を表しているが
            nn.Linearが引数で受け取るときは(1 x size_res)
            の形で要求しているためである.
        """
        tilde_x = x @ self.weight_in + self.state @ self.weight_res + self.bias
        self.state = torch.tanh((1 - self.leaky) * self.state + self.leaky * tilde_x + self.noise)
        return self.Linear(self.state)
