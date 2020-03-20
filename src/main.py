import reservoir
import torch


def main():

    model = reservoir.ESN(1, 32, 1, reservoir.reset_weight_res)
    print(model(torch.Tensor([[1]])))


if __name__ == '__main__':
    main()
