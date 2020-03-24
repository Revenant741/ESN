import reservoir
import torch


def main():

    model = reservoir.ESN(2, 32, 3, reservoir.reset_weight_res)
    print(model(torch.Tensor([1, 2])))
    print(model(torch.Tensor([3, 4])))


if __name__ == '__main__':
    main()
