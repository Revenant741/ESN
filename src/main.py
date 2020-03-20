import reservoir
import torch


def main():
    model = reservoir.ESN(1, 32, 1, reservoir.reset_weight_res)
    out = model(torch.Tensor([1]))
    print(out)


if __name__ == '__main__':
    main()
