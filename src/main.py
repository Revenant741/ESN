import reservoir
import torch

DEVICE = 'cuda'

def main():

    model = reservoir.ESN(2, 32, 3, reservoir.reset_weight_res)
    model.to(DEVICE)
    print(model(torch.Tensor([1, 2]).to(DEVICE)))
    print(model(torch.Tensor([3, 4]).to(DEVICE)))


if __name__ == '__main__':
    main()
