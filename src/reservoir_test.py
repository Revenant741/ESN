import reservoir as res

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

def add_arguments(parser):
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--download_mnist', type=bool, default=True, help='True when you download mnist')
    parser.add_argument('--res_size', type=int, default=1024, help='size of reservoir unit')

def prepare_dataset(args):
    train_data = dsets.MNIST(
        root='./mnist/',
        train=True,
        transform=transforms.ToTensor(),
        download=args.download_mnist
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_data = dsets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.ToTensor()
    )

    test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
    test_x = test_x.to(args.device)
    test_y = test_data.test_labels.numpy()[:2000]

    return train_loader, test_x, test_y

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        self.esn = res.ESN(
            size_in=28,
            size_res=args.res_size,
            size_out=10,
        )

    def forward(self, x):
        """
        Args:
            x(tuple): shape (batch_size, time_size, input_size)
        """

        batch_size, time_size, _ = x.shape
        
        out = torch.Tensor(batch_size, self.esn.size_out).to(x.device)

        for batch_i, batch in enumerate(x):
            for inputs in batch[:-1]:
                self.esn(inputs)
            out[batch_i] = self.esn(batch[-1])
        return out

def train(args, model, train_loader, test_x, test_y):
    model.to(args.device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    train_loader, test_x, test_y = prepare_dataset(args)

    epochs = []
    accuracys = []
    for epoch in range(args.epoch):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(args.device)
            b_y = b_y.to(args.device)

            b_x = b_x.view(-1, 28, 28).to(args.device)
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            accuracy = float(
                (pred_y == test_y).astype(int).sum()) / float(test_y.size)
            epoch_str = f'Epoch: {epoch}'
            train_loss_str = f'train loss: {loss.data.cpu().numpy():.4f}'
            test_accuracy_str = f'test_accuracy: {accuracy:.2f}'
            print(f'{epoch_str} | {train_loss_str} | {test_accuracy_str}')
            epochs.append(epoch + 1)
            accuracys.append(accuracy)

    return epochs, accuracys

def train_to_image(epochs, acuuracys):
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, accuracys)
    plt.savefig('image/esn-mnist-100epoch.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    
    train_loader, test_x, test_y = prepare_dataset(args)

    epochs, accuracys = train(args, RNN(args), train_loader, test_x, test_y)

    train_to_image(epochs, accuracys)
