import reservoir as res

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EPOCH = 20
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.001
DOWNLOAD_MNIST = True

train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = dsets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.ToTensor()
)

test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = res.ESN(
            size_in=INPUT_SIZE,
            size_res=128,
            size_out=10,
            reset_weight_res=res.reset_weight_res,
        )

    def forward(self, x):
        """
        Args:
            x(tuple): shape (batch_size, time_step, input_size)
        """
        # x shape (batch, time_step, input_size)

        out_list = []
        for batch in x:
            for input in batch[:-1]:
                self.rnn(input)
            out_list.append(self.rnn(batch[-1]))
        return torch.stack(out_list)


model = RNN()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


epochs = []
accuracys = []
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)
        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float(
                (pred_y == test_y).astype(int).sum()) / float(test_y.size)
            epoch_str = f'Epoch: {epoch}'
            train_loss_str = f'train loss: {loss.data.numpy():.4f}'
            test_accuracy_str = f'test_accuracy: {accuracy:.2f}'
            print(f'{epoch_str} | {train_loss_str} | {test_accuracy_str}')
            break
    test_output = model(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    accuracy = float(
        (pred_y == test_y).astype(int).sum()) / float(test_y.size)
    epoch_str = f'Epoch: {epoch}'
    train_loss_str = f'train loss: {loss.data.numpy():.4f}'
    test_accuracy_str = f'test_accuracy: {accuracy:.2f}'
    print(f'{epoch_str} | {train_loss_str} | {test_accuracy_str}')
    epochs.append(epoch + 1)
    accuracys.append(accuracy)

test_output = model(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(f'{pred_y} prediction number')
print(f'{test_y[:10]} real number')
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, accuracys)
plt.savefig('figure.png')
