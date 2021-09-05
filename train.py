import torch

from config import PATH
from data_prep_utils import get_data_loaders
from model import optimizer, net, criterion
from utils import file_output

if __name__ == '__main__':
    train_loader, _ = get_data_loaders()

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                file_output('[{}, {:5}] loss: %{:.3f}'.format(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    file_output('Finished Training')
    torch.save(net.state_dict(), PATH)
